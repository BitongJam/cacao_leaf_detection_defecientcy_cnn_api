import io
import time
import asyncio
import base64
import numpy as np
import tensorflow as tf
import pywt
import cv2  # Gikinahanglan para sa heatmap processing ug colormap

from PIL import Image
from pathlib import Path
from datetime import datetime
from collections import Counter

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import random

# =========================================================
# APP SETUP
# =========================================================

# Sensor data storage
latest_sensor = {
    "n": 0,
    "p": 0,
    "k": 0,
    "timestamp": None
}

clients = set()

# =========================================================
# SENSOR THRESHOLDS & RECOMMENDATION
# =========================================================
# Define desired ranges for NPK (min, max). Values outside range will trigger recommendations.
SENSOR_THRESHOLDS = {
    "n": {"min": 20, "max": 60},
    "p": {"min": 15, "max": 50},
    "k": {"min": 10, "max": 40},
}

def compute_recommendation(sensor_values: dict) -> dict:
    """Return simple recommendations based on SENSOR_THRESHOLDS.

    Output example: {"recommendations": ["Increase N...", "P OK", ...]}.
    """
    recs = []
    for nutrient in ("n", "p", "k"):
        try:
            val = float(sensor_values.get(nutrient, 0))
        except Exception:
            recs.append(f"{nutrient.upper()}: unknown")
            continue

        th = SENSOR_THRESHOLDS.get(nutrient)
        if th is None:
            recs.append(f"{nutrient.upper()}: no threshold")
            continue

        if val < th["min"]:
            recs.append(f"Increase {nutrient.upper()} (current {val} < {th['min']})")
        elif val > th["max"]:
            recs.append(f"Decrease {nutrient.upper()} (current {val} > {th['max']})")
        else:
            recs.append(f"{nutrient.upper()} OK (in range)")

    # Fertilizer suggestions (general guidance)
    fert_recs = []
    if any("Increase N" in r for r in recs):
        fert_recs.append({
            "nutrient": "N",
            "suggestion": "Apply a nitrogen-rich fertilizer (e.g., urea 46-0-0 or ammonium nitrate). Start with 10-20 g/m^2, then re-check sensor after 1-2 weeks.",
        })
    if any("Decrease N" in r for r in recs):
        fert_recs.append({
            "nutrient": "N",
            "suggestion": "Avoid additional N fertilizer; consider cover crops or sorption strategies to reduce excess nitrogen.",
        })

    if any("Increase P" in r for r in recs):
        fert_recs.append({
            "nutrient": "P",
            "suggestion": "Use a phosphorus source such as triple superphosphate (0-46-0) or bone meal. Apply 10-30 g/m^2 if soil test confirms deficiency.",
        })
    if any("Decrease P" in r for r in recs):
        fert_recs.append({
            "nutrient": "P",
            "suggestion": "Avoid phosphorus additions; improve drainage and avoid P-containing manure until levels normalize.",
        })

    if any("Increase K" in r for r in recs):
        fert_recs.append({
            "nutrient": "K",
            "suggestion": "Apply potassium fertilizers such as muriate of potash (0-0-60) or potassium sulfate. Suggested starting rate 10-20 g/m^2.",
        })
    if any("Decrease K" in r for r in recs):
        fert_recs.append({
            "nutrient": "K",
            "suggestion": "Avoid K-containing fertilizers; flush or leach if water management permits.",
        })

    return {"recommendations": recs, "fertilizer_recommendations": fert_recs}

async def sensor_loop():
    """Background task to read NPK sensor values periodically"""
    global latest_sensor
    while True:
        try:
            # Try to read from real sensor
            from sensor import get_npk_values
            n, p, k = get_npk_values()
            if n is not None:
                latest_sensor = {
                    "n": n,
                    "p": p,
                    "k": k,
                    "timestamp": datetime.now().isoformat()
                }
                # compute recommendation based on thresholds
                latest_sensor["recommendation"] = compute_recommendation(latest_sensor)
                print(f"Sensor: N={n}, P={p}, K={k}")
        except Exception as e:
            # Fallback: simulate sensor data if real sensor fails
            print(f"Sensor read error: {e}, using simulated values")
            latest_sensor = {
                "n": random.randint(0, 100),
                "p": random.randint(0, 100),
                "k": random.randint(0, 100),
                "timestamp": datetime.now().isoformat()
            }
            latest_sensor["recommendation"] = compute_recommendation(latest_sensor)
        
        # Broadcast to WebSocket clients
        await broadcast_sensor_data()
        await asyncio.sleep(5)  # Read every 5 seconds

async def broadcast_sensor_data():
    """Send sensor data to all connected WebSocket clients"""
    disconnected = []
    for ws in clients:
        try:
            await ws.send_json({"sensor": latest_sensor})
        except:
            disconnected.append(ws)
    
    for ws in disconnected:
        clients.discard(ws)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    sensor_task = asyncio.create_task(sensor_loop())
    yield
    # Shutdown
    sensor_task.cancel()

app = FastAPI(title="Cacao Disease Classifier with Grad-CAM", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# CUSTOM LAYER (WAVELET TRANSFORM)
# =========================================================

@tf.keras.utils.register_keras_serializable()
class WaveletLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):

        def wavelet_transform(img):
            img = img.numpy()
            coeffs = []

            for channel in range(img.shape[-1]):
                cA, (cH, cV, cD) = pywt.dwt2(img[:, :, channel], 'haar')

                merged = np.concatenate([
                    cA,
                    cH,
                    cV,
                    cD
                ], axis=1)

                coeffs.append(merged)

            coeffs = np.stack(coeffs, axis=-1)
            return coeffs.astype(np.float32)

        output = tf.map_fn(
            lambda x: tf.py_function(
                wavelet_transform,
                [x],
                Tout=tf.float32
            ),
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(112, 448, 3), dtype=tf.float32)
        )

        return output
            
@tf.keras.utils.register_keras_serializable()
class WTResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):

        shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + shortcut

        x = self.relu(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config
    
    # =========================================================
# MODEL LOAD & CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "wt_resnet_cacao.keras"

print(f"Loading model from {MODEL_PATH}...")
try:
    MODEL = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "WaveletLayer": WaveletLayer,
            "WTResidualBlock": WTResidualBlock,
        },
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    raise e

# Awtomatiko nga pangitaon ang kataposang convolutional layer para sa Grad-CAM.
# Importante kini tungod kay ang ResNet adunay espesipikong ngalan sa mga layer.
LAST_CONV_LAYER_NAME = None
for layer in reversed(MODEL.layers):
    # Mahimong anaa kini sa sulod sa usa ka functional sub-model (sama sa ResNet backbone)
    if isinstance(layer, tf.keras.Model):
        for sub_layer in reversed(layer.layers):
            if isinstance(sub_layer, tf.keras.layers.Conv2D) or "conv" in sub_layer.name.lower():
                LAST_CONV_LAYER_NAME = f"{layer.name}.{sub_layer.name}"  # Path notation if nested
                # Tipigi usab ang reference sa sulod nga layer para sa Grad-CAM model construction
                GRAD_CAM_TARGET_LAYER = sub_layer
                GRAD_CAM_MODEL_BACKBONE = layer
                break
        if LAST_CONV_LAYER_NAME:
            break
    elif isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
        LAST_CONV_LAYER_NAME = layer.name
        GRAD_CAM_TARGET_LAYER = layer
        GRAD_CAM_MODEL_BACKBONE = MODEL
        break

print(f"Target layer selected for Heatmap: {LAST_CONV_LAYER_NAME}")

# Warm-up (Gi-execute kausa aron paspas ang mosunod nga mga request sa Raspberry Pi)
print("Warming up model...")
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = MODEL.predict(dummy, verbose=0)
print("Warm-up complete!")

CLASS_NAMES = ["n", "p", "k", "healthy", "not_cacao"]
CONF_THRESHOLD = 0.70

# Lock aron malikayan ang race conditions ug memory spike sa Raspberry Pi
prediction_lock = asyncio.Lock()

# =========================================================
# GRAD-CAM HEATMAP GENERATOR
# =========================================================

def generate_heatmap(img_array, model, pred_index=None):
    """
    Nag-compute sa Grad-CAM activation map ug mobalik og base64 image string.
    """
    try:
        # Pagtukod og gradient model base kung asa nakit-an ang kataposang conv layer
        if "GRAD_CAM_MODEL_BACKBONE" in globals() and GRAD_CAM_MODEL_BACKBONE != model:
            # Kon ang conv layer naa sa sulod sa usa ka sub-model (e.g. ResNet backbone)
            grad_model = tf.keras.models.Model(
                inputs=[GRAD_CAM_MODEL_BACKBONE.inputs],
                outputs=[GRAD_CAM_TARGET_LAYER.output, GRAD_CAM_MODEL_BACKBONE.output]
            )
            
            # Kinahanglan ipasa una ang imahe sa mga layer sa wala pa ang backbone kon duna man
            # Alang sa kayano, kung ang modelo diretso ra, gamiton nato ang nag-unang lohika:
        else:
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
            )

        # Pagkuha sa gradients gamit ang GradientTape
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Gradients sa target class counter-bahin sa conv layer feature map
        grads = tape.gradient(class_channel, conv_outputs)

        # I-average ang gradients matag channel (Global Average Pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # I-multiply ang feature map sa iyang ka-importante
        conv_outputs = conv_outputs[0]
        pooled_grads = tf.expand_dims(pooled_grads, axis=-1)
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # ReLU ug Normalization
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        # Pag-andam sa orihinal nga imahe (I-convert gikan sa float input format ngadto sa uint8)
        orig_img = img_array[0].astype(np.uint8)
        
        # I-resize ang heatmap aron maparehas sa sukod sa input image (224x224)
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        
        # I-convert ang heatmap ngadto sa usa ka Color Map (JET color map para sa rainbow-effect)
        heatmap_uint8 = np.ascontiguousarray(np.uint8(255 * heatmap_resized))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # I-overlay ang heatmap ngadto sa orihinal nga hulagway (60% orig, 40% heatmap)
        superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

        # I-encode ang imahe ngadto sa Base64 string para dali gamiton sa Frontend
        pil_img = Image.fromarray(superimposed_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{img_str}"
        
    except Exception as e:
        print(f"Grad-CAM failed: {e}. Returning None for heatmap.")
        return None
    
    # =========================================================
# PREDICTION CORE FUNCTION
# =========================================================

def predict_image(image: Image.Image):

    import pywt

    image = image.resize((224, 224))
    arr = np.array(image).astype("float32")

    # normalize if training used it
    # arr = arr / 255.0

    def wavelet_transform(img):
        coeffs = []
        for channel in range(img.shape[-1]):
            cA, (cH, cV, cD) = pywt.dwt2(img[:, :, channel], 'haar')
            merged = np.concatenate([cA, cH, cV, cD], axis=1)
            coeffs.append(merged)
        return np.stack(coeffs, axis=-1)

    arr = wavelet_transform(arr)

    print("DEBUG SHAPE:", arr.shape)  # Should be (112, 448, 3)

    batch = np.expand_dims(arr, 0)

    preds = MODEL.predict(batch, verbose=0)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]))

    cls = CLASS_NAMES[idx]
    if conf < CONF_THRESHOLD:
        cls = "uncertain"

    heatmap = None
    if LAST_CONV_LAYER_NAME:
        heatmap = generate_heatmap(batch, MODEL, idx)

    return cls, conf, heatmap

# =========================================================
# API ENDPOINTS
# =========================================================
@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Walay file nga nadawat.")

    results = []
    predictions = []

    # 🌱 Agriculture scoring system (confidence-weighted)
    npk_scores = {"n": 0.0, "p": 0.0, "k": 0.0}

    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")

            async with prediction_lock:
                cls, conf, heatmap_data = predict_image(img)

            predictions.append(cls)

            # store per image result
            results.append({
                "filename": file.filename,
                "cnn_class": cls,
                "cnn_confidence": round(conf, 4),
                "heatmap": heatmap_data
            })

            # 🌱 weighted scoring (only NPK classes)
            if cls in ["n", "p", "k"]:
                npk_scores[cls] += conf

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    # =====================================================
    # FILTER VALID PREDICTIONS
    # =====================================================
    valid_predictions = [p for p in predictions if p in ["n", "p", "k"]]

    # =====================================================
    # NORMALIZE SCORES (0–1)
    # =====================================================
    total = sum(npk_scores.values())

    if total > 0:
        normalized = {k: v / total for k, v in npk_scores.items()}
    else:
        normalized = npk_scores

    # =====================================================
    # CONVERT TO PERCENTAGE (for farmers)
    # =====================================================
    percent = {
        k.upper(): round(v * 100, 2) for k, v in normalized.items()
    }

    # =====================================================
    # SMART AGRICULTURE DECISION LOGIC
    # =====================================================
    values = normalized
    max_val = max(values.values()) if values else 0
    min_val = min(values.values()) if values else 0

    primary = max(values, key=values.get).upper() if values else None

    low_nutrients = [
        k.upper() for k, v in values.items() if v < 0.3
    ]

    # =====================================================
    # FINAL DECISION (AGRICULTURE-GRADE)
    # =====================================================
    if len(valid_predictions) == 0:
        final_decision = {
            "status": "no_valid_leaf_detected",
            "message": "No valid plant leaf detected",
            "npk_levels": percent
        }

    elif max_val - min_val < 0.15:
        final_decision = {
            "status": "mixed_deficiency",
            "message": "Multiple nutrient imbalance detected",
            "npk_levels": percent
        }

    else:
        final_decision = {
            "status": f"{primary}_deficiency",
            "message": f"{primary} is the most deficient nutrient",

            "npk_levels": percent,

            "details": {
                "primary_deficiency": {
                    "nutrient": primary,
                    "score": round(values[primary.lower()], 4),
                    "percentage": percent[primary]
                },

                "low_nutrients": [
                    {
                        "nutrient": n,
                        "percentage": percent[n]
                    }
                    for n in low_nutrients
                ]
            }
        }

    # =====================================================
    # OVERALL CLASS (kept from your system)
    # =====================================================
    if valid_predictions:
        overall = Counter(valid_predictions).most_common(1)[0][0]
    else:
        overall = "unknown"

    # =====================================================
    # FINAL RESPONSE
    # =====================================================
    return {
        "count": len(results),

        # 🌱 main agriculture output
        "npk_distribution_percent": percent,

        "raw_scores": npk_scores,

        "valid_count": len(valid_predictions),

        "final_decision": final_decision,

        "overall_diagnosis": overall,

        "results": results
    }

# @app.post("/predict")
# async def predict(files: list[UploadFile] = File(...)):
#     if not files:
#         raise HTTPException(status_code=400, detail="Walay file nga nadawat.")

#     results = []
#     predictions = []

#     for file in files:
#         try:
#             contents = await file.read()
#             # Pwersahon nga RGB aron malikayan ang PNG alpha-channel error (4 channels)
#             img = Image.open(io.BytesIO(contents)).convert("RGB")

#             # Paggamit sa lock para sa dungan nga mga request (luwas sa memory crash)
#             async with prediction_lock:
#                 cls, conf, heatmap_data = predict_image(img)

#             predictions.append(cls)

#             results.append({
#                 "filename": file.filename,
#                 "cnn_class": cls,
#                 "cnn_confidence": round(conf, 4),
#                 "heatmap": heatmap_data  # Base64 string data URI
#             })
#         except Exception as e:
#             results.append({
#                 "filename": file.filename,
#                 "error": f"Dili maproseso ang imahe: {str(e)}"
#             })

#     # Majority voting logic para sa kinatibuk-ang diagnosis
#     valid_predictions = [p for p in predictions if p != "uncertain"]
#     if valid_predictions:
#         overall = Counter(valid_predictions).most_common(1)[0][0]
#     elif predictions:
#         overall = Counter(predictions).most_common(1)[0][0]
#     else:
#         overall = "unknown"

#     return {
#         "count": len(results),
#         "overall_diagnosis": overall,
#         "results": results
#     }

# =========================================================
# SENSOR & WEBSOCKET PLACEHOLDERS
# =========================================================

# =========================================================
# SENSOR & WEBSOCKET
# =========================================================

@app.get("/sensor")
def sensor():
    # Return sensor values together with recommendation
    return {"sensor": latest_sensor}


@app.get("/recommendation")
def recommendation():
    """Return only the latest recommendation computed from sensor values."""
    return compute_recommendation(latest_sensor)

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # Send latest sensor data to client
            await websocket.send_json({"sensor": latest_sensor})
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        clients.discard(websocket)

# =========================================================
# RUN SERVER (UVICORN ENTRYPOINT)
# =========================================================
if __name__ == "__main__":
    import uvicorn
    # Mahimo nimo usbon ang host/port base sa configuration sa imong network
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=False)