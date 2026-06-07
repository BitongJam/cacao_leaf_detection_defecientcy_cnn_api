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

latest_sensor = {
    "n": 0,
    "p": 0,
    "k": 0,
    "timestamp": None,
    "recommendation": None
}

clients = set()

SENSOR_THRESHOLDS = {
    "n": {"min": 20, "max": 60},
    "p": {"min": 15, "max": 50},
    "k": {"min": 10, "max": 40},
}

def compute_recommendation(sensor_values: dict) -> dict:
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
    global latest_sensor
    while True:
        try:
            from sensor import get_npk_values
            n, p, k = get_npk_values()
            if n is not None:
                latest_sensor = {
                    "n": n,
                    "p": p,
                    "k": k,
                    "timestamp": datetime.now().isoformat()
                }
                latest_sensor["recommendation"] = compute_recommendation(latest_sensor)
                print(f"Sensor: N={n}, P={p}, K={k}")
        except Exception as e:
            latest_sensor = {
                "n": random.randint(0, 100),
                "p": random.randint(0, 100),
                "k": random.randint(0, 100),
                "timestamp": datetime.now().isoformat()
            }
            latest_sensor["recommendation"] = compute_recommendation(latest_sensor)
        
        await broadcast_sensor_data()
        await asyncio.sleep(5)

async def broadcast_sensor_data():
    disconnected = []
    for ws in list(clients):
        try:
            await ws.send_json({"sensor": latest_sensor})
        except:
            disconnected.append(ws)
    
    for ws in disconnected:
        clients.discard(ws)

@asynccontextmanager
async def lifespan(app: FastAPI):
    sensor_task = asyncio.create_task(sensor_loop())
    yield
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

    def wavelet_np(self, img):
        coeffs = []
        for c in range(img.shape[-1]):
            cA, (cH, cV, cD) = pywt.dwt2(img[:, :, c], "haar")
            merged = np.concatenate([cA, cH, cV, cD], axis=1)
            coeffs.append(merged)
        out = np.stack(coeffs, axis=-1).astype(np.float32)
        return out

    def call(self, inputs):
        def fn(x):
            out = tf.py_function(
                func=self.wavelet_np,
                inp=[x],
                Tout=tf.float32
            )
            out.set_shape([112, 448, 3])
            return out

        return tf.map_fn(
            fn,
            inputs,
            fn_output_signature=tf.TensorSpec(
                shape=(112, 448, 3),
                dtype=tf.float32
            )
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 112, 448, 3)

    def get_config(self):
        return super().get_config()
            
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

MODEL_PATH = Path("/home/pi/Documents/cacao_project/models/wt_resnet_cacao.keras")

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

# Dynamic activation parsing for Grad-CAM
LAST_CONV_LAYER_NAME = None
GRAD_CAM_TARGET_LAYER = None
GRAD_CAM_MODEL_BACKBONE = MODEL

for layer in reversed(MODEL.layers):
    if isinstance(layer, tf.keras.Model):
        for sub_layer in reversed(layer.layers):
            if isinstance(sub_layer, tf.keras.layers.Conv2D) or "conv" in sub_layer.name.lower():
                LAST_CONV_LAYER_NAME = sub_layer.name
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

try:
    # Ang husto nga input visualization base sa imong predict_image function size
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = MODEL(dummy, training=False)
    print("Warm-up complete!")
except Exception as e:
    print("Warm-up skipped:", e)

CLASS_NAMES = ["n", "p", "k", "healthy", "not_cacao"]
CONF_THRESHOLD = 0.70

prediction_lock = asyncio.Lock()

# =========================================================
# GRAD-CAM HEATMAP GENERATOR
# =========================================================

def generate_heatmap(img_array, model, pred_index=None):
    """
    Nagcompute sa husto nga Grad-CAM activation model bisan pa og functional sub-backbone ang gigamit.
    """
    try:
        # Pagtukod sa husto nga internal reference model aron malikayan ang Graph Disconnected framework errors
        if GRAD_CAM_MODEL_BACKBONE != model:
            # Kon ang target convolution layer naa sa sulod sa sub-model backbone
            grad_model = tf.keras.models.Model(
                inputs=[GRAD_CAM_MODEL_BACKBONE.inputs],
                outputs=[GRAD_CAM_TARGET_LAYER.output, GRAD_CAM_MODEL_BACKBONE.output]
            )
            # Ipasa ang image array latas sa mga unang structures sa pangunang modelo hangtod makasulod sa backbone
            # Para sa kasagaran nga architectures, i-extract ang internal state input:
            backbone_input = model.layers[1](img_array) if len(model.layers) > 1 else img_array
            # Kon ang input naay branching, i-extract ang layer outputs directly
            for l in model.layers:
                if l == GRAD_CAM_MODEL_BACKBONE:
                    break
                img_array = l(img_array)
            conv_img_input = img_array
        else:
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
            )
            conv_img_input = img_array

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(conv_img_input)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[np.newaxis, np.newaxis, :]
        
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        # Siguradohon nga uint8 ang configuration parameters
        orig_img = img_array[0].numpy() if tf.is_tensor(img_array) else img_array[0]
        orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)
        
        # I-resize ang heatmap pabalik sa configuration geometry structure sa original system image
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

        pil_img = Image.fromarray(superimposed_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{img_str}"
        
    except Exception as e:
        print(f"Grad-CAM execution failure matrix: {e}")
        return None

# =========================================================
# PREDICTION CORE FUNCTION
# =========================================================
def predict_image(image: Image.Image):
    image = image.resize((224, 224))
    arr = np.array(image).astype(np.float32)
    batch = np.expand_dims(arr, axis=0)

    print("INPUT SHAPE:", batch.shape)
    preds = MODEL.predict(batch, verbose=0)

    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]))
    cls = CLASS_NAMES[idx]

    if conf < CONF_THRESHOLD:
        cls = "uncertain"

    heatmap = None
    try:
        if LAST_CONV_LAYER_NAME:
            heatmap = generate_heatmap(batch, MODEL, idx)
    except Exception as e:
        print("Grad-CAM error hook:", e)

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
    npk_scores = {"n": 0.0, "p": 0.0, "k": 0.0}

    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")

            async with prediction_lock:
                cls, conf, heatmap_data = predict_image(img)

            predictions.append(cls)
            results.append({
                "filename": file.filename,
                "cnn_class": cls,
                "cnn_confidence": round(conf, 4),
                "heatmap": heatmap_data
            })

            if cls in ["n", "p", "k"]:
                npk_scores[cls] += conf

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    valid_predictions = [p for p in predictions if p in ["n", "p", "k"]]
    total = sum(npk_scores.values())
    normalized = {k: (v / total if total > 0 else 0.0) for k, v in npk_scores.items()}
    percent = {k.upper(): round(v * 100, 2) for k, v in normalized.items()}

    values = normalized
    max_val = max(values.values()) if values else 0
    min_val = min(values.values()) if values else 0

    primary = max(values, key=lambda k: values[k]).upper() if total > 0 else "N"
    low_nutrients = [k.upper() for k, v in values.items() if v < 0.3]

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
                    {"nutrient": n, "percentage": percent[n]} for n in low_nutrients
                ]
            }
        }

    overall = Counter(valid_predictions).most_common(1)[0][0] if valid_predictions else "unknown"

    return {
        "count": len(results),
        "npk_distribution_percent": percent,
        "raw_scores": npk_scores,
        "valid_count": len(valid_predictions),
        "final_decision": final_decision,
        "overall_diagnosis": overall,
        "results": results
    }

@app.get("/sensor")
def sensor():
    return {"sensor": latest_sensor}

@app.get("/recommendation")
def recommendation():
    return compute_recommendation(latest_sensor)

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.send_json({"sensor": latest_sensor})
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket execution exception tracking: {e}")
    finally:
        clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=False)