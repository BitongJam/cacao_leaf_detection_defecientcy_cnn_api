import io
import os
import cv2
import time
import json
import base64
import serial
import asyncio
import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    WebSocket
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    StreamingResponse
)

from pydantic import BaseModel

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = "detected_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD CNN MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_model.keras"

MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "k",
    "n",
    "p",
    "healthy",
    "not_cacao"
]

print("✅ CNN MODEL LOADED")

# =========================================================
# GRAD-CAM HEATMAP FUNCTION
# =========================================================

def generate_heatmap(image, img_batch, predicted_idx):
    """
    Generate Grad-CAM heatmap for the prediction.
    Returns the superimposed image as numpy array and base64 encoded string.
    """
    try:
        # Get the last convolutional layer
        try:
            target_layer = MODEL.get_layer("last_conv_layer")
        except:
            # Fallback to last Conv2D layer if named layer not found
            target_layer = [layer for layer in MODEL.layers if "conv" in layer.name.lower()][-1]

        # Create gradient model
        grad_model = tf.keras.Model(
            inputs=MODEL.inputs,
            outputs=[target_layer.output, MODEL.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            loss = predictions[:, predicted_idx]

        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global Average Pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ tf.expand_dims(pooled_grads, -1)
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        # Convert image to numpy
        img_np = np.array(image)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap_np, (img_np.shape[1], img_np.shape[0]))
        
        # Apply colormap
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(np.ascontiguousarray(heatmap_color), cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Blend images (60% original, 40% heatmap)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color_rgb, 0.4, 0)

        # Convert back to BGR for saving
        final_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', final_bgr)
        heatmap_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        return final_bgr, heatmap_base64

    except Exception as e:
        print(f"❌ HEATMAP ERROR: {str(e)}")
        return None, None

# =========================================================
# SERIAL SETUP (NPK SENSOR)
# =========================================================

ser = serial.Serial(
    '/dev/serial0',
    baudrate=9600,
    timeout=1
)

npk_query = bytearray([
    0x01,
    0x03,
    0x00,
    0x1e,
    0x00,
    0x03,
    0x65,
    0xcd
])

# =========================================================
# GLOBALS
# =========================================================

clients = set()

# =========================================================
# SENSOR MODEL
# =========================================================

class NPKData(BaseModel):
    n: int
    p: int
    k: int
    time: str

latest_sensor_data = NPKData(
    n=0,
    p=0,
    k=0,
    time=""
)

# =========================================================
# THRESHOLD SETTINGS
# =========================================================

LOW_THRESHOLD_nitrogine = 15
HIGH_THRESHOLD_nitrogine = 25

LOW_THRESHOLD_phosphorus = 10
HIGH_THRESHOLD_phosphorus = 20

LOW_THRESHOLD_potassium = 20
HIGH_THRESHOLD_potassium = 40

LOW_THRESHOLD = 20
HIGH_THRESHOLD = 80

# =========================================================
# NPK STATUS CHECKER
# =========================================================

def check_npk_status(sensor):

    n = sensor["n"]
    p = sensor["p"]
    k = sensor["k"]

    notifications = []
    recommendations = []

    # =====================================================
    # NITROGEN
    # =====================================================

    if n <  LOW_THRESHOLD_nitrogine:

        notifications.append(
            "LOW NITROGEN"
        )

        recommendations.append(
            "Apply Nitrogen fertilizer (Urea)"
        )

    elif n > HIGH_THRESHOLD_nitrogine:

        notifications.append(
            "HIGH NITROGEN"
        )

        recommendations.append(
            "Reduce Nitrogen fertilizer application"
        )

    else:

        notifications.append(
            "NORMAL NITROGEN"
        )

    # =====================================================
    # PHOSPHORUS
    # =====================================================

    if p < LOW_THRESHOLD_phosphorus:

        notifications.append(
            "LOW PHOSPHORUS"
        )

        recommendations.append(
            "Apply Phosphorus fertilizer"
        )

    elif p > HIGH_THRESHOLD_phosphorus:

        notifications.append(
            "HIGH PHOSPHORUS"
        )

        recommendations.append(
            "Reduce Phosphorus fertilizer"
        )

    else:

        notifications.append(
            "NORMAL PHOSPHORUS"
        )

    # =====================================================
    # POTASSIUM
    # =====================================================

    if k < LOW_THRESHOLD_potassium:

        notifications.append(
            "LOW POTASSIUM"
        )

        recommendations.append(
            "Apply Potassium fertilizer"
        )

    elif k > HIGH_THRESHOLD_potassium:

        notifications.append(
            "HIGH POTASSIUM"
        )

        recommendations.append(
            "Reduce Potassium fertilizer"
        )

    else:

        notifications.append(
            "NORMAL POTASSIUM"
        )

    # =====================================================
    # SOIL HEALTH
    # =====================================================

    if (
        LOW_THRESHOLD_nitrogine <= n <= HIGH_THRESHOLD_nitrogine and
        LOW_THRESHOLD_phosphorus <= p <= HIGH_THRESHOLD_phosphorus and
        LOW_THRESHOLD_potassium <= k <= HIGH_THRESHOLD_potassium
    ):

        soil_status = "SOIL HEALTHY"

    else:

        soil_status = "SOIL NEEDS ATTENTION"

    return {
        "notifications": notifications,
        "recommendations": recommendations,
        "soil_status": soil_status
    }

# =========================================================
# READ NPK SENSOR
# =========================================================

def read_npk_sensor():

    try:

        ser.write(npk_query)

        time.sleep(0.15)

        response = ser.read(11)

        print("RAW:", response.hex())

        if (
            len(response) == 11 and
            response[0] == 0x01 and
            response[1] == 0x03
        ):

            n = (response[3] << 8) | response[4]
            p = (response[5] << 8) | response[6]
            k = (response[7] << 8) | response[8]

            return {
                "n": n,
                "p": p,
                "k": k,
                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }

    except Exception as e:

        print("SENSOR ERROR:", e)

    return None

# =========================================================
# SENSOR LOOP
# =========================================================

async def sensor_loop():

    global latest_sensor_data

    while True:

        data = read_npk_sensor()

        if data:

            latest_sensor_data = NPKData(**data)

            threshold_result = check_npk_status(
                data
            )

            payload = {

                "sensor_data":
                    latest_sensor_data.dict(),

                "notifications":
                    threshold_result["notifications"],

                "recommendations":
                    threshold_result["recommendations"],

                "soil_status":
                    threshold_result["soil_status"]
            }

            print("LIVE:", payload)

            await broadcast(payload)

        await asyncio.sleep(2)

# =========================================================
# WEBSOCKET BROADCAST
# =========================================================

async def broadcast(data):

    disconnected = []

    for ws in clients:

        try:

            await ws.send_json(data)

        except:

            disconnected.append(ws)

    for ws in disconnected:

        clients.discard(ws)

# =========================================================
# APP LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    task = asyncio.create_task(
        sensor_loop()
    )

    yield

    task.cancel()

app.router.lifespan_context = lifespan

# =========================================================
# SENSOR API
# =========================================================

@app.get("/sensor")
def get_sensor():

    threshold_result = check_npk_status(
        latest_sensor_data.dict()
    )

    return {

        "sensor_data":
            latest_sensor_data.dict(),

        "notifications":
            threshold_result["notifications"],

        "recommendations":
            threshold_result["recommendations"],

        "soil_status":
            threshold_result["soil_status"]
    }

# =========================================================
# SENSOR STREAM
# =========================================================

@app.get("/sensor-stream")
async def sensor_stream():

    async def event_generator():

        while True:

            threshold_result = check_npk_status(
                latest_sensor_data.dict()
            )

            payload = {

                "sensor_data":
                    latest_sensor_data.dict(),

                "notifications":
                    threshold_result["notifications"],

                "recommendations":
                    threshold_result["recommendations"],

                "soil_status":
                    threshold_result["soil_status"]
            }

            yield f"data: {json.dumps(payload)}\n\n"

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# =========================================================
# WEBSOCKET
# =========================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    clients.add(websocket)

    try:

        while True:

            await asyncio.sleep(1)

    except:

        pass

    finally:

        clients.discard(websocket)

# =========================================================
# SENSOR SUPPORT
# =========================================================

def evaluate_sensor_support(predicted_class, sensor):

    n, p, k = sensor["n"], sensor["p"], sensor["k"]

    score = 0.5

    if predicted_class == "n":
        if n < 20:
            score = 0.75
        elif n > 40:
            score = 0.30
        else:
            score = 0.95

    elif predicted_class == "p":
        if p < 10:
            score = 0.75
        elif p > 20:
            score = 0.30
        else:
            score = 0.95

    elif predicted_class == "k":
        if k < 20:
            score = 0.75
        elif k > 40:
            score = 0.30
        else:
            score = 0.95

    elif predicted_class == "healthy":
        if n >= 10 and n <= 25 and p >= 10 and p <= 20 and k >= 20 and k <= 40:
            score = 0.95
        else:
            score = 0.40

    return score
# =========================================================
# HYBRID FUSION
# =========================================================

def hybrid_fusion(
    cnn_class,
    cnn_confidence,
    sensor_data
):

    sensor_support = evaluate_sensor_support(
        cnn_class,
        sensor_data
    )

    final_confidence = (
        (cnn_confidence * 0.7) +
        (sensor_support * 0.3)
    )

    # =====================================================
    # STATUS
    # =====================================================

    if final_confidence >= 0.85:

        status = "STRONG DETECTION"

    elif final_confidence >= 0.60:

        status = "MODERATE DETECTION"

    else:

        status = "WEAK DETECTION"

    return {

        "cnn_confidence":
            cnn_confidence,

        "sensor_support":
            sensor_support,

        "final_confidence":
            final_confidence,

        "status":
            status
    }

# =========================================================
# UNIFIED PREDICT (SINGLE & MULTIPLE IMAGES)
# =========================================================

@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...)
):
    """
    Unified endpoint that handles both single and multiple image predictions.
    - Single image: returns one prediction with heatmap
    - Multiple images: returns best_class with all predictions and heatmaps
    """

    try:

        results = []
        total_confidence = 0
        class_confidences = defaultdict(list)

        # =================================================
        # PROCESS EACH IMAGE
        # =================================================

        for file in files:

            file_bytes = await file.read()

            image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")

            img_resized = image.resize(
                (224, 224)
            )

            img_array = np.array(
                img_resized
            ).astype("float32")

            img_batch = tf.expand_dims(
                img_array,
                0
            )

            # =================================================
            # CNN PREDICTION
            # =================================================

            predictions = MODEL.predict(
                img_batch,
                verbose=0
            )

            predicted_idx = np.argmax(
                predictions[0]
            )

            predicted_class = CLASS_NAMES[
                predicted_idx
            ]

            cnn_confidence = float(
                np.max(predictions[0])
            )

            # =================================================
            # HEATMAP GENERATION
            # =================================================

            heatmap_bgr, heatmap_base64 = generate_heatmap(
                image,
                img_batch,
                predicted_idx
            )

            heatmap_filename = None
            if heatmap_bgr is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                label = CLASS_NAMES[predicted_idx].replace(" ", "_")
                heatmap_filename = f"{timestamp}_{label}.jpg"
                save_path = os.path.join(OUTPUT_DIR, heatmap_filename)
                cv2.imwrite(save_path, heatmap_bgr)

            # =================================================
            # SENSOR DATA
            # =================================================

            sensor_data = latest_sensor_data.dict()

            threshold_result = check_npk_status(
                sensor_data
            )

            # =================================================
            # HYBRID FUSION
            # =================================================

            fusion_result = hybrid_fusion(
                predicted_class,
                cnn_confidence,
                sensor_data
            )

            # =================================================
            # STORE RESULT
            # =================================================

            result = {
                "success": True,

                # CNN
                "cnn_class":
                    predicted_class,

                "cnn_confidence":
                    fusion_result[
                        "cnn_confidence"
                    ],

                # HEATMAP
                "heatmap_filename":
                    heatmap_filename,

                "heatmap_base64":
                    heatmap_base64,

                # SENSOR
                "sensor_data":
                    sensor_data,

                "sensor_support":
                    fusion_result[
                        "sensor_support"
                    ],

                # HYBRID
                "final_confidence":
                    fusion_result[
                        "final_confidence"
                    ],

                "status":
                    fusion_result[
                        "status"
                    ],

                # THRESHOLD
                "soil_status":
                    threshold_result[
                        "soil_status"
                    ],

                "notifications":
                    threshold_result[
                        "notifications"
                    ],

                "recommendations":
                    threshold_result[
                        "recommendations"
                    ]
            }

            results.append(result)

            cls = result["cnn_class"]
            conf = result["final_confidence"]

            total_confidence += conf
            class_confidences[cls].append(conf)

        # =================================================
        # RETURN BASED ON IMAGE COUNT
        # =================================================

        if len(results) == 1:
            # Single image - return single prediction
            return results[0]

        else:
            # Multiple images - return best class summary + all details
            avg_confidence = (
                total_confidence / len(results)
            )

            best_class = max(
                class_confidences.items(),
                key=lambda x: (
                    sum(x[1]) / len(x[1])
                )
            )[0]

            return {
                "success": True,
                "best_class": best_class,
                "avg_confidence": avg_confidence,
                "total_images": len(results),
                "details": results
            }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# DOWNLOAD
# =========================================================

@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if not os.path.exists(file_path):

        return {
            "detail": "File not found"
        }

    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3000
    )