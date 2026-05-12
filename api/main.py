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
from collections import defaultdict, Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# OUTPUT DIR
# =========================================================

OUTPUT_DIR = "detected_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.keras"

MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["k", "n", "p", "healthy", "not_cacao"]

print("✅ CNN MODEL LOADED")

# =========================================================
# SERIAL SENSOR
# =========================================================

ser = serial.Serial(
    '/dev/serial0',
    baudrate=9600,
    timeout=1
)

npk_query = bytearray([
    0x01, 0x03, 0x00, 0x1e,
    0x00, 0x03, 0x65, 0xcd
])

# =========================================================
# GLOBALS
# =========================================================

clients = set()

class NPKData(BaseModel):
    n: int
    p: int
    k: int
    time: str

latest_sensor_data = NPKData(n=0, p=0, k=0, time="")

# =========================================================
# THRESHOLDS
# =========================================================

LOW_N, HIGH_N = 15, 25
LOW_P, HIGH_P = 10, 20
LOW_K, HIGH_K = 20, 40

# =========================================================
# SENSOR CHECK
# =========================================================

def check_npk_status(sensor):

    n, p, k = sensor["n"], sensor["p"], sensor["k"]

    notifications = []
    recommendations = []

    # N
    if n < LOW_N:
        notifications.append("LOW NITROGEN")
        recommendations.append("Apply Nitrogen fertilizer")
    elif n > HIGH_N:
        notifications.append("HIGH NITROGEN")
        recommendations.append("Reduce Nitrogen fertilizer")
    else:
        notifications.append("NORMAL NITROGEN")

    # P
    if p < LOW_P:
        notifications.append("LOW PHOSPHORUS")
        recommendations.append("Apply Phosphorus fertilizer")
    elif p > HIGH_P:
        notifications.append("HIGH PHOSPHORUS")
        recommendations.append("Reduce Phosphorus fertilizer")
    else:
        notifications.append("NORMAL PHOSPHORUS")

    # K
    if k < LOW_K:
        notifications.append("LOW POTASSIUM")
        recommendations.append("Apply Potassium fertilizer")
    elif k > HIGH_K:
        notifications.append("HIGH POTASSIUM")
        recommendations.append("Reduce Potassium fertilizer")
    else:
        notifications.append("NORMAL POTASSIUM")

    if (
        LOW_N <= n <= HIGH_N and
        LOW_P <= p <= HIGH_P and
        LOW_K <= k <= HIGH_K
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
# SENSOR READ
# =========================================================

def read_npk_sensor():

    try:
        ser.write(npk_query)
        time.sleep(0.15)

        response = ser.read(11)

        if len(response) == 11:

            n = (response[3] << 8) | response[4]
            p = (response[5] << 8) | response[6]
            k = (response[7] << 8) | response[8]

            return {
                "n": n,
                "p": p,
                "k": k,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

            result = check_npk_status(data)

            payload = {
                "sensor_data": latest_sensor_data.dict(),
                "notifications": result["notifications"],
                "recommendations": result["recommendations"],
                "soil_status": result["soil_status"]
            }

            await broadcast(payload)

        await asyncio.sleep(2)

# =========================================================
# BROADCAST
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
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    task = asyncio.create_task(sensor_loop())

    yield

    task.cancel()

app.router.lifespan_context = lifespan

# =========================================================
# SENSOR API
# =========================================================

@app.get("/sensor")
def get_sensor():

    result = check_npk_status(latest_sensor_data.dict())

    return {
        "sensor_data": latest_sensor_data.dict(),
        "notifications": result["notifications"],
        "recommendations": result["recommendations"],
        "soil_status": result["soil_status"]
    }

# =========================================================
# SENSOR STREAM
# =========================================================

@app.get("/sensor-stream")
async def sensor_stream():

    async def event_generator():

        while True:

            result = check_npk_status(latest_sensor_data.dict())

            payload = {
                "sensor_data": latest_sensor_data.dict(),
                "notifications": result["notifications"],
                "recommendations": result["recommendations"],
                "soil_status": result["soil_status"]
            }

            yield f"data: {json.dumps(payload)}\n\n"

            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# =========================================================
# WEBSOCKET
# =========================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    clients.add(websocket)

    try:
        while True:
            await asyncio.sleep(1)
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

def hybrid_fusion(cnn_class, cnn_confidence, sensor_data):

    sensor_support = evaluate_sensor_support(cnn_class, sensor_data)

    final_confidence = (cnn_confidence * 0.7) + (sensor_support * 0.3)

    if final_confidence >= 0.85:
        status = "STRONG DETECTION"
    elif final_confidence >= 0.60:
        status = "MODERATE DETECTION"
    else:
        status = "WEAK DETECTION"

    return {
        "cnn_confidence": cnn_confidence,
        "sensor_support": sensor_support,
        "final_confidence": final_confidence,
        "status": status
    }

# =========================================================
# MULTI-IMAGE PREDICT (MAIN ONLY)
# =========================================================

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):

    try:

        predictions = []
        confidences = []

        # =================================================
        # LOOP IMAGES
        # =================================================

        for file in files:

            img_bytes = await file.read()

            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image = image.resize((224, 224))

            img_array = np.array(image).astype("float32")
            img_batch = tf.expand_dims(img_array, 0)

            pred = MODEL.predict(img_batch, verbose=0)

            idx = np.argmax(pred[0])
            cls = CLASS_NAMES[idx]
            conf = float(np.max(pred[0]))

            predictions.append(cls)
            confidences.append(conf)


         # =================================================
        # GRAD-CAM
        # =================================================

        target_layer = MODEL.get_layer(
            "last_conv_layer"
        )

        grad_model = tf.keras.models.Model(
            [MODEL.inputs],
            [
                target_layer.output,
                MODEL.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, preds = grad_model(
                img_batch
            )

            loss = preds[:, idx]

        grads = tape.gradient(
            loss,
            conv_outputs
        )

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ tf.expand_dims(
            pooled_grads,
            -1
        )

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(
            heatmap,
            0
        ) / (
            tf.math.reduce_max(heatmap) + 1e-10
        )

        heatmap_np = heatmap.numpy()

        # =================================================
        # OVERLAY
        # =================================================

        img_np = np.array(image)

        heatmap_resized = cv2.resize(
            heatmap_np,
            (
                img_np.shape[1],
                img_np.shape[0]
            )
        )

        heatmap_color = np.uint8(
            255 * heatmap_resized
        )

        heatmap_color = cv2.applyColorMap(
            np.ascontiguousarray(heatmap_color),
            cv2.COLORMAP_JET
        )

        heatmap_color_rgb = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        superimposed_img = cv2.addWeighted(
            img_np,
            0.6,
            heatmap_color_rgb,
            0.4,
            0
        )

        # =================================================
        # SAVE IMAGE
        # =================================================

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = f"{timestamp}_{cls}.jpg"

        save_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        final_bgr = cv2.cvtColor(
            superimposed_img,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(save_path, final_bgr)

        # =================================================
        # BASE64
        # =================================================

        _, buffer = cv2.imencode(
            '.jpg',
            final_bgr
        )

        heatmap_base64 = base64.b64encode(
            buffer.tobytes()
        ).decode('utf-8')

        # =================================================
        # MAJORITY VOTE
        # =================================================

        final_class = Counter(predictions).most_common(1)[0][0]
        avg_conf = sum(confidences) / len(confidences)

        # =================================================
        # SENSOR + HYBRID
        # =================================================

        sensor_data = latest_sensor_data.dict()

        threshold = check_npk_status(sensor_data)

        fusion = hybrid_fusion(final_class, avg_conf, sensor_data)

        return {
            "success": True,
            "final_class": final_class,
            "cnn_confidence": avg_conf,
            "sensor_data": sensor_data,
            "sensor_support": fusion["sensor_support"],
            "final_confidence": fusion["final_confidence"],
            "status": fusion["status"],
            "soil_status": threshold["soil_status"],
            "notifications": threshold["notifications"],
            "recommendations": threshold["recommendations"]
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)