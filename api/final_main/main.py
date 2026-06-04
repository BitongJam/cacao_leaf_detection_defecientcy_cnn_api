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
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# =========================================================
# APP
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
# MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.keras"

MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["n", "p", "k", "healthy", "not_cacao"]

# =========================================================
# SENSOR
# =========================================================

ser = serial.Serial('/dev/serial0', 9600, timeout=1)

npk_query = bytearray([0x01,0x03,0x00,0x1e,0x00,0x03,0x65,0xcd])

latest_sensor = {"n": 0, "p": 0, "k": 0, "time": ""}

clients = set()

# =========================================================
# THRESHOLDS (KEEPED)
# =========================================================

LOW_N, HIGH_N = 15, 25
LOW_P, HIGH_P = 10, 20
LOW_K, HIGH_K = 20, 40

# =========================================================
# SENSOR READ
# =========================================================

def read_sensor():
    try:
        ser.write(npk_query)
        time.sleep(0.15)
        res = ser.read(11)

        if len(res) == 11:
            return {
                "n": (res[3]<<8)|res[4],
                "p": (res[5]<<8)|res[6],
                "k": (res[7]<<8)|res[8],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except:
        pass
    return None

# =========================================================
# THRESHOLD CHECK ONLY (NO FUSION)
# =========================================================

def check_threshold(sensor):
    n, p, k = sensor["n"], sensor["p"], sensor["k"]

    notifications = []

    notifications.append(
        "LOW N" if n < LOW_N else "HIGH N" if n > HIGH_N else "NORMAL N"
    )
    notifications.append(
        "LOW P" if p < LOW_P else "HIGH P" if p > HIGH_P else "NORMAL P"
    )
    notifications.append(
        "LOW K" if k < LOW_K else "HIGH K" if k > HIGH_K else "NORMAL K"
    )

    soil_status = (
        "SOIL HEALTHY"
        if (LOW_N<=n<=HIGH_N and LOW_P<=p<=HIGH_P and LOW_K<=k<=HIGH_K)
        else "SOIL NEEDS ATTENTION"
    )

    return {
        "notifications": notifications,
        "soil_status": soil_status
    }

# =========================================================
# SENSOR LOOP + WEBSOCKET
# =========================================================

async def broadcast(data):
    dead = []
    for ws in clients:
        try:
            await ws.send_json(data)
        except:
            dead.append(ws)

    for d in dead:
        clients.discard(d)


async def sensor_loop():
    global latest_sensor

    while True:
        data = read_sensor()
        if data:
            latest_sensor = data

            result = check_threshold(data)

            await broadcast({
                "sensor": latest_sensor,
                "soil_status": result["soil_status"],
                "notifications": result["notifications"]
            })

        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sensor_loop())
    yield
    task.cancel()

app.router.lifespan_context = lifespan

# =========================================================
# WEBSOCKET
# =========================================================

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        clients.discard(websocket)

# =========================================================
# HEATMAP
# =========================================================

def generate_heatmap(image, img_batch, predicted_idx):
    try:
        layer = MODEL.get_layer("last_conv_layer")

        grad_model = tf.keras.Model(
            MODEL.inputs,
            [layer.output, MODEL.output]
        )

        with tf.GradientTape() as tape:
            conv, preds = grad_model(img_batch)
            loss = preds[:, predicted_idx]

        grads = tape.gradient(loss, conv)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))

        conv = conv[0]
        heatmap = conv @ tf.expand_dims(pooled, -1)
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.math.reduce_max(heatmap)

        heatmap = cv2.resize(heatmap.numpy(),
                              (image.size[0], image.size[1]))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img = np.array(image)
        out = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        _, buf = cv2.imencode(".jpg", out)
        return out, base64.b64encode(buf).decode()

    except:
        return None, None

# =========================================================
# CNN ONLY PREDICT
# =========================================================

def predict_image(image):
    img = image.resize((224,224))
    arr = np.array(img).astype("float32")
    batch = tf.expand_dims(arr, 0)

    preds = MODEL.predict(batch, verbose=0)

    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]))

    return CLASS_NAMES[idx], conf, batch, idx

# =========================================================
# PREDICT API
# =========================================================

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):

    results = []

    sensor = latest_sensor
    threshold = check_threshold(sensor)

    for file in files:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")

        cls, conf, batch, idx = predict_image(img)

        heatmap, _ = generate_heatmap(img, batch, idx)

        results.append({
            "cnn_class": cls,
            "cnn_confidence": conf,
            "sensor": sensor,
            "soil_status": threshold["soil_status"],
            "notifications": threshold["notifications"]
        })

    return results[0] if len(results) == 1 else {"results": results}

# =========================================================
# SENSOR API
# =========================================================

@app.get("/sensor")
def sensor_api():
    return {
        "sensor": latest_sensor,
        "status": check_threshold(latest_sensor)
    }

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)