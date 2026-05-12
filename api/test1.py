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
from collections import Counter
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
# MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.keras"

MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["k", "n", "p", "healthy", "not_cacao"]

print("✅ MODEL LOADED")

# =========================================================
# SENSOR
# =========================================================

ser = serial.Serial('/dev/serial0', 9600, timeout=1)

npk_query = bytearray([0x01, 0x03, 0x00, 0x1e, 0x00, 0x03, 0x65, 0xcd])

# =========================================================
# SENSOR DATA
# =========================================================

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

    if n < LOW_N:
        notifications.append("LOW NITROGEN")
        recommendations.append("Apply Nitrogen fertilizer")
    elif n > HIGH_N:
        notifications.append("HIGH NITROGEN")
        recommendations.append("Reduce Nitrogen fertilizer")
    else:
        notifications.append("NORMAL NITROGEN")

    if p < LOW_P:
        notifications.append("LOW PHOSPHORUS")
        recommendations.append("Apply Phosphorus fertilizer")
    elif p > HIGH_P:
        notifications.append("HIGH PHOSPHORUS")
        recommendations.append("Reduce Phosphorus fertilizer")
    else:
        notifications.append("NORMAL PHOSPHORUS")

    if k < LOW_K:
        notifications.append("LOW POTASSIUM")
        recommendations.append("Apply Potassium fertilizer")
    elif k > HIGH_K:
        notifications.append("HIGH POTASSIUM")
        recommendations.append("Reduce Potassium fertilizer")
    else:
        notifications.append("NORMAL POTASSIUM")

    soil_status = (
        "SOIL HEALTHY"
        if (LOW_N <= n <= HIGH_N and LOW_P <= p <= HIGH_P and LOW_K <= k <= HIGH_K)
        else "SOIL NEEDS ATTENTION"
    )

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
# SENSOR SUPPORT
# =========================================================

def evaluate_sensor_support(cls, sensor):

    n, p, k = sensor["n"], sensor["p"], sensor["k"]

    if cls == "n":
        return 0.95 if n < 25 else 0.3
    if cls == "p":
        return 0.95 if p < 20 else 0.3
    if cls == "k":
        return 0.95 if k < 40 else 0.3
    if cls == "healthy":
        return 0.95 if (n > 25 and p > 20 and k > 40) else 0.4

    return 0.5

# =========================================================
# HYBRID FUSION
# =========================================================

def hybrid_fusion(cls, conf, sensor):

    support = evaluate_sensor_support(cls, sensor)

    final = (conf * 0.7) + (support * 0.3)

    status = (
        "STRONG DETECTION" if final >= 0.85 else
        "MODERATE DETECTION" if final >= 0.60 else
        "WEAK DETECTION"
    )

    return {
        "sensor_support": support,
        "final_confidence": final,
        "status": status
    }

# =========================================================
# GRAD-CAM
# =========================================================

def generate_heatmap(model, img_array, layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, pred = grad_model(img_array)
        loss = tf.reduce_max(pred)

    grads = tape.gradient(loss, conv)

    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv = conv[0]

    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# =========================================================
# OVERLAY
# =========================================================

def overlay(img, heatmap):

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# =========================================================
# MAIN MULTI IMAGE PREDICT + HEATMAP
# =========================================================

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):

    predictions = []
    confidences = []

    best_img = None
    best_array = None
    best_conf = 0

    # =====================================================
    # LOOP IMAGES
    # =====================================================

    for file in files:

        img_bytes = await file.read()

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))

        arr = np.array(image).astype("float32")
        batch = tf.expand_dims(arr, 0)

        pred = MODEL.predict(batch, verbose=0)

        idx = np.argmax(pred[0])
        cls = CLASS_NAMES[idx]
        conf = float(np.max(pred[0]))

        predictions.append(cls)
        confidences.append(conf)

        if conf > best_conf:
            best_conf = conf
            best_img = image
            best_array = batch

    # =====================================================
    # FINAL CLASS (MAJORITY VOTE)
    # =====================================================

    final_class = Counter(predictions).most_common(1)[0][0]
    avg_conf = sum(confidences) / len(confidences)

    # =====================================================
    # SENSOR
    # =====================================================

    sensor = latest_sensor_data.dict()
    threshold = check_npk_status(sensor)

    # =====================================================
    # HYBRID
    # =====================================================

    fusion = hybrid_fusion(final_class, avg_conf, sensor)

    # =====================================================
    # HEATMAP (BEST IMAGE ONLY)
    # =====================================================

    layer_name = "conv2d"  # CHANGE based on your model

    heatmap = generate_heatmap(MODEL, best_array, layer_name)

    heatmap_img = overlay(np.array(best_img), heatmap)

    _, buffer = cv2.imencode(".jpg", heatmap_img)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "success": True,

        "final_class": final_class,
        "cnn_confidence": avg_conf,

        "sensor_data": sensor,

        "sensor_support": fusion["sensor_support"],
        "final_confidence": fusion["final_confidence"],
        "status": fusion["status"],

        "soil_status": threshold["soil_status"],
        "notifications": threshold["notifications"],
        "recommendations": threshold["recommendations"],

        "heatmap": heatmap_base64
    }

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)