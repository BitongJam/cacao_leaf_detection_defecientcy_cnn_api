import io
import os
import cv2
import json
import time
import serial
import asyncio
import traceback
import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path
from datetime import datetime
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    WebSocket
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

# =========================================================
# ENVIRONMENT
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Hybrid CNN + NPK API"
)

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

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

app.mount(
    "/heatmaps",
    StaticFiles(directory=OUTPUT_DIR),
    name="heatmaps"
)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "final_model.keras"
)

MODEL = tf.keras.models.load_model(
    MODEL_PATH
)

CLASS_NAMES = [
    "k",
    "n",
    "p",
    "healthy",
    "not_cacao"
]

print("✅ MODEL LOADED")

# =========================================================
# FIND LAST CONV
# =========================================================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        try:

            if len(layer.output.shape) == 4:
                return layer.name

        except:
            continue

    raise Exception(
        "No convolution layer found"
    )

LAST_CONV_LAYER = find_last_conv_layer(
    MODEL
)

print(
    "✅ LAST CONV LAYER:",
    LAST_CONV_LAYER
)

# =========================================================
# SERIAL
# =========================================================

try:

    ser = serial.Serial(
        "/dev/serial0",
        baudrate=9600,
        timeout=1
    )

    print("✅ SERIAL CONNECTED")

except Exception as e:

    print("❌ SERIAL ERROR:", e)

    ser = None

npk_query = bytearray([
    0x01, 0x03, 0x00, 0x1e,
    0x00, 0x03, 0x65, 0xcd
])

# =========================================================
# GLOBALS
# =========================================================

clients = set()

MAX_FILE_SIZE = 5 * 1024 * 1024

ALLOWED_TYPES = [
    "image/jpeg",
    "image/png"
]

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
# THRESHOLDS
# =========================================================

LOW_N = 15
HIGH_N = 25

LOW_P = 10
HIGH_P = 20

LOW_K = 20
HIGH_K = 40

# =========================================================
# TF INFERENCE
# =========================================================

@tf.function
def infer(x):

    return MODEL(
        x,
        training=False
    )

# =========================================================
# SENSOR STATUS
# =========================================================

def check_npk_status(sensor):

    n = sensor["n"]
    p = sensor["p"]
    k = sensor["k"]

    notifications = []
    recommendations = []

    # =====================================================
    # N
    # =====================================================

    if n < LOW_N:

        notifications.append(
            "LOW NITROGEN"
        )

        recommendations.append(
            "Apply Nitrogen fertilizer"
        )

    elif n > HIGH_N:

        notifications.append(
            "HIGH NITROGEN"
        )

        recommendations.append(
            "Reduce Nitrogen fertilizer"
        )

    else:

        notifications.append(
            "NORMAL NITROGEN"
        )

    # =====================================================
    # P
    # =====================================================

    if p < LOW_P:

        notifications.append(
            "LOW PHOSPHORUS"
        )

        recommendations.append(
            "Apply Phosphorus fertilizer"
        )

    elif p > HIGH_P:

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
    # K
    # =====================================================

    if k < LOW_K:

        notifications.append(
            "LOW POTASSIUM"
        )

        recommendations.append(
            "Apply Potassium fertilizer"
        )

    elif k > HIGH_K:

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
    # SOIL STATUS
    # =====================================================

    if (
        LOW_N <= n <= HIGH_N and
        LOW_P <= p <= HIGH_P and
        LOW_K <= k <= HIGH_K
    ):

        soil_status = "SOIL HEALTHY"

    else:

        soil_status = "SOIL NEEDS ATTENTION"

    return {

        "notifications":
            notifications,

        "recommendations":
            recommendations,

        "soil_status":
            soil_status
    }

# =========================================================
# READ SENSOR
# =========================================================

def read_npk_sensor():

    if ser is None:
        return None

    try:

        ser.write(npk_query)

        time.sleep(0.15)

        response = ser.read(11)

        if len(response) == 11:

            n = (
                response[3] << 8
            ) | response[4]

            p = (
                response[5] << 8
            ) | response[6]

            k = (
                response[7] << 8
            ) | response[8]

            return {

                "n": n,
                "p": p,
                "k": k,

                "time":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
            }

    except Exception as e:

        print(
            "SENSOR ERROR:",
            e
        )

    return None

# =========================================================
# SENSOR LOOP
# =========================================================

async def sensor_loop():

    global latest_sensor_data

    while True:

        try:

            loop = asyncio.get_running_loop()

            data = await loop.run_in_executor(
                None,
                read_npk_sensor
            )

            if data:

                latest_sensor_data = (
                    NPKData(**data)
                )

                result = check_npk_status(
                    data
                )

                payload = {

                    "sensor_data":
                        latest_sensor_data.dict(),

                    "notifications":
                        result["notifications"],

                    "recommendations":
                        result["recommendations"],

                    "soil_status":
                        result["soil_status"]
                }

                await broadcast(payload)

        except Exception as e:

            print(
                "SENSOR LOOP ERROR:",
                e
            )

        await asyncio.sleep(2)

# =========================================================
# BROADCAST
# =========================================================

async def broadcast(data):

    disconnected = []

    for ws in list(clients):

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

    task = asyncio.create_task(
        sensor_loop()
    )

    yield

    task.cancel()

    try:

        await task

    except asyncio.CancelledError:

        print(
            "Sensor Loop Stopped"
        )

app.router.lifespan_context = lifespan

# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "running"
    }

# =========================================================
# SENSOR API
# =========================================================

@app.get("/sensor")
def get_sensor():

    sensor = latest_sensor_data.dict()

    result = check_npk_status(
        sensor
    )

    return {

        "sensor_data":
            sensor,

        "notifications":
            result["notifications"],

        "recommendations":
            result["recommendations"],

        "soil_status":
            result["soil_status"]
    }

# =========================================================
# SENSOR STREAM
# =========================================================

@app.get("/sensor-stream")
async def sensor_stream():

    async def event_generator():

        while True:

            sensor = latest_sensor_data.dict()

            result = check_npk_status(
                sensor
            )

            payload = {

                "sensor_data":
                    sensor,

                "notifications":
                    result["notifications"],

                "recommendations":
                    result["recommendations"],

                "soil_status":
                    result["soil_status"]
            }

            yield (
                f"data: {json.dumps(payload)}\n\n"
            )

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

    finally:

        clients.discard(websocket)

# =========================================================
# HYBRID FUSION
# =========================================================

def hybrid_fusion(
    predicted_class,
    cnn_confidence,
    sensor
):

    final_conf = cnn_confidence

    n = sensor["n"]
    p = sensor["p"]
    k = sensor["k"]

    # =====================================================
    # N
    # =====================================================

    if predicted_class == "n":

        if n < 15:

            final_conf += 0.10

        elif n > 35:

            final_conf -= 0.25

    # =====================================================
    # P
    # =====================================================

    elif predicted_class == "p":

        if p < 10:

            final_conf += 0.10

        elif p > 25:

            final_conf -= 0.25

    # =====================================================
    # K
    # =====================================================

    elif predicted_class == "k":

        if k < 20:

            final_conf += 0.10

        elif k > 45:

            final_conf -= 0.25

    # =====================================================
    # HEALTHY
    # =====================================================

    elif predicted_class == "healthy":

        if (
            15 <= n <= 25 and
            10 <= p <= 20 and
            20 <= k <= 40
        ):

            final_conf += 0.10

        else:

            final_conf -= 0.20

    # =====================================================
    # LIMIT
    # =====================================================

    final_conf = max(
        0.0,
        min(final_conf, 1.0)
    )

    # =====================================================
    # STATUS
    # =====================================================

    if final_conf >= 0.90:

        status = "VERY STRONG"

    elif final_conf >= 0.75:

        status = "STRONG"

    elif final_conf >= 0.60:

        status = "MODERATE"

    else:

        status = "WEAK"

    return {

        "final_confidence":
            final_conf,

        "status":
            status
    }

# =========================================================
# GENERATE GRADCAM
# =========================================================

def generate_gradcam(
    model,
    img_array
):

    grad_model = tf.keras.models.Model(

        [model.inputs],

        [
            model.get_layer(
                LAST_CONV_LAYER
            ).output,

            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(img_array)
        )

        predicted_idx = tf.argmax(
            predictions[0]
        )

        loss = predictions[
            :,
            predicted_idx
        ]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(

        tf.multiply(
            pooled_grads,
            conv_outputs
        ),

        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_heat = tf.reduce_max(
        heatmap
    )

    if max_heat == 0:
        return None

    heatmap /= max_heat

    return heatmap.numpy()

# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(

    files: list[UploadFile] = File(...)
):

    try:

        predictions = []
        confidences = []
        heatmaps = []

        for file in files:

            # =================================================
            # VALIDATE
            # =================================================

            if (
                file.content_type
                not in ALLOWED_TYPES
            ):
                continue

            img_bytes = await file.read()

            if len(img_bytes) == 0:
                continue

            if len(img_bytes) > MAX_FILE_SIZE:

                return {

                    "success": False,

                    "error":
                        "Image too large"
                }

            # =================================================
            # LOAD IMAGE
            # =================================================

            try:

                image = Image.open(
                    io.BytesIO(img_bytes)
                ).convert("RGB")

            except:

                continue

            image = image.resize(
                (224, 224)
            )

            img_np = np.array(image)

            # =================================================
            # PREPROCESS
            # =================================================

            img_array = (
                img_np.astype("float32")
                / 255.0
            )

            img_batch = np.expand_dims(
                img_array,
                axis=0
            )

            # =================================================
            # PREDICT
            # =================================================

            loop = asyncio.get_running_loop()

            preds = await loop.run_in_executor(

                None,

                lambda:
                    infer(img_batch).numpy()
            )

            idx = np.argmax(
                preds[0]
            )

            cls = CLASS_NAMES[idx]

            conf = float(
                np.max(preds[0])
            )

            predictions.append(cls)

            confidences.append(conf)

            # =================================================
            # GRADCAM
            # =================================================

            heatmap = generate_gradcam(
                MODEL,
                img_batch
            )

            if heatmap is not None:

                heatmap = cv2.resize(

                    heatmap,

                    (
                        img_np.shape[1],
                        img_np.shape[0]
                    )
                )

                heatmap = np.uint8(
                    255 * heatmap
                )

                heatmap_color = (
                    cv2.applyColorMap(
                        heatmap,
                        cv2.COLORMAP_JET
                    )
                )

                superimposed = (
                    cv2.addWeighted(

                        img_np,
                        0.65,

                        heatmap_color,
                        0.35,

                        0
                    )
                )

                # =============================================
                # SAVE IMAGE
                # =============================================

                timestamp = (
                    datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )
                )

                filename = (
                    f"{timestamp}_{cls}.jpg"
                )

                save_path = os.path.join(
                    OUTPUT_DIR,
                    filename
                )

                final_bgr = cv2.cvtColor(
                    superimposed,
                    cv2.COLOR_RGB2BGR
                )

                cv2.imwrite(
                    save_path,
                    final_bgr
                )

                heatmaps.append({

                    "class":
                        cls,

                    "confidence":
                        conf,

                    "heatmap_url":
                        f"/heatmaps/{filename}",

                    "local_path":
                        save_path
                })

        # =====================================================
        # NO VALID IMAGE
        # =====================================================

        if not predictions:

            return {

                "success": False,

                "error":
                    "No valid image uploaded"
            }

        # =====================================================
        # MAJORITY VOTE
        # =====================================================

        final_class = Counter(
            predictions
        ).most_common(1)[0][0]

        avg_conf = (
            sum(confidences)
            / len(confidences)
        )

        # =====================================================
        # SENSOR DATA
        # =====================================================

        sensor_data = latest_sensor_data.dict()

        threshold = check_npk_status(
            sensor_data
        )

        fusion = hybrid_fusion(

            final_class,
            avg_conf,
            sensor_data
        )

        # =====================================================
        # RESPONSE
        # =====================================================

        return {

            "success": True,

            "final_class":
                final_class,

            "cnn_confidence":
                avg_conf,

            "sensor_data":
                sensor_data,

            "final_confidence":
                fusion["final_confidence"],

            "status":
                fusion["status"],

            "soil_status":
                threshold["soil_status"],

            "notifications":
                threshold["notifications"],

            "recommendations":
                threshold["recommendations"],

            "heatmaps":
                heatmaps
        }

    except Exception as e:

        traceback.print_exc()

        return {

            "success": False,

            "error": str(e)
        }

# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3000
    )