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

    if n < LOW_THRESHOLD_nitrogine:

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

def evaluate_sensor_support(
    predicted_class,
    sensor
):

    n = sensor["n"]
    p = sensor["p"]
    k = sensor["k"]

    score = 0.5

    # =====================================================
    # NITROGEN
    # =====================================================

    if predicted_class == "n":

        if n < 20:
            score = 0.95

        elif n < 40:
            score = 0.75

        else:
            score = 0.30

    # =====================================================
    # PHOSPHORUS
    # =====================================================

    elif predicted_class == "p":

        if p < 20:
            score = 0.95

        elif p < 40:
            score = 0.75

        else:
            score = 0.30

    # =====================================================
    # POTASSIUM
    # =====================================================

    elif predicted_class == "k":

        if k < 20:
            score = 0.95

        elif k < 40:
            score = 0.75

        else:
            score = 0.30

    # =====================================================
    # HEALTHY
    # =====================================================

    elif predicted_class == "healthy":

        if (
            n > 40 and
            p > 40 and
            k > 40
        ):

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
# MAIN PREDICT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # =================================================
        # IMAGE READ
        # =================================================

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
        # PREDICT
        # =================================================

        predictions = MODEL.predict(
            img_batch
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
        # RETURN
        # =================================================

        return {

            "success": True,

            # CNN
            "cnn_class":
                predicted_class,

            "cnn_confidence":
                fusion_result[
                    "cnn_confidence"
                ],

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

    except Exception as e:

        print("ERROR:", str(e))

        return {

            "success": False,
            "error": str(e)
        }

# =========================================================
# MULTI PREDICT
# =========================================================

@app.post("/multi-predict")
async def multi_predict(
    files: list[UploadFile] = File(...)
):

    results = []

    total_confidence = 0

    class_confidences = defaultdict(
        list
    )

    for file in files:

        result = await predict(file)

        results.append(result)

        cls = result["cnn_class"]

        conf = result["final_confidence"]

        total_confidence += conf

        class_confidences[cls].append(conf)

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

        "best_class":
            best_class,

        "avg_confidence":
            avg_confidence,

        "details":
            results
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