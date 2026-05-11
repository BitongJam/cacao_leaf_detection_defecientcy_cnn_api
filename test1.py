import asyncio
import json
import serial
import time
import io

from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    WebSocket,
    File,
    UploadFile
)

from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from PIL import Image
import tensorflow as tf

# =========================================================
# SERIAL CONFIG
# =========================================================
ser = serial.Serial(
    '/dev/serial0',
    9600,
    timeout=1
)

query = bytearray([
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
# MODEL CONFIG
# =========================================================
MODEL = None

CLASS_NAMES = [
    "Nitrogen Deficiency",
    "Phosphorus Deficiency",
    "Potassium Deficiency",
    "Healthy",
    "Not Cacao"
]

# =========================================================
# HYBRID FUSION WEIGHTS
# =========================================================
ALPHA = 0.7   # CNN WEIGHT
BETA = 0.3    # SENSOR WEIGHT

# =========================================================
# LOAD MODEL
# =========================================================
def get_model():

    global MODEL

    if MODEL is None:

        MODEL = tf.keras.models.load_model(
            "../cacao_project/models/1.keras"
        )

        print("✅ CNN MODEL LOADED")

    return MODEL

# =========================================================
# DATA MODEL
# =========================================================
class NPK(BaseModel):

    n: int
    p: int
    k: int
    time: str

# =========================================================
# GLOBAL STATE
# =========================================================
clients = set()

latest = NPK(
    n=0,
    p=0,
    k=0,
    time="starting..."
)

# =========================================================
# SENSOR FUNCTION
# =========================================================
def read_sensor():

    try:

        ser.write(query)

        time.sleep(0.15)

        response = ser.read(11)

        print("RAW:", response.hex())

        if (
            len(response) == 11 and
            response[0] == 1 and
            response[1] == 3
        ):

            nitrogen = (
                response[3] << 8
            ) | response[4]

            phosphorus = (
                response[5] << 8
            ) | response[6]

            potassium = (
                response[7] << 8
            ) | response[8]

            return {
                "n": nitrogen,
                "p": phosphorus,
                "k": potassium,
                "time": datetime.now().strftime(
                    "%H:%M:%S"
                )
            }

    except Exception as e:

        print("❌ SENSOR ERROR:", e)

# =========================================================
# LIVE SENSOR LOOP
# =========================================================
async def sensor_loop():

    global latest

    while True:

        data = await asyncio.to_thread(
            read_sensor
        )

        if data:

            changed = (
                data["n"],
                data["p"],
                data["k"]
            ) != (
                latest.n,
                latest.p,
                latest.k
            )

            if changed:

                latest = NPK(**data)

                print("\n======================")
                print("🌱 LIVE SENSOR")
                print("======================")
                print(latest.dict())

                # SEND TO WEBSOCKET CLIENTS
                for ws in list(clients):

                    try:
                        await ws.send_json(
                            latest.dict()
                        )

                    except:
                        clients.remove(ws)

        await asyncio.sleep(2)

# =========================================================
# FASTAPI LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):

    task = asyncio.create_task(
        sensor_loop()
    )

    yield

    task.cancel()

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():

    return {
        "message":
            "HYBRID CNN + NPK SENSOR RUNNING"
    }

# =========================================================
# SENSOR API
# =========================================================
@app.get("/sensor")
def get_sensor():

    return latest

# =========================================================
# SENSOR STREAM
# =========================================================
@app.get("/sensor-stream")
async def sensor_stream():

    async def event_generator():

        while True:

            yield (
                f"data:{json.dumps(latest.dict())}\n\n"
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
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()

    clients.add(ws)

    await ws.send_json(
        latest.dict()
    )

    print("✅ CLIENT CONNECTED")

    try:

        while True:
            await asyncio.sleep(1)

    finally:

        clients.remove(ws)

        print("❌ CLIENT DISCONNECTED")

# =========================================================
# HYBRID CNN + SENSOR FUSION
# =========================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # =================================================
        # READ IMAGE
        # =================================================
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize((256, 256))

        image_array = np.array(
            image
        ).astype("float32")

        # NORMALIZE
        image_array = image_array / 255.0

        image_batch = np.expand_dims(
            image_array,
            0
        )

        # =================================================
        # CNN PREDICTION
        # =================================================
        model = get_model()

        predictions = model.predict(
            image_batch
        )

        cnn_probs = tf.nn.softmax(
            predictions[0]
        ).numpy()

        cnn_idx = np.argmax(
            cnn_probs
        )

        cnn_class = CLASS_NAMES[
            cnn_idx
        ]

        print("\n======================")
        print("🧠 CNN RESULT")
        print("======================")
        print("CNN CLASS:", cnn_class)
        print("CNN PROBS:", cnn_probs)

        # =================================================
        # SENSOR VALUES
        # =================================================
        n = latest.n
        p = latest.p
        k = latest.k

        print("\n======================")
        print("🌱 SENSOR VALUES")
        print("======================")
        print("N:", n)
        print("P:", p)
        print("K:", k)

        # =================================================
        # SENSOR LOGIC
        # =================================================
        sensor_scores = np.array([0.0, 0.0, 0.0, 0.0])

        # LOW NITROGEN
        if n < p and n < k:
            sensor_scores[0] = 1.0

        # LOW PHOSPHORUS
        elif p < n and p < k:
            sensor_scores[1] = 1.0

        # LOW POTASSIUM
        elif k < n and k < p:
            sensor_scores[2] = 1.0

        else:
            sensor_scores[3] = 1.0

        print("\n======================")
        print("📡 SENSOR SCORES")
        print("======================")
        print(sensor_scores)

        # =================================================
        # HYBRID FUSION
        # =================================================
        final_probs = (
            ALPHA * cnn_probs
        ) + (
            BETA * sensor_scores
        )

        final_idx = np.argmax(
            final_probs
        )

        final_class = CLASS_NAMES[
            final_idx
        ]

        confidence = float(
            final_probs[final_idx]
        )

        print("\n======================")
        print("🔥 FINAL FUSION")
        print("======================")
        print("FINAL CLASS:", final_class)
        print("CONFIDENCE:", confidence)
        print("FINAL PROBS:", final_probs)

        # =================================================
        # RETURN
        # =================================================
        return {

            "success": True,

            # FINAL RESULT
            "final_class":
                final_class,

            "confidence":
                confidence,

            # CNN ONLY
            "cnn_prediction":
                cnn_class,

            # SENSOR DATA
            "sensor_values": {
                "nitrogen": n,
                "phosphorus": p,
                "potassium": k
            },

            # DEBUGGING
            "cnn_probs":
                cnn_probs.tolist(),

            "sensor_scores":
                sensor_scores.tolist(),

            "final_probs":
                final_probs.tolist(),

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }

    except Exception as e:

        print("❌ ERROR:", str(e))

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