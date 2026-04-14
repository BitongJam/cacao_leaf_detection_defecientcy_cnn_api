import asyncio,json,serial,time
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


# =========================
# SERIAL CONFIG
# =========================
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
query = bytearray([0x01,0x03,0x00,0x1e,0x00,0x03,0x65,0xcd])


# =========================
# MODEL CONFIG
# =========================
MODEL = None
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model("../cacao_project/models/1.keras")
    return MODEL


# =========================
# DATA MODEL
# =========================
class NPK(BaseModel):
    n: int
    p: int
    k: int
    time: str


# =========================
# GLOBAL STATE
# =========================
clients = set()
latest = NPK(n=0, p=0, k=0, time="starting...")


# =========================
# SENSOR FUNCTION
# =========================
def read_sensor():
    try:
        ser.write(query)
        time.sleep(0.15)
        response = ser.read(11)

        if len(response) == 11 and response[0] == 1 and response[1] == 3:
            return {
                "n": (response[3] << 8) | response[4],
                "p": (response[5] << 8) | response[6],
                "k": (response[7] << 8) | response[8],
                "time": datetime.now().strftime("%H:%M:%S")
            }

    except Exception as e:
        print("Serial Error:", e)


# =========================
# BACKGROUND LOOP
# =========================
async def sensor_loop():
    global latest

    while True:
        data = await asyncio.to_thread(read_sensor)

        if data:
            changed = (
                data["n"], data["p"], data["k"]
            ) != (
                latest.n, latest.p, latest.k
            )

            if changed:
                latest = NPK(**data)
                print("LIVE:", latest.dict())

                for ws in list(clients):
                    try:
                        await ws.send_json(latest.dict())
                    except:
                        clients.remove(ws)

        await asyncio.sleep(2)


# =========================
# FASTAPI SETUP
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sensor_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ROUTES
# =========================
# =========================
# IMAGE PREDICTION
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes)).resize((256, 256))
        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, 0)

        model = get_model()
        predictions = model.predict(image_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/sensor")
def get_sensor():
    return latest


@app.get("/sensor-stream")
async def sensor_stream():
    async def event_generator():
        while True:
            yield f"data:{json.dumps(latest.dict())}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    await ws.send_json(latest.dict())

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        clients.remove(ws)





# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=3000)