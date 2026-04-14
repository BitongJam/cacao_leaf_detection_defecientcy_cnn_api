import asyncio
import json
import serial
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

# -----------------------------
# SERIAL SETUP (NPK SENSOR)
# -----------------------------
ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

npk_query = bytearray([0x01, 0x03, 0x00, 0x1e, 0x00, 0x03, 0x65, 0xcd])

def read_npk_sensor():
    try:
        ser.write(npk_query)
        time.sleep(0.15)

        response = ser.read(11)
        print("Raw:", response.hex())

        if len(response) == 11 and response[0] == 0x01 and response[1] == 0x03:
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
        print("Sensor error:", e)

    return None

# -----------------------------
# MODEL
# -----------------------------
class NPKData(BaseModel):
    n: int
    p: int
    k: int
    time: str

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
clients: set[WebSocket] = set()
latest_data = NPKData(n=0, p=0, k=0, time="")

# -----------------------------
# BROADCAST FUNCTION
# -----------------------------
async def broadcast(data: dict):
    disconnected = []
    for ws in clients:
        try:
            await ws.send_json(data)
        except:
            disconnected.append(ws)

    for ws in disconnected:
        clients.discard(ws)

# -----------------------------
# SENSOR LOOP (REAL DATA)
# -----------------------------
async def sensor_loop():
    global latest_data

    while True:
        new_data = read_npk_sensor()

        if new_data:
            if (new_data["n"] != latest_data.n or
                new_data["p"] != latest_data.p or
                new_data["k"] != latest_data.k):

                latest_data = NPKData(**new_data)

                print("LIVE:", latest_data.dict())

                await broadcast(latest_data.dict())

        await asyncio.sleep(2)

# -----------------------------
# LIFESPAN
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sensor_loop())
    yield
    task.cancel()

# -----------------------------
# APP
# -----------------------------
app = FastAPI(lifespan=lifespan)

# -----------------------------
# NORMAL API
# -----------------------------
@app.get("/sensor")
def get_sensor():
    return latest_data

# -----------------------------
# SSE STREAM
# -----------------------------
@app.get("/sensor-stream")
async def sensor_stream():
    async def event_generator():
        while True:
            yield f"data: {json.dumps(latest_data.dict())}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -----------------------------
# WEBSOCKET
# -----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    try:
        while True:
            await asyncio.sleep(1)
    except:
        pass
    finally:
        clients.discard(websocket)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)