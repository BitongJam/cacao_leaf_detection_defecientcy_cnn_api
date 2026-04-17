import random
import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

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
# SIMULATED SENSOR
# -----------------------------
def read_npk_simulated():
    return {
        "n": random.randint(0, 100),
        "p": random.randint(0, 100),
        "k": random.randint(0, 100),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

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
# SENSOR LOOP (LIVE UPDATE)
# -----------------------------
async def sensor_loop():
    global latest_data
    while True:
        new_data = read_npk_simulated()

        # only update if changed
        if (new_data["n"] != latest_data.n or
            new_data["p"] != latest_data.p or
            new_data["k"] != latest_data.k):

            latest_data = NPKData(**new_data)

            print("LIVE:", latest_data.dict())

            # send to websocket clients
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
# LIVE STREAM (SSE) ??
# -----------------------------
@app.get("/sensor-stream")
async def sensor_stream():
    async def event_generator():
        while True:
            data = latest_data.dict()
            yield f"data: {json.dumps(data)}\n\n"
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