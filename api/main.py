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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CUSTOM LAYERS (IMPORTANT)
# =========================================================

@tf.keras.utils.register_keras_serializable()
class WaveletLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs  # simplified for stability

@tf.keras.utils.register_keras_serializable()
class WTResidualBlock(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs
OUTPUT_DIR = "detected_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "wt_resnet_model.keras"
MODEL = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "WaveletLayer": WaveletLayer,
        "WTResidualBlock": WTResidualBlock
    },
    compile=False
)

CLASS_NAMES = ["k", "N", "P", "healthy", "not_cacao"]


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
        print(f"? HEATMAP ERROR: {str(e)}")
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


# =========================================================||
#               NPK STATUS CHECKER                         ||
# =========================================================||

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
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
     # ?? Agriculture scoring system (confidence-weighted)
    results = []
    predictions = []

     # Store confidence percentages per deficiency class
    npk_groups = {
        "n": [],
        "p": [],
        "k": []
    }

    npk_scores = {"n": 0.0, "p": 0.0, "k": 0.0}

    try:
        # results = []
        confidence_map = defaultdict(list)

        for file in files:
            image_bytes = await file.read()

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((224, 224))

            arr = np.array(img).astype("float32")
            batch = np.expand_dims(arr, axis=0)

            preds = MODEL.predict(batch, verbose=0)

            predicted_idx = int(np.argmax(preds[0]))
            cls = CLASS_NAMES[predicted_idx]
            conf = float(np.max(preds[0]))

#==================================================================
#     perccentage
# ==================================================================
            percentage = round(conf * 100, 2)

            confidence_map[cls].append(percentage)
#===================================================================
#               GROUP N/P/K PERCENTAGES
#===================================================================

            if cls.lower() in npk_groups:
                npk_groups[cls.lower()].append(percentage)
    
# ==================================================================
#                   HEATMAP GENERATION
# ==================================================================

            heatmap_bgr, heatmap_base64 = generate_heatmap(
                img,          # FIX: was "image"
                batch,        # FIX: was "img_batch"
                predicted_idx # FIX: was undefined
            )

            heatmap_filename = None

            if heatmap_bgr is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                label = cls.replace(" ", "_")

                heatmap_filename = f"{timestamp}_{label}.jpg"
                save_path = os.path.join(OUTPUT_DIR, heatmap_filename)

                cv2.imwrite(save_path, heatmap_bgr)





            results.append({
                "file": file.filename,
                "class": cls,
                "confidence": conf,
                "heatmap_file": heatmap_filename
            })

        # =====================================================
        #       compute mean value per NPK
        # =====================================================
        npk_scores = {}
        for nutirents, value in npk_groups.items():
            if value:
                npk_scores[nutirents] = round(
                    sum(value) /len(files)
                )
            else:
                npk_scores[nutirents] = 0.0

# ==================================================
#                status defficiency
# ==================================================


        status = {}

        NPK_DEFICIENCY_THRESHOLD = 50

        for nutrient, score in npk_scores.items():

            if score >= NPK_DEFICIENCY_THRESHOLD:

                status[nutrient] = "Deficiency"

            else:

                status[nutrient] = "Healthy"
#====================================================
#                BEST CLASS
#====================================================

            if confidence_map:

                best = max(
                    confidence_map.items(),
                    key=lambda x: sum(x[1])
                )[0]

            else:

                best = "healthy"
        

          

        # best = max(confidence_map.items(), key=lambda x: sum(x[1]))[0]

        return {
            "success": True,
            "best_class": best,
            "results": results,
            "npk_scores": npk_scores,
            'status': status
            
        }
    except Exception as e:
        print("ERROR:", str(e))
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000)