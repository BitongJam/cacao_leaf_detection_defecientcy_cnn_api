import io
import os
import cv2
import json
import base64
import serial
import asyncio
import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# =========================================================
# GLOBALS & CONSTANTS
# =========================================================
OUTPUT_DIR = "detected_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["k", "n", "p", "healthy", "not_cacao"]
DEFICIENCY_MAP = {
    "n": "Nitrogen Deficiency",
    "p": "Phosphorus Deficiency",
    "k": "Potassium Deficiency",
    "healthy": "Healthy Leaf",
    "not_cacao": "Not Cacao Leaf"
}

# Threshold settings
LOW_THRESHOLD_NITROGEN = 12
HIGH_THRESHOLD_NITROGEN = 18

LOW_THRESHOLD_PHOSPHORUS = 12
HIGH_THRESHOLD_PHOSPHORUS = 20

LOW_THRESHOLD_POTASSIUM = 15
HIGH_THRESHOLD_POTASSIUM = 25

clients = set()
ser = None  # I-initialize sa lifespan
MODEL = None

# NPK Query Bytearray
NPK_QUERY = bytearray([0x01, 0x03, 0x00, 0x1e, 0x00, 0x03, 0x65, 0xcd])

class NPKData(BaseModel):
    n: int
    p: int
    k: int
    time: str

latest_sensor_data = NPKData(n=0, p=0, k=0, time="")

# =========================================================
# LIFESPAN (STARTUP/SHUTDOWN)
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, ser
    # Load Model
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "final_model.keras"
    MODEL = tf.keras.models.load_model(model_path)
    print("✅ CNN MODEL LOADED")

    # Open Serial Port
    try:
        ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)
        print("✅ SERIAL PORT OPENED")
    except Exception as e:
        print(f"❌ SERIAL PORT ERROR: {e}")

    # Start Background Sensor Task
    task = asyncio.create_task(sensor_loop())
    
    yield
    
    # Cleanup
    task.cancel()
    if ser and ser.is_open:
        ser.close()
        print("🔒 SERIAL PORT CLOSED")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def generate_heatmap(image, img_batch, predicted_idx):
    """Generates Grad-CAM heatmap for the prediction."""
    try:
        try:
            target_layer = MODEL.get_layer("last_conv_layer")
        except Exception:
            target_layer = [layer for layer in MODEL.layers if "conv" in layer.name.lower()][-1]

        grad_model = tf.keras.Model(inputs=MODEL.inputs, outputs=[target_layer.output, MODEL.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            loss = predictions[:, predicted_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ tf.expand_dims(pooled_grads, -1)
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        img_np = np.array(image)
        heatmap_resized = cv2.resize(heatmap_np, (img_np.shape[1], img_np.shape[0]))
        
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(np.ascontiguousarray(heatmap_color), cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color_rgb, 0.4, 0)
        final_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        
        _, buffer = cv2.imencode('.jpg', final_bgr)
        heatmap_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        return final_bgr, heatmap_base64
    except Exception as e:
        print(f"❌ HEATMAP ERROR: {str(e)}")
        return None, None


def check_npk_status(sensor):
    """Evaluates NPK values against thresholds."""
    n, p, k = sensor["n"], sensor["p"], sensor["k"]
    notifications, recommendations = [], []

    # Nitrogen
    if n < LOW_THRESHOLD_NITROGEN:
        notifications.append("LOW NITROGEN")
        recommendations.append("Apply Nitrogen fertilizer")
    elif n > HIGH_THRESHOLD_NITROGEN:
        notifications.append("HIGH NITROGEN")
        recommendations.append("Reduce Nitrogen fertilizer application")
    else:
        notifications.append("NORMAL NITROGEN")

    # Phosphorus
    if p < LOW_THRESHOLD_PHOSPHORUS:
        notifications.append("LOW PHOSPHORUS")
        recommendations.append("Apply Phosphorus fertilizer")
    elif p > HIGH_THRESHOLD_PHOSPHORUS:
        notifications.append("HIGH PHOSPHORUS")
        recommendations.append("Reduce Phosphorus fertilizer")
    else:
        notifications.append("NORMAL PHOSPHORUS")

    # Potassium
    if k < LOW_THRESHOLD_POTASSIUM:
        notifications.append("LOW POTASSIUM")
        recommendations.append("Apply Potassium fertilizer")
    elif k > HIGH_THRESHOLD_POTASSIUM:
        notifications.append("HIGH POTASSIUM")
        recommendations.append("Reduce Potassium fertilizer")
    else:
        notifications.append("NORMAL POTASSIUM")

    # Soil Health Status
    soil_healthy = (
        LOW_THRESHOLD_NITROGEN <= n <= HIGH_THRESHOLD_NITROGEN and
        LOW_THRESHOLD_PHOSPHORUS <= p <= HIGH_THRESHOLD_PHOSPHORUS and
        LOW_THRESHOLD_POTASSIUM <= k <= HIGH_THRESHOLD_POTASSIUM
    )
    soil_status = "SOIL HEALTHY" if soil_healthy else "SOIL NEEDS ATTENTION"

    return {
        "notifications": notifications,
        "recommendations": recommendations,
        "soil_status": soil_status
    }


def get_real_deficiency(sensor):
    """Determines the most severe actual soil deficiency."""
    n, p, k = sensor["n"], sensor["p"], sensor["k"]
    deficiencies = {}
    
    if n < LOW_THRESHOLD_NITROGEN:
        deficiencies["n"] = {"value": n, "threshold": LOW_THRESHOLD_NITROGEN, "severity": round(1 - (n / LOW_THRESHOLD_NITROGEN), 2)}
    if p < LOW_THRESHOLD_PHOSPHORUS:
        deficiencies["p"] = {"value": p, "threshold": LOW_THRESHOLD_PHOSPHORUS, "severity": round(1 - (p / LOW_THRESHOLD_PHOSPHORUS), 2)}
    if k < LOW_THRESHOLD_POTASSIUM:
        deficiencies["k"] = {"value": k, "threshold": LOW_THRESHOLD_POTASSIUM, "severity": round(1 - (k / LOW_THRESHOLD_POTASSIUM), 2)}
    
    if deficiencies:
        primary = max(deficiencies.items(), key=lambda x: x[1]["severity"])
        return primary[0], primary[1]
    
    return None, None


def evaluate_sensor_support(predicted_class, sensor):
    """Calculates how much the sensor confirms the CNN prediction."""
    n, p, k = sensor["n"], sensor["p"], sensor["k"]
    
    if predicted_class == "n":
        return 0.75 if n < 20 else (0.30 if n > 40 else 0.95)
    elif predicted_class == "p":
        return 0.75 if p < 10 else (0.30 if p > 20 else 0.95)
    elif predicted_class == "k":
        return 0.75 if k < 20 else (0.30 if k > 40 else 0.95)
    elif predicted_class == "healthy":
        return 0.95 if (10 <= n <= 25 and 10 <= p <= 20 and 20 <= k <= 40) else 0.40
    return 0.50


def hybrid_fusion(cnn_class, cnn_confidence, sensor_data):
    """Combines CNN confidence and Sensor support."""
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


def sync_read_npk():
    """Synchronous read helper to run in an executor thread."""
    if not ser or not ser.is_open:
        return None
    try:
        ser.write(NPK_QUERY)
        import time
        time.sleep(0.15)
        response = ser.read(11)
        if len(response) == 11 and response[0] == 0x01 and response[1] == 0x03:
            return {
                "n": (response[3] << 8) | response[4],
                "p": (response[5] << 8) | response[6],
                "k": (response[7] << 8) | response[8],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except Exception as e:
        print("SENSOR ERROR:", e)
    return None

# =========================================================
# BACKGROUND LOOPS & BROADCAST
# =========================================================

async def sensor_loop():
    """Non-blocking sensor poll loop using thread executor."""
    global latest_sensor_data
    loop = asyncio.get_running_loop()
    
    while True:
        # Padaganon sa laing thread aron dili ma-block ang main async event loop
        data = await loop.run_in_executor(None, sync_read_npk)
        
        if data:
            latest_sensor_data = NPKData(**data)
            threshold_result = check_npk_status(data)
            
            payload = {
                "sensor_data": latest_sensor_data.model_dump(),
                "notifications": threshold_result["notifications"],
                "recommendations": threshold_result["recommendations"],
                "soil_status": threshold_result["soil_status"]
            }
            await broadcast(payload)
            
        await asyncio.sleep(2)


async def broadcast(data):
    """Sends JSON to all connected WebSocket clients."""
    disconnected = set()
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.add(ws)
            
    for ws in disconnected:
        clients.discard(ws)

# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/sensor")
def get_sensor():
    sensor_dict = latest_sensor_data.model_dump()
    threshold_result = check_npk_status(sensor_dict)
    return {
        "sensor_data": sensor_dict,
        "notifications": threshold_result["notifications"],
        "recommendations": threshold_result["recommendations"],
        "soil_status": threshold_result["soil_status"]
    }


@app.get("/sensor-stream")
async def sensor_stream():
    async def event_generator():
        while True:
            sensor_dict = latest_sensor_data.model_dump()
            threshold_result = check_npk_status(sensor_dict)
            payload = {
                "sensor_data": sensor_dict,
                "notifications": threshold_result["notifications"],
                "recommendations": threshold_result["recommendations"],
                "soil_status": threshold_result["soil_status"]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(3600)  # Pagpabiling buhi sa connection
    except Exception:
        pass
    finally:
        clients.discard(websocket)


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    try:
        results = []
        total_confidence = 0
        class_confidences = defaultdict(list)
        sensor_data = latest_sensor_data.model_dump()
        threshold_result = check_npk_status(sensor_data)
        real_deficiency_class, real_deficiency_info = get_real_deficiency(sensor_data)

        # Triggers to monitor non-cacao leaf detection
        has_not_cacao = False
        not_cacao_result = None

        for file in files:
            file_bytes = await file.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized).astype("float32")
            img_batch = tf.expand_dims(img_array, 0)

            # Inference
            predictions = MODEL.predict(img_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_idx]
            cnn_confidence = float(np.max(predictions[0]))

            # Heatmap
            heatmap_bgr, _ = generate_heatmap(image, img_batch, predicted_idx)
            heatmap_filename = None
            
            if heatmap_bgr is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                label = predicted_class.replace(" ", "_")
                heatmap_filename = f"{timestamp}_{label}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, heatmap_filename), heatmap_bgr)

            fusion_result = hybrid_fusion(predicted_class, cnn_confidence, sensor_data)

            result = {
                "success": True,
                "cnn_class": predicted_class,
                "cnn_confidence": fusion_result["cnn_confidence"],
                "real_deficiency_info": real_deficiency_info,
                "heatmap_filename": heatmap_filename,
                "sensor_data": sensor_data,
                "sensor_support": fusion_result["sensor_support"],
                "final_confidence": fusion_result["final_confidence"],
                "status": fusion_result["status"],
                "soil_status": threshold_result["soil_status"],
                "notifications": threshold_result["notifications"],
                "recommendations": threshold_result["recommendations"]
            }
            results.append(result)

            total_confidence += fusion_result["final_confidence"]
            class_confidences[predicted_class].append(fusion_result["final_confidence"])

            # Strict Filter Check: Kung naay makit-an nga not_cacao, i-flag dayon
            if predicted_class == "not_cacao":
                has_not_cacao = True
                not_cacao_result = result

        # =========================================================
        # STRICT FILTER BREAK / EARLY EXIT
        # =========================================================
        if has_not_cacao:
            # Maskin pila pa ang gi-upload, kung naay not_cacao, i-short-circuit ang voting response
            return {
                "success": True,
                "overall_result": {
                    "final_class": "not_cacao",
                    "deficiency": DEFICIENCY_MAP.get("not_cacao"),
                    "confidence": not_cacao_result["final_confidence"] if len(results) == 1 else round(total_confidence / len(results), 4),
                    "status": "STRONG DETECTION" if not_cacao_result["final_confidence"] >= 0.85 else "MODERATE DETECTION",
                    "severity": "NORMAL",
                    "decision_source": "strict_not_cacao_filter"
                },
                "sensor_analysis": {
                    "sensor_data": sensor_data,
                    "real_deficiency_class": real_deficiency_class,
                    "real_deficiency_info": real_deficiency_info,
                    "soil_status": threshold_result["soil_status"],
                    "notifications": threshold_result["notifications"],
                    "recommendations": threshold_result["recommendations"]
                },
                "image_analysis": {
                    "total_images": len(results),
                    "note": "Process short-circuited due to non-cacao leaf detection."
                },
                "details": results
            }

        # Return format for Single Image (Normal Flow - if walay not_cacao)
        if len(results) == 1:
            return results[0]

        # Multi-image processing logic (Normal Flow)
        class_scores = {cls: 0.0 for cls in CLASS_NAMES}
        class_counts = {cls: 0 for cls in CLASS_NAMES}

        for item in results:
            cls = item["cnn_class"]
            conf = item["final_confidence"]
            class_scores[cls] += conf
            class_counts[cls] += 1

        avg_confidence = round(total_confidence / len(results), 4)
        final_class = max(class_scores, key=lambda cls: class_scores[cls])
        deficiency_label = DEFICIENCY_MAP.get(final_class, "Unknown")

        # Severity assessment
        severity_label = "NORMAL"
        if real_deficiency_info:
            severity = real_deficiency_info["severity"]
            if severity >= 0.7:
                severity_label = "SEVERE"
            elif severity >= 0.4:
                severity_label = "MODERATE"
            else:
                severity_label = "MILD"

        overall_status = "STRONG DETECTION" if avg_confidence >= 0.85 else ("MODERATE DETECTION" if avg_confidence >= 0.60 else "WEAK DETECTION")

        return {
            "success": True,
            "overall_result": {
                "final_class": final_class,
                "deficiency": deficiency_label,
                "confidence": avg_confidence,
                "status": overall_status,
                "severity": severity_label,
                "decision_source": "cnn_weighted_voting"
            },
            "sensor_analysis": {
                "sensor_data": sensor_data,
                "real_deficiency_class": real_deficiency_class,
                "real_deficiency_info": real_deficiency_info,
                "soil_status": threshold_result["soil_status"],
                "notifications": threshold_result["notifications"],
                "recommendations": threshold_result["recommendations"]
            },
            "image_analysis": {
                "total_images": len(results),
                "class_counts": class_counts,
                "class_scores": class_scores
            },
            "details": results
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"success": False, "error": str(e)}


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return {"detail": "File not found"}
    return FileResponse(file_path, media_type="image/jpeg", filename=filename)


@app.get("/image/{filename}")
def view_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
