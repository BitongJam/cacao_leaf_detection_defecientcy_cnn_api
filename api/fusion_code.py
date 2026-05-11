import io
import os
import base64
import numpy as np
import tensorflow as tf
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from datetime import datetime
from pathlib import Path

app = FastAPI()

# ----------------------------
# CORS CONFIG
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# SAVE DIRECTORY
# ----------------------------
OUTPUT_DIR = "detected_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.keras"

MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["k", "n", "p", "healthy", "not_cacao"]

print("✅ Model loaded successfully")

# ----------------------------
# FUSION WEIGHTS
# ----------------------------
ALPHA = 0.7  # CNN weight
BETA = 0.3   # Sensor weight


# =========================================================
# MAIN PREDICTION ENDPOINT (HYBRID FUSION)
# =========================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),

    # 🌱 SENSOR INPUTS
    n: float = 0,
    p: float = 0,
    k: float = 0,
    ph: float = 0,
    moisture: float = 0
):
    try:
        # ----------------------------
        # IMAGE PREPROCESSING
        # ----------------------------
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized).astype("float32")

        img_batch = tf.expand_dims(img_array, 0)

        # ----------------------------
        # GRAD-CAM MODEL
        # ----------------------------
        try:
            target_layer = MODEL.get_layer("last_conv_layer")
        except:
            target_layer = MODEL.layers[-3]  # fallback safe

        grad_model = tf.keras.Model(
            inputs=MODEL.inputs,
            outputs=[target_layer.output, MODEL.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            predicted_idx = tf.argmax(predictions[0])
            loss = predictions[:, predicted_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ tf.expand_dims(pooled_grads, -1)
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        # ----------------------------
        # CNN PROBABILITIES
        # ----------------------------
        cnn_probs = tf.nn.softmax(predictions[0]).numpy()

        # ----------------------------
        # SENSOR NORMALIZATION
        # ----------------------------
        sensor_values = np.array([
            n / 100,
            p / 100,
            k / 100,
            ph / 14,
            moisture / 100
        ])

        sensor_probs = sensor_values / (np.sum(sensor_values) + 1e-10)

        # ----------------------------
        # HYBRID FUSION (CORE LOGIC)
        # ----------------------------
        final_probs = (ALPHA * cnn_probs) + (BETA * sensor_probs)

        final_idx = np.argmax(final_probs)
        final_class = CLASS_NAMES[final_idx]

        confidence = float(final_probs[final_idx])

        # ----------------------------
        # HEATMAP OVERLAY
        # ----------------------------
        img_np = np.array(image)

        heatmap_resized = cv2.resize(
            heatmap_np,
            (img_np.shape[1], img_np.shape[0])
        )

        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(
            np.ascontiguousarray(heatmap_color),
            cv2.COLORMAP_JET
        )

        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        superimposed_img = cv2.addWeighted(
            img_np, 0.6,
            heatmap_rgb, 0.4,
            0
        )

        # ----------------------------
        # SAVE IMAGE
        # ----------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{final_class}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)

        final_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_bgr)

        # ----------------------------
        # BASE64 OUTPUT
        # ----------------------------
        _, buffer = cv2.imencode(".jpg", final_bgr)
        img_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # ----------------------------
        # RESPONSE
        # ----------------------------
        return {
            "success": True,

            # FINAL FUSION RESULT
            "final_class": final_class,
            "confidence": confidence,

            # DEBUG INFO
            "cnn_prediction": CLASS_NAMES[int(np.argmax(cnn_probs))],
            "cnn_probs": cnn_probs.tolist(),
            "sensor_probs": sensor_probs.tolist(),

            # OUTPUT
            "local_path": save_path,
            "heatmap_base64": img_base64
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)