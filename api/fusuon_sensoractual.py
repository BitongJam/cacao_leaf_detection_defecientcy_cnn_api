import io
import os
import cv2
import time
import base64
import serial
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from datetime import datetime
from pathlib import Path

# =========================================================
# FASTAPI SETUP
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

print("✅ MODEL LOADED SUCCESSFULLY")

# =========================================================
# SERIAL SENSOR SETUP
# =========================================================
try:
    ser = serial.Serial(
        '/dev/serial0',   # change if needed
        baudrate=9600,
        timeout=1
    )

    print("✅ NPK SENSOR CONNECTED")

except Exception as e:
    print("❌ SENSOR CONNECTION ERROR:", e)

# MODBUS QUERY
npk_query = bytearray([
    0x01,
    0x03,
    0x00,
    0x1E,
    0x00,
    0x03,
    0x65,
    0xCD
])

# =========================================================
# HYBRID FUSION WEIGHTS
# =========================================================
ALPHA = 0.7   # CNN weight
BETA = 0.3    # SENSOR weight

# =========================================================
# READ REAL NPK SENSOR
# =========================================================
def get_npk_values():

    try:
        ser.write(npk_query)

        time.sleep(0.15)

        response = ser.read(11)

        print("RAW SENSOR RESPONSE:", response.hex())

        # VALIDATE RESPONSE
        if (
            len(response) == 11 and
            response[0] == 0x01 and
            response[1] == 0x03 and
            response[2] == 0x06
        ):

            nitrogen = (response[3] << 8) | response[4]
            phosphorus = (response[5] << 8) | response[6]
            potassium = (response[7] << 8) | response[8]

            return nitrogen, phosphorus, potassium

        else:
            print("❌ INVALID SENSOR RESPONSE")
            return None, None, None

    except Exception as e:
        print("❌ SENSOR READ ERROR:", e)
        return None, None, None

# =========================================================
# MAIN PREDICTION API
# =========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # =================================================
        # READ IMAGE
        # =================================================
        file_bytes = await file.read()

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        # =================================================
        # IMAGE PREPROCESSING
        # =================================================
        img_resized = image.resize((224, 224))

        img_array = np.array(img_resized).astype("float32")

        # NORMALIZATION
        img_array = img_array / 255.0

        img_batch = tf.expand_dims(img_array, 0)

        # =================================================
        # REAL SENSOR VALUES
        # =================================================
        n, p, k = get_npk_values()

        if n is None:
            return {
                "success": False,
                "error": "Failed to read NPK sensor"
            }

        print("\n========== SENSOR VALUES ==========")
        print("Nitrogen:", n)
        print("Phosphorus:", p)
        print("Potassium:", k)

        # =================================================
        # GRAD-CAM TARGET LAYER
        # =================================================
        try:
            target_layer = MODEL.get_layer(
                "last_conv_layer"
            )

        except:
            # fallback
            target_layer = MODEL.layers[-3]

        # =================================================
        # CREATE GRAD MODEL
        # =================================================
        grad_model = tf.keras.models.Model(
            inputs=MODEL.inputs,
            outputs=[
                target_layer.output,
                MODEL.output
            ]
        )

        # =================================================
        # CNN PREDICTION
        # =================================================
        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                img_batch
            )

            predicted_idx = tf.argmax(
                predictions[0]
            )

            loss = predictions[:, predicted_idx]

        # =================================================
        # CNN PROBABILITIES
        # =================================================
        cnn_probs = tf.nn.softmax(
            predictions[0]
        ).numpy()

        print("\n========== CNN PROBABILITIES ==========")
        print(cnn_probs)

        # =================================================
        # SENSOR NORMALIZATION
        # =================================================
        sensor_values = np.array([
            n / 100,
            p / 100,
            k / 100,
            0.5,
            0.5
        ])

        sensor_probs = sensor_values / (
            np.sum(sensor_values) + 1e-10
        )

        print("\n========== SENSOR PROBABILITIES ==========")
        print(sensor_probs)

        # =================================================
        # HYBRID SENSOR FUSION
        # =================================================
        final_probs = (
            ALPHA * cnn_probs
        ) + (
            BETA * sensor_probs
        )

        final_idx = np.argmax(final_probs)

        final_class = CLASS_NAMES[final_idx]

        confidence = float(
            final_probs[final_idx]
        )

        print("\n========== FINAL FUSION ==========")
        print("FINAL PROBS:", final_probs)
        print("FINAL CLASS:", final_class)
        print("CONFIDENCE:", confidence)

        # =================================================
        # GRAD-CAM
        # =================================================
        grads = tape.gradient(
            loss,
            conv_outputs
        )

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ tf.expand_dims(
            pooled_grads,
            -1
        )

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(
            heatmap,
            0
        ) / (
            tf.reduce_max(heatmap) + 1e-10
        )

        heatmap_np = heatmap.numpy()

        # =================================================
        # OVERLAY HEATMAP
        # =================================================
        img_np = np.array(image)

        heatmap_resized = cv2.resize(
            heatmap_np,
            (
                img_np.shape[1],
                img_np.shape[0]
            )
        )

        heatmap_color = np.uint8(
            255 * heatmap_resized
        )

        heatmap_color = cv2.applyColorMap(
            np.ascontiguousarray(
                heatmap_color
            ),
            cv2.COLORMAP_JET
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        superimposed_img = cv2.addWeighted(
            img_np,
            0.6,
            heatmap_rgb,
            0.4,
            0
        )

        # =================================================
        # SAVE OUTPUT IMAGE
        # =================================================
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = f"{timestamp}_{final_class}.jpg"

        save_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        final_bgr = cv2.cvtColor(
            superimposed_img,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            save_path,
            final_bgr
        )

        print("\n✅ HEATMAP SAVED:", save_path)

        # =================================================
        # BASE64 ENCODE
        # =================================================
        _, buffer = cv2.imencode(
            '.jpg',
            final_bgr
        )

        heatmap_base64 = base64.b64encode(
            buffer.tobytes()
        ).decode('utf-8')

        # =================================================
        # API RESPONSE
        # =================================================
        return {

            "success": True,

            # FINAL RESULT
            "final_class": final_class,
            "confidence": confidence,

            # CNN ONLY
            "cnn_prediction":
                CLASS_NAMES[
                    int(np.argmax(cnn_probs))
                ],

            # SENSOR VALUES
            "sensor_values": {
                "nitrogen": int(n),
                "phosphorus": int(p),
                "potassium": int(k)
            },

            # DEBUGGING
            "cnn_probs":
                cnn_probs.tolist(),

            "sensor_probs":
                sensor_probs.tolist(),

            "final_probs":
                final_probs.tolist(),

            # IMAGE
            "local_path":
                save_path,

            "heatmap_base64":
                heatmap_base64
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