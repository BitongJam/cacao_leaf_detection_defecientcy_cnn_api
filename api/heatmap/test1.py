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

app = FastAPI()

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Local Storage Setup
# Dinhi i-save ang mga heatmap pictures sa imong server
OUTPUT_DIR = "detected_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Load Model
# Siguroha nga ang '1.keras' kay Functional model na
MODEL_PATH = "/opt/cacao_leaf_detection_defecientcy_cnn_api/models/1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

print("✅ Model loaded successfully. Ready for predictions.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # --- A. PREPROCESSING ---
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # Resize image para sa model (256x256)
        img_resized = image.resize((256, 256))
        img_array = np.array(img_resized).astype("float32")
        
        # E-expand ang dimensions (1, 256, 256, 3) ug himoong Tensor
        img_batch = tf.expand_dims(img_array, 0)

        # --- B. GRAD-CAM HEATMAP LOGIC ---
        
        # Gamita ang layer name nga imong gi-set sa training (e.g., 'last_conv_layer')
        # Kung wala nimo na-set, gamita ang 'conv2d_5' o unsa may naa sa model.summary()
        try:
            target_layer = MODEL.get_layer("last_conv_layer")
        except:
            # Fallback kung wala na-rename ang layer
            target_layer = MODEL.get_layer("conv2d_5")

        # Paghimo og Gradient Model
        grad_model = tf.keras.Model(
            inputs=MODEL.inputs,
            outputs=[target_layer.output, MODEL.output]
        )

        # I-trace ang gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            predicted_idx = tf.argmax(predictions[0])
            loss = predictions[:, predicted_idx]

        # Kuhaon ang gradients relative sa conv output
        grads = tape.gradient(loss, conv_outputs)
        
        # Global Average Pooling sa gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # I-multiply ang feature map sa iyang importance weights
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU ug Normalization (0 hangtod 1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        # --- C. SUPERIMPOSE (OVERLAY) ---
        
        # I-convert ang original high-res image sa numpy
        img_np = np.array(image)
        
        # I-resize ang heatmap para motakdo sa original photo size
        heatmap_resized = cv2.resize(heatmap_np, (img_np.shape[1], img_np.shape[0]))
        
        # I-apply ang COLORMAP_JET (Blue to Red)
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        
        # Convert BGR (OpenCV default) to RGB para sa blending
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # I-blend: 60% original image, 40% heatmap color
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color_rgb, 0.4, 0)

        # --- D. SAVE TO LOCAL STORAGE ---
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = CLASS_NAMES[predicted_idx].replace(" ", "_")
        filename = f"{timestamp}_{label}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # Kinahanglan i-convert balik sa BGR para sa imwrite
        final_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_bgr)

        # --- E. ENCODE TO BASE64 (PARA SA FRONTEND) ---
        
        _, buffer = cv2.imencode('.jpg', final_bgr)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "class": CLASS_NAMES[predicted_idx],
            "confidence": float(np.max(predictions[0])),
            "local_path": save_path,
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # I-run sa port 3000
    uvicorn.run(app, host="0.0.0.0", port=3000)