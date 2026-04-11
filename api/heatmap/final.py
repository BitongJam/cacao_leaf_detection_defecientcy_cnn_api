import io
import base64
import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the model as is
MODEL_PATH = "/opt/cacao_leaf_detection_defecientcy_cnn_api/models/1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Read and Preprocess Image
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_resized = image.resize((256, 256))
        img_array = np.array(img_resized).astype("float32")
        img_batch = tf.expand_dims(img_array, 0) # Convert to Tensor dayon

        # 2. Grad-CAM Logic sulod sa usa ka Tape
        # Dili na nato i-wrap ang model. Atong gamiton ang MODEL direkta.
        target_layer = MODEL.get_layer("conv2d_4")
        
        # Paghimo og Sub-Model para sa Heatmap lang
        grad_model = tf.keras.Model(
            inputs=MODEL.inputs, 
            outputs=[target_layer.output, MODEL.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            predicted_idx = tf.argmax(predictions[0])
            loss = predictions[:, predicted_idx]

        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global Average Pooling sa gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply feature map sa weights
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU & Normalization
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        # 3. Superimpose (Overlay)
        img_np = np.array(image)
        heatmap_resized = cv2.resize(heatmap_np, (img_np.shape[1], img_np.shape[0]))
        
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        superimposed = cv2.addWeighted(img_np, 0.6, heatmap_color_rgb, 0.4, 0)
        
        # Encode Result
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'class': CLASS_NAMES[predicted_idx],
            'confidence': float(np.max(predictions[0])),
            'heatmap': heatmap_base64
        }

    except Exception as e:
        # I-print sa terminal para makita nimo ang detalye
        print(f"DEBUG ERROR: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=3000)