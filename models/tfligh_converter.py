import tensorflow as tf

# load original keras model
model = tf.keras.models.load_model("models/1.keras")

# convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save output
with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")