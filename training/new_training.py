import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pywt
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# =========================================================
# GPU CONFIG
# =========================================================

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# =========================================================
# CONSTANTS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
DATASET_PATH = "/home/odoo/cnn_project/Dataset/cacao_dataset_train1"

# =========================================================
# LOAD DATASET
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
n_classes = len(class_names)

print("Classes:", class_names)

val_batches = tf.data.experimental.cardinality(val_test_ds).numpy()

val_ds = val_test_ds.take(val_batches // 2)
test_ds = val_test_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================================================
# DATA AUGMENTATION
# =========================================================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

train_ds = train_ds.map(
    lambda x, y: (augmentation(x, training=True), y)
).prefetch(AUTOTUNE)

# =========================================================
# WAVELET LAYER
# =========================================================

@tf.keras.utils.register_keras_serializable()
class WaveletLayer(layers.Layer):

    def __init__(self):
        super(WaveletLayer, self).__init__()

    def call(self, inputs):

        def wavelet_transform(img):

            img = img.numpy()
            coeffs = []

            for channel in range(img.shape[-1]):

                cA, (cH, cV, cD) = pywt.dwt2(img[:, :, channel], 'haar')

                merged = np.concatenate([
                    cA,
                    cH,
                    cV,
                    cD
                ], axis=1)

                coeffs.append(merged)

            coeffs = np.stack(coeffs, axis=-1)
            return coeffs.astype(np.float32)

        output = tf.map_fn(
            lambda x: tf.py_function(
                wavelet_transform,
                [x],
                Tout=tf.float32
            ),
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(112, 448, 3), dtype=tf.float32)
        )

        return output

    def get_config(self):
        config = super(WaveletLayer, self).get_config()
        return config

# =========================================================
# WT RESIDUAL BLOCK
# =========================================================

@tf.keras.utils.register_keras_serializable()
class WTResidualBlock(layers.Layer):

    def __init__(self, filters):
        super(WTResidualBlock, self).__init__()
        self.filters = filters

        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.relu = layers.ReLU()

    def call(self, inputs):

        shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + shortcut

        x = self.relu(x)

        return x

# =========================================================
# BUILD WT-RESNET MODEL
# =========================================================

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

x = layers.Rescaling(1./255)(inputs)

# =========================================================
# APPLY WAVELET TRANSFORM
# =========================================================

x = WaveletLayer()(x)

# =========================================================
# INITIAL CONVOLUTION
# =========================================================

x = layers.Conv2D(32, 5, padding='same', activation='relu')(x)
x = layers.MaxPooling2D()(x)

# =========================================================
# WT RESIDUAL BLOCKS
# =========================================================

x = WTResidualBlock(32)(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = WTResidualBlock(64)(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = WTResidualBlock(128)(x)
x = layers.MaxPooling2D()(x)

# =========================================================
# GLOBAL AVERAGE POOLING
# =========================================================

x = layers.GlobalAveragePooling2D()(x)

# =========================================================
# DENSE LAYERS
# =========================================================

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(n_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.summary()

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# TRAIN MODEL
# =========================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose=1
)

# =========================================================
# EVALUATE MODEL
# =========================================================

results = model.evaluate(test_ds)
print("Test Results:", results)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

y_true = []
y_pred = []

for images, labels in test_ds:

    preds = model.predict(images, verbose=0)

    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred, target_names=class_names))

# =========================================================
# SAVE MODEL
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
model_dir = ROOT_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model.save(model_dir / "wt_resnet_cacao.keras")

print(f"WT-ResNet model saved to {model_dir / 'wt_resnet_cacao.keras'}")
