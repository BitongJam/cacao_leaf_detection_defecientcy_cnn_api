import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
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
DATASET_PATH = "/home/odoo/cnn_project/Dataset/cacao-dataset"

# =========================================================
# 1. LOAD DATASET (TRAIN + VAL SPLIT BUILT-IN)
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

# Split validation into validation + test
val_batches = tf.data.experimental.cardinality(val_test_ds).numpy()

val_ds = val_test_ds.take(val_batches // 2)
test_ds = val_test_ds.skip(val_batches // 2)

# =========================================================
# 2. DATA PIPELINE OPTIMIZATION
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================================================
# 3. DATA AUGMENTATION (TRAIN ONLY)
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(AUTOTUNE)

# =========================================================
# 4. BUILD MODEL
# =========================================================

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

x = layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(inputs)
x = layers.Rescaling(1./255)(x)

x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation='relu', name="last_conv_layer")(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

outputs = layers.Dense(n_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

# =========================================================
# 5. COMPILE MODEL
# =========================================================

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# =========================================================
# 6. TRAIN MODEL
# =========================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose=1
)

# =========================================================
# 7. PLOT RESULTS
# =========================================================

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title("Loss")

os.makedirs("./graphs", exist_ok=True)
plt.savefig("./graphs/training_metrics.png", dpi=300)

print("\n✅ Saved training graph")

plt.show()

# =========================================================
# 8. EVALUATION (FIXED)
# =========================================================

print("\nEvaluating test dataset...\n")

results = model.evaluate(test_ds)
print("Test Results:", results)

# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================================================
# 10. SAVE MODEL
# =========================================================

os.makedirs("./models", exist_ok=True)

model_version = max(
    [int(i) for i in os.listdir("./models") if i.isdigit()] + [0]
) + 1

model_path = f"./models/{model_version}.keras"
model.save(model_path)

print(f"\n✅ Model saved at {model_path}")