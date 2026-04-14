import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
DATASET_PATH = "/opt/cacao_leaf_detection_defecientcy_cnn_api/Dataset/villageplant"

# 1. Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
n_classes = len(class_names)

# 2. Split dataset function
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# 3. Configure datasets
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# 4. Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 5. BUILD MODEL USING FUNCTIONAL API (MAO NI ANG FIX) ---

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="original_input")

# Preprocessing layers
x = layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(inputs)
x = layers.Rescaling(1./255)(x)

# Convolutional blocks
x = layers.Conv2D(32, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# IMPORTANT: Last Conv layer with explicit name for Grad-CAM
x = layers.Conv2D(64, (3, 3), activation='relu', name="last_conv_layer")(x)
x = layers.MaxPooling2D((2, 2))(x)

# Classifier head
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)

# Create the Functional Model
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Cacao_Functional_Model")

# --- 6. COMPILE AND TRAIN ---

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    verbose=1,
    epochs=50,
)

# 7. Save the model
os.makedirs("./models", exist_ok=True)
model_version = max([int(i) for i in os.listdir("./models") if i.isdigit()] + [0]) + 1
model.save(f"./models/{model_version}.keras")
print(f"✅ Functional Model saved as ./models/{model_version}.keras")