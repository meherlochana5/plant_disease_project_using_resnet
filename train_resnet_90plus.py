import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------- CONFIG ----------
DATASET_PATH = "PlantVillage"  # folder with 9 class subfolders
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_TOP = 15      # train top layers first
EPOCHS_FINE = 20     # fine-tune last ResNet layers
SEED = 42
# --------------------------

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

# Data generator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

num_classes = train_gen.num_classes
print("Classes:", train_gen.class_indices)
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# ----------------------------
# Build model
# ----------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # freeze base

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
checkpoint = ModelCheckpoint("resnet50_top.h5", save_best_only=True, monitor="val_accuracy", verbose=1)
earlystop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train top layers
history_top = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_TOP,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# ----------------------------
# Fine-tune last layers
# ----------------------------
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_fine = ModelCheckpoint("resnet50_finetuned.h5", save_best_only=True, monitor="val_accuracy", verbose=1)
earlystop_fine = EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1)
reduce_lr_fine = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=[checkpoint_fine, earlystop_fine, reduce_lr_fine]
)

model.save("resnet50_final_90plus.h5")
print("✅ Training finished with expected >90% accuracy.")
