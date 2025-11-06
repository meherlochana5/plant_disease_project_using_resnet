# predict_resnet.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# load your saved model
model = tf.keras.models.load_model("resnet50_final_90plus.h5")

# your class names (same order that model used)
class_names = [
    'Potato___Early_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# test image name
img_path = "p.jpg"     # give your test image name here

# preprocessing
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# IMPORTANT: use resnet preprocessing (more accurate)
x = tf.keras.applications.resnet50.preprocess_input(x)

# prediction
pred = model.predict(x)
index = np.argmax(pred)
confidence = np.max(pred) * 100

print("\nPredicted Disease:", class_names[index])
print(f"Confidence: {confidence:.2f}%\n")
