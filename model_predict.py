import tensorflow as tf
from tensorflow import keras
from Cats_VS_Dogs.config import cfg
import numpy as np

model = keras.models.load_model(cfg.save_mode_path)

imgfile_path = "./val_imgs/2_0.png"
img = keras.preprocessing.image.load_img(imgfile_path, color_mode="grayscale", target_size=(256, 256))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, 0)   # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predic_cls = np.argmax(score)
predic_scr = 100 * np.max(score)
print(predic_cls)
print(predic_scr)