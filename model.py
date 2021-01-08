from tensorflow import keras
from tensorflow.keras import layers

num_classes = 2


def dogs_vs_cats_sequential_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(112, 112, 1)))
    model.add(layers.Conv2D(16, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3,  activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(16, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))
    return model


def dogs_vs_cats_function_model():
    input_img = keras.Input(shape=(112, 112, 1))
    x = layers.Conv2D(16, 3, activation='relu')(input_img)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3,  activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    model_output = layers.Dense(num_classes)(x)
    model = keras.Model(input_img, model_output)
    return model


# mymodel = dogs_vs_cats_sequential_model()
# mymodel.summary()