import tensorflow as tf
import pathlib

train_tfrecord_file = "train.tfrecords"


# 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def _parse_example_img(example_string):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description_ = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description_)

    images = feature_dict['image']
    images = tf.io.decode_raw(images, tf.uint8)
    images = tf.reshape(images, [112, 112, 1])  # 灰度图
    images = tf.cast(images, tf.float32) / 255.0

    return images


def gen_quan_data(file_pattern, num_samples):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_example_img)
    dataset = dataset.batch(1)
    dataset = dataset.take(num_samples)
    return dataset


def representative_data_gen():
    for input_value in gen_quan_data(train_tfrecord_file, 1000):
        # Model has only one input so each data point has one element.
        yield [input_value]


model = tf.keras.models.load_model("cats_vs_dogs.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model_quant = converter.convert()

# 保存量化后的模型
tflite_models_dir = pathlib.Path("./quan_model")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_quant_file = tflite_models_dir/"cat_vs_dog_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
