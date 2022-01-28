import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def get_dataset(dataset_name, shape, num_classes, batch_size, complex_dtype):
    def preprocess(data, shape = shape, num_classes = num_classes):

        image = data["image"]
        label = data["label"]
        x = tf.cast(image,tf.float32) /255
        x = tf.image.resize(x, shape)
        mins = tf.math.reduce_min(x,axis=[1,2], keepdims=True)
        maxs = tf.math.reduce_max(x,axis=[1,2], keepdims=True)
        x = (x - mins)/(maxs-mins)* 2 * np.pi - np.pi
        x = tf.ones(tf.shape(x), dtype=complex_dtype)*tf.math.exp(tf.cast(x, dtype=complex_dtype)*tf.constant(1j, dtype=complex_dtype))
        x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
        label = tf.one_hot(tf.cast(label, tf.int32), num_classes)
        return x,label



    train_images, val_images = tfds.load(dataset_name, split=['train[:90%]', 'train[90%:]'], batch_size=batch_size)
    #test_images = tfds.load(dataset_name, split='test', as_supervised=True, batch_size=batch_size)

    train_images = train_images.map(preprocess)
    val_images = val_images.map(preprocess)
    #test_images = test_images.map(preprocess)

    return train_images, val_images