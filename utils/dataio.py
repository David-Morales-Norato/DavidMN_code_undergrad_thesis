import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def get_dataset(dataset_name, shape, num_classes, batch_size, complex_dtype):
    def preprocess(image, label, shape = shape, num_classes = num_classes):

        # image = data["image"]
        # label = data["label"]
        x = tf.cast(image,tf.float32) /255
        x = tf.image.resize(x, shape)
        mins = tf.math.reduce_min(x,axis=[1,2], keepdims=True)
        maxs = tf.math.reduce_max(x,axis=[1,2], keepdims=True)
        x = (x - mins)/(maxs-mins)* 2 * np.pi - np.pi
        x = tf.ones(tf.shape(x), dtype=complex_dtype)*tf.math.exp(tf.cast(x, dtype=complex_dtype)*tf.constant(1j, dtype=complex_dtype))
        x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
        label = tf.one_hot(tf.cast(label, tf.int32), num_classes)
        return x,label



    (ds_train, ds_test), ds_info = tfds.load(dataset_name, split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True, batch_size=batch_size)
    #test_images = tfds.load(dataset_name, split='test', as_supervised=True, batch_size=batch_size)

    train_images = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    #test_images = test_images.map(preprocess)


    train_images = train_images.cache()
    train_images = train_images.shuffle(ds_info.splits['train'].num_examples)
    #train_images = train_images.batch(128)
    train_images = train_images.prefetch(tf.data.AUTOTUNE)

    test_images = test_images.cache()
    test_images = test_images.shuffle(ds_info.splits['test'].num_examples)
    #test_images = test_images.batch(128)
    test_images = test_images.prefetch(tf.data.AUTOTUNE)
    return train_images, test_images