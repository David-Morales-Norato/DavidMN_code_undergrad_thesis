import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json

def preprocess(image, label, shape, num_classes, complex_dtype):

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
    
def get_dataset(dataset_name, shape, num_classes, batch_size, complex_dtype):

    ds_train, ds_val, ds_test = tfds.load(dataset_name, split=['train[:90%]', 'train[90%:]', 'test'], shuffle_files=True, as_supervised=True, with_info=False, batch_size=batch_size)
    #test_images = tfds.load(dataset_name, split='test', as_supervised=True, batch_size=batch_size)

    train_images = ds_train.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype), num_parallel_calls=tf.data.AUTOTUNE)
    val_images = ds_val.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype), num_parallel_calls=tf.data.AUTOTUNE)
    test_images = ds_test.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype), num_parallel_calls=tf.data.AUTOTUNE)
    #test_images = test_images.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype))


    train_images = train_images.cache()
    train_images = train_images.shuffle(len(ds_train))
    train_images = train_images.prefetch(tf.data.AUTOTUNE)
    #train_images = train_images.batch(batch_size)

    val_images = val_images.cache()
    val_images = val_images.shuffle(len(ds_train))
    val_images = val_images.prefetch(tf.data.AUTOTUNE)
    #val_images = val_images.batch(batch_size)

    test_images = test_images.cache()
    test_images = test_images.shuffle(len(ds_test))
    test_images = test_images.prefetch(tf.data.AUTOTUNE)
    #test_images = test_images.batch(batch_size)
    
    return train_images, val_images, test_images



def get_kfolds_dataset(dataset_name, shape, num_classes, batch_size, complex_dtype, kfolds = 10):


    ds_test = tfds.load(dataset_name, split='test', shuffle_files=True, as_supervised=True, with_info=False, batch_size=batch_size)

    #train_images_clean = ds_train.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype), num_parallel_calls=tf.data.AUTOTUNE)
    test_images = ds_test.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype), num_parallel_calls=tf.data.AUTOTUNE)

    #train_images_clean = train_images_clean.shuffle(len(ds_train))

    test_images = test_images.cache()
    test_images = test_images.shuffle(len(ds_test))
    test_images = test_images.prefetch(tf.data.AUTOTUNE)
    #test_images = test_images.batch(batch_size)
    train_k_datasets = []
    val_k_datasets = []


    for k in range(kfolds):
        train_range = 'train[10%:]'
        val_range = 'train[:10%]' 
        ds_train, ds_val = tfds.load(dataset_name, split=[train_range, val_range], shuffle_files=True, as_supervised=True, with_info=False, batch_size=batch_size)

        train_images = ds_train.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype), num_parallel_calls=tf.data.AUTOTUNE)
        val_images = ds_val.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype), num_parallel_calls=tf.data.AUTOTUNE)
        
        #test_images = test_images.map(lambda x, y: preprocess(x, y, shape, num_classes, complex_dtype))

        train_images = train_images.cache()
        train_images = train_images.shuffle(len(ds_train))
        train_images = train_images.prefetch(tf.data.AUTOTUNE)
        #train_images = train_images.batch(batch_size)

        val_images = val_images.cache()
        val_images = val_images.shuffle(len(ds_train))
        val_images = val_images.prefetch(tf.data.AUTOTUNE)
        #val_images = val_images.batch(batch_size)
        yield train_images, val_images, test_images
    #     train_k_datasets.append(train_images)
    #     val_k_datasets.append(val_images)
    
    # return train_k_datasets, val_k_datasets, test_images
    
def read_config(config_file):

    if os.path.exists(config_file):
        print("opening config file")
        with open(config_file) as json_file:
            params = json.load(json_file)
            asm_params, fresnel_params, fran_params = params["asm"], params["fresnel"], params["fran"]
    else:
        print("Config file", config_file)
        print("actual path", os.getcwd())
        raise Exception("Config file not found: "+ config_file)
    return asm_params, fresnel_params, fran_params


if __name__ == "__main__":
    datasets = get_kfolds_dataset("mnist", [32, 32], 10, 5, complex_dtype = tf.complex128, kfolds = 10)

    for indx_dataset, (dataset) in enumerate(datasets):
        train_dataset, val_dataset, test_dataset = dataset

        print("asd")
