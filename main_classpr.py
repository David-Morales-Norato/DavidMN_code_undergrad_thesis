'''
import the necessary packages
'''

from utils.dataio import get_dataset
from model.FinalModel import CLAS_PR, CLAS_PR_BACK, CLAS_PR_INIT
from utils.callbacks import log_predictions
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import json
import sys
import os


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

def main(dataset_name, num_classes, model_type, batch_size, epochs, lr, shape, p_value, k_size, n_iter, clasification_network, results_folder, forward_params, cut_dataset = False):
    # other
    float_dtype =  "float32"
    complex_dtype =  "complex64"

    
    ''' 
    Data preprocessing
    '''

    SNAPSHOTS =  forward_params["number_snapshots"]
    TIPO_MUESTREO =  forward_params["tipo_muestreo"]
    WAVE_LENGTH =  forward_params["wavelength"]
    DX =  forward_params["pixel_size"]
    DISTANCE_SENSOR =  forward_params["distance_sensor"]
    normalize_measurements = forward_params["normalize_measurements"]
    SNR = forward_params["snr"]

    results_folder =  os.path.join(results_folder, TIPO_MUESTREO,  model_type, clasification_network, dataset_name)
    tensorboard_path = os.path.join(results_folder, "tensorboard")
    WEIGHTS_PATH =  os.path.join(results_folder, "checkpoint.h5")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if (dataset_name == "mnist" or dataset_name == "fashion_mnist"):
        train_dataset, val_dataset = get_dataset(dataset_name, shape = shape, batch_size = batch_size, num_classes = 10, complex_dtype=complex_dtype)
        if cut_dataset == True:
            train_dataset = train_dataset.take(10)
            val_dataset = val_dataset.take(10)
    else:
        raise Exception("invalid type dataset")

    plot_init = True
    '''
    Model definition
    '''
    if model_type == "none":
        print(" ---------------- using model without any initialization ---------------- ")
        modelo_class = CLAS_PR(shape = shape, num_classes = num_classes, clasification_network = clasification_network, snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
                dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
                normalize_measurements = normalize_measurements, 
                float_dtype = float_dtype, complex_dtype = complex_dtype, snr = SNR)
        modelo_class.build((batch_size, *shape, 2))
        plot_init = False
    elif model_type == "back":
        print(" ---------------- using model using the back propagation operator ---------------- ")
        modelo_class = CLAS_PR_BACK(shape = shape, num_classes = num_classes, clasification_network = clasification_network, snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
                dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
                normalize_measurements = normalize_measurements, 
                float_dtype = float_dtype, complex_dtype = complex_dtype, snr = SNR)
        modelo_class.build((batch_size, *shape, 2))
    elif model_type == "fsi":
        print(" ---------------- using model using the FSI initialization algorithm---------------- ")
        modelo_class = CLAS_PR_INIT(shape = shape, num_classes = num_classes, p = p_value, k_size = k_size, n_iterations = n_iter, 
                clasification_network = clasification_network, snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
                normalize_measurements = normalize_measurements, 
                float_dtype = float_dtype, complex_dtype = complex_dtype, snr = SNR)
        modelo_class.build((batch_size, *shape, 2))
    else:
        raise Exception("invalid model type")

    print(modelo_class.summary())

    '''
    metrics and callbacks
    '''
    train_img =     next(iter(train_dataset))
    val_img =     next(iter(val_dataset))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
    fw_train = tf.summary.create_file_writer(os.path.join(tensorboard_path, 'predictions_train'))
    fw_val = tf.summary.create_file_writer(os.path.join(tensorboard_path, 'predictions_val'))

    callback_predictions_train = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_predictions(epoch, logs, model = modelo_class, val_images = train_img, fw_results = fw_train, plot_init = plot_init))
    callback_predictions_val = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_predictions(epoch, logs, model = modelo_class, val_images = val_img, fw_results = fw_val, plot_init = plot_init))
    check_point = tf.keras.callbacks.ModelCheckpoint(
                                    WEIGHTS_PATH,
                                    monitor="val_loss",
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode="min",
                                    save_freq="epoch")
    callbacks = [tensorboard_callback, callback_predictions_train,callback_predictions_val, check_point]
    #callbacks = [tensorboard_callback, check_point]


    '''
    training loop
    '''
    opti = tf.keras.optimizers.Adam(amsgrad = True, learning_rate = lr)
    modelo_class.compile(optimizer = opti, loss = "categorical_crossentropy", metrics = ["categorical_crossentropy", tf.keras.metrics.Recall(name = "recall_1"), tf.keras.metrics.Precision(name = "precision_1"), tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average = "macro")])
    modelo_class.fit(x = train_dataset, validation_data=val_dataset, epochs=epochs, callbacks = callbacks, verbose=1)

if __name__ == "__main__":
    datasets_name = ["fashion_mnist"]#["fashion_mnist", "mnist"]
    num_classes = 10
    model_types = ["none"]#["none", "back", "fsi"]
    classifiers = ["mobilnetv2"]
    batch_size = 5
    epochs = 5
    lr = 1e-3
    shape = [128, 128]
    p_value = 6
    k_size = 5
    n_iter = 3
    results_folder = "results"

    config_file = sys.argv[-1]
    asm_params, fresnel_params, fran_params = read_config(config_file)

    for forward_params in [asm_params, fresnel_params, fran_params]:
        for model_type in model_types:
            for clasification_network in classifiers:
                for dataset_name in datasets_name:
                    print("##########################################")
                    print("RUNING EXP", forward_params["tipo_muestreo"], model_type, clasification_network, dataset_name)
                    main(dataset_name, num_classes, model_type, batch_size, epochs, lr, shape, p_value, k_size, n_iter, clasification_network, results_folder, forward_params, cut_dataset = True)
                    print("##########################################")