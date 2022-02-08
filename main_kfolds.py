from utils.dataio import read_config, get_kfolds_dataset
from model.FinalModel import CLAS_PR, CLAS_PR_BACK, CLAS_PR_INIT
from utils.callbacks import log_predictions
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import sys
import os


def main_kfolds(dataset_name, num_classes, model_type, batch_size, epochs, lr, shape, p_value, k_size, n_iter, clasification_network, results_folder, forward_params, cut_dataset = False, set_gpu = None, k_folds = 10):
    # other
    if set_gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(set_gpu))])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


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

    

    if (dataset_name == "mnist" or dataset_name == "fashion_mnist"):
        train_list_datasets, val_list_datasets, test_dataset = get_kfolds_dataset(dataset_name, shape = shape, batch_size = batch_size, num_classes = 10, complex_dtype=complex_dtype, kfolds = k_folds)

        if cut_dataset == True:
            for indx_dataset, (train_ds, val_ds) in enumerate(zip(train_list_datasets, val_list_datasets)):
                train_list_datasets[indx_dataset] = train_ds.take(10)
                val_list_datasets[indx_dataset] = val_ds.take(10)
            test_dataset = test_dataset.take(10)
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
    for indx_dataset, (train_dataset, val_dataset) in enumerate(zip(train_list_datasets, val_list_datasets)):
        results_folder =  os.path.join(results_folder, TIPO_MUESTREO,  model_type, clasification_network, dataset_name, "fold_" + str(indx_dataset))
        tensorboard_path = os.path.join(results_folder, "tensorboard")
        WEIGHTS_PATH =  os.path.join(results_folder, "checkpoint.h5")
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
    
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
        modelo_class.compile(optimizer = opti, loss = "categorical_crossentropy", metrics = ["accuracy", tf.keras.metrics.Recall(name = "recall_1"), tf.keras.metrics.Precision(name = "precision_1"), tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average = "macro")])
        history = modelo_class.fit(x = train_dataset, validation_data=val_dataset, epochs=epochs, callbacks = callbacks, verbose=1, batch_size=batch_size)

        df = pd.DataFrame.from_dict(history.history)
        df.to_csv(os.path.join(results_folder, "history.csv"), index = False)
        modelo_class.load_weights(WEIGHTS_PATH)
        evaluatation_results = [modelo_class.evaluate(test_dataset)]
        
        df = pd.DataFrame(evaluatation_results)
        df.to_csv(os.path.join(results_folder, "test.csv"), index = False, header =modelo_class.metrics_names)



if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #datasets_name = ["fashion_mnist", "mnist"]
    num_classes = 10
    model_types = ["fsi", "none", "back"]
    classifier = "mobilnet"#["mobilnet", "xception", "inception"]
    batch_size = 6
    epochs = 1
    lr = 1e-3
    shape = [32, 32]
    p_value = 6
    k_size = 5
    n_iter = 15
    results_folder = "results_kfold"

    config_file = sys.argv[-1]
    dataset_name = sys.argv[-2]
    asm_params, _, _ = read_config(config_file)

    for model_type in model_types:

        print("##########################################")
        print("RUNING EXP", asm_params["tipo_muestreo"], model_type, classifier, dataset_name)
        main_kfolds(dataset_name, num_classes, model_type, batch_size, epochs, lr, shape, p_value, k_size, n_iter, classifier, results_folder, asm_params, cut_dataset = True, set_gpu = 10*1024)
        print("##########################################")