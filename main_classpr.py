'''
import the necessary packages
'''

from utils.dataio import get_dataset
from model.FinalModel import CLAS_PR, CLAS_PR_BACK, CLAS_PR_INIT
from utils.callbacks import log_confusion_depth
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import json
import sys
import os

''' 
parameters
'''
config_file = sys.argv[-1]
if os.path.exists(config_file):
    print("opening config file")
    with open(config_file) as json_file:
        params = json.load(json_file)
        data_params, forward_params, training_params, other_params = params["data"], params["forward_model"], params["training"], params["other"]
else:
    print("Config file", config_file)
    print("actual path", os.getcwd())
    raise Exception("Config file not found: "+ config_file)

# data
SHAPE = data_params["image_shape"]
DATASET_NAME = data_params["dataset_name"]
NUM_CLASES = data_params["num_classes"]


# Forward model parameters
SNAPSHOTS =  forward_params["number_snapshots"]
TIPO_MUESTREO =  forward_params["tipo_muestreo"]
WAVE_LENGTH =  forward_params["wavelength"]
DX =  forward_params["pixel_size"]
DISTANCE_SENSOR =  forward_params["distance_sensor"]
NORMALIZE_MEASUREMENTS = forward_params["normalize_measurements"]
SNR = forward_params["snr"]

# Training 
MODEL_TYPE = training_params["model_type"]
BATCH_SIZE =  training_params["batch_size"]
EPOCHS =  training_params["number_epochs"]
LR =  training_params["learning_rate"]

RESULTS_FOLDER =  os.path.join(training_params["results_folder"], MODEL_TYPE, DATASET_NAME)
tensorboard_path = os.path.join(RESULTS_FOLDER, "tensorboard")
WEIGHTS_PATH =  os.path.join(RESULTS_FOLDER, training_params["weights_path"])

if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
    
# other
FLOAT_DTYPE =  "float32"
COMPLEX_DTYPE =  "complex64"

P =  other_params["p"]
K_SIZE = other_params["k_size"]
N_ITER = other_params["n_iterations"]


''' 
Data preprocessing
'''
if (DATASET_NAME == "mnist" or DATASET_NAME == "fashion_mnist"):
    train_dataset, val_dataset = get_dataset(DATASET_NAME, shape = SHAPE, batch_size = BATCH_SIZE, num_classes = 10, complex_dtype=COMPLEX_DTYPE)
    train_dataset = train_dataset.take(64)
    val_dataset = val_dataset.take(64)
else:
    raise Exception("invalid type dataset")


'''
Model definition
'''
if MODEL_TYPE == "None":
    print(" ---------------- using model without any initialization ---------------- ")
    modelo_class = CLAS_PR(shape = SHAPE, num_classes = NUM_CLASES, snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
            dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
            normalize_measurements = NORMALIZE_MEASUREMENTS, 
            float_dtype = FLOAT_DTYPE, complex_dtype = COMPLEX_DTYPE, snr = SNR)
    modelo_class.build((BATCH_SIZE, *SHAPE, 2))
elif MODEL_TYPE == "back_propagation":
    print(" ---------------- using model using the back propagation operator ---------------- ")
    modelo_class = CLAS_PR_BACK(shape = SHAPE, num_classes = NUM_CLASES, snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
            dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
            normalize_measurements = NORMALIZE_MEASUREMENTS, 
            float_dtype = FLOAT_DTYPE, complex_dtype = COMPLEX_DTYPE, snr = SNR)
    modelo_class.build((BATCH_SIZE, *SHAPE, 2))
elif MODEL_TYPE == "fsi_initialization":
    print(" ---------------- using model using the FSI initialization algorithm---------------- ")
    modelo_class = CLAS_PR_INIT(shape = SHAPE, num_classes = NUM_CLASES, p = P, k_size = K_SIZE, n_iterations = N_ITER, 
            snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
            normalize_measurements = NORMALIZE_MEASUREMENTS, 
            float_dtype = FLOAT_DTYPE, complex_dtype = COMPLEX_DTYPE, snr = SNR)
    modelo_class.build((BATCH_SIZE, *SHAPE, 2))
else:
    raise Exception("invalid model type")

print(modelo_class.summary())

'''
metrics and callbacks
'''
train_img =     next(iter(train_dataset))
val_img =     next(iter(val_dataset))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
# fw_train = tf.summary.create_file_writer(os.path.join(tensorboard_path, 'confusion_train'))
# fw_val = tf.summary.create_file_writer(os.path.join(tensorboard_path, 'confusion_val'))
# callback_confusion_train = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_confusion_depth(epoch, logs, model = modelo_class, val_img = train_img, fw_results = fw_train, name = "train"))
# callback_confusion_val = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_confusion_depth(epoch, logs, model = modelo_class, val_img = val_img, fw_results = fw_val, name = "val"))
check_point = tf.keras.callbacks.ModelCheckpoint(
                                WEIGHTS_PATH,
                                monitor="val_loss",
                                save_best_only=True,
                                save_weights_only=True,
                                mode="min",
                                save_freq="epoch")
# callbacks = [tensorboard_callback, callback_confusion_train,callback_confusion_val, check_point]
callbacks = [tensorboard_callback, check_point]


'''
training loop
'''
opti = tf.keras.optimizers.Adam(amsgrad = True, learning_rate = LR)
modelo_class.compile(optimizer = opti, loss = "categorical_crossentropy", metrics = ["categorical_crossentropy", tf.keras.metrics.Recall(name = "recall_1"), tf.keras.metrics.Precision(name = "precision_1"), tfa.metrics.F1Score(num_classes=NUM_CLASES, threshold=0.5, average = "macro")])
modelo_class.fit(x = train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks = callbacks, verbose=2)