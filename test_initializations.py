'''
import the necessary packages
'''
from utils.dataio import get_dataset
from model.AcquisitionLayer import Muestreo
from model.InitializationLayer import BackPropagationLayer, FSI_Initial, FSI_cell
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import sys
import os

gpus = tf.config.list_physical_devices('GPU')

if len(gpus)> 0:
    print("RUNNING ON GPU")
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)   
else: 
    print("running on cpu")  


def plot_image(image, path2save):
    plt.imshow(image)
    plt.colorbar()
    plt.savefig(path2save)
    plt.close()

''' 
parameters
'''

config_file = sys.argv[-1]
if os.path.exists(config_file):
    print("opening config file")
    print(config_file)
    with open(config_file) as json_file:
        params = json.load(json_file)
        data_params, forward_params, training_params, other_params = params["data"], params["forward_model"], params["training"], params["other"]
else:
    raise Exception("Config file not found: "+ config_file)

# data
SHAPE = data_params["image_shape"]
DATASET_NAME = data_params["dataset_name"]

# Forward model parameters
SNAPSHOTS =  forward_params["number_snapshots"]
TIPO_MUESTREO =  forward_params["tipo_muestreo"]
WAVE_LENGTH =  forward_params["wavelength"]
DX =  forward_params["pixel_size"]
DISTANCE_SENSOR =  forward_params["distance_sensor"]
NORMALIZE_MEASUREMENTS = forward_params["normalize_measurements"]

# Training 
BATCH_SIZE = training_params["batch_size"]
RESULTS_FOLDER =  training_params["results_folder"]

if os.path.exists(RESULTS_FOLDER):
    for files in os.listdir(RESULTS_FOLDER):
        os.remove(os.path.join(RESULTS_FOLDER, files))
else:
    os.makedirs(RESULTS_FOLDER)


    

# other
FLOAT_DTYPE =  "float32"
COMPLEX_DTYPE =  "complex64"
P =  other_params["p"]
K_SIZE = other_params["k_size"]
N_ITER = other_params["n_iterations"]


''' 
Data preprocessing
'''

_,  dataset = get_dataset(DATASET_NAME, shape = SHAPE, batch_size = BATCH_SIZE, num_classes = 10, complex_dtype=COMPLEX_DTYPE)
x_image, _ = next(iter(dataset))



''' 
Acquisition Layer definition
'''
muestreo_layer = Muestreo(snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
                dx = DX,distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
                normalize_measurements = NORMALIZE_MEASUREMENTS, 
                float_dtype = FLOAT_DTYPE, complex_dtype=COMPLEX_DTYPE, name="Muestreo")


muestras_entrada, mascara_usada, var_de_interes = muestreo_layer(x_image)

'''
image and matrices
'''
x_complex = tf.complex(x_image[0,...,0], x_image[0,...,1])

plot_image(x_image[0,...,0], os.path.join(RESULTS_FOLDER, "imagen_real.png"))
plot_image(x_image[0,...,1], os.path.join(RESULTS_FOLDER, "imagen_imag.png"))
plot_image(tf.math.abs(x_complex), os.path.join(RESULTS_FOLDER, "imagen_abs.png"))
plot_image(tf.math.angle(x_complex), os.path.join(RESULTS_FOLDER, "imagen_angle.png"))

if (TIPO_MUESTREO=="ASM" or TIPO_MUESTREO=="FRESNEL"):
    plot_image(tf.math.abs(var_de_interes[0,0,...]), os.path.join(RESULTS_FOLDER, "T_muestreo_.png"))
    plot_image(tf.math.angle(var_de_interes[0,0,...]), os.path.join(RESULTS_FOLDER, "T_muestreo_.png"))
plot_image(tf.math.abs(mascara_usada[0,0,]), os.path.join(RESULTS_FOLDER, "mascara_abs.png"))
plot_image(tf.math.angle(mascara_usada[0,0,]), os.path.join(RESULTS_FOLDER, "mascara_angle.png"))
#print("mascara rango", tf.reduce_min(tf.math.angle(mascara_usada[0,0,])), tf.reduce_max(tf.math.angle(mascara_usada[0,0,])))
plot_image(muestras_entrada[0,...,0], os.path.join(RESULTS_FOLDER, "muestras_imagen.png"))


'''
run back proopagation
'''

backpropagation_layer  = BackPropagationLayer(tipo_muestreo = TIPO_MUESTREO, complex_dtype = COMPLEX_DTYPE, name = "BackPropagationLayer")
back = backpropagation_layer([muestras_entrada,mascara_usada,var_de_interes])
back_real, back_imag = tf.unstack(back, num=2, axis=-1)
back_complex = tf.complex(back_real[0,], back_imag[0,])

plot_image(back_real[0,], os.path.join(RESULTS_FOLDER, "back_real.png"))
plot_image(back_imag[0,], os.path.join(RESULTS_FOLDER, "back_imag.png"))
plot_image(tf.math.abs(back_complex), os.path.join(RESULTS_FOLDER, "back_abs.png"))
plot_image(tf.math.angle(back_complex), os.path.join(RESULTS_FOLDER, "back_angle.png"))


'''
run fsi initialization
'''
init_initialzation = FSI_Initial(p = P, float_dtype=FLOAT_DTYPE,complex_dtype=COMPLEX_DTYPE, name = "init_initialzation")
Initialation = FSI_cell(p = P, k_size = K_SIZE, tipo_muestreo = TIPO_MUESTREO, float_dtype=FLOAT_DTYPE, complex_dtype=COMPLEX_DTYPE, train_initialization = True, name = "initialization_cell")
Ytr, Z = init_initialzation(muestras_entrada)

for i in range(N_ITER):
    Z = Initialation([Ytr, Z, mascara_usada, var_de_interes])

normalization = tf.reduce_max(tf.math.abs(Z), axis=(1,2,3), keepdims=True)

Z = tf.divide(Z, tf.cast(normalization, tf.complex64))
Z_abs = tf.squeeze(tf.math.abs(Z))
Z_angle = tf.squeeze(tf.math.angle(Z))

kernel_real, kernel_imag = Initialation.get_kernel_initialization()

plot_image(tf.math.real(Z)[0,0], os.path.join(RESULTS_FOLDER, "init_real.png"))
plot_image(tf.math.imag(Z)[0,0], os.path.join(RESULTS_FOLDER, "init_imag.png"))
plot_image(tf.math.abs(Z)[0,0], os.path.join(RESULTS_FOLDER, "init_abs.png"))
plot_image(tf.math.angle(Z)[0,0], os.path.join(RESULTS_FOLDER, "init_angle.png"))
plot_image(kernel_real, os.path.join(RESULTS_FOLDER, "kernel_real.png"))
plot_image(kernel_imag, os.path.join(RESULTS_FOLDER, "kernel_imag.png"))


print("The code ran successfully")

