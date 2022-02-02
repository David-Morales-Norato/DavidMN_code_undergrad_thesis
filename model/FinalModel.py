import tensorflow as tf
from .AcquisitionLayer import Muestreo
from .InitializationLayer import FSI_Initial,FSI_cell, BackPropagationLayer


class customGaussian(tf.keras.layers.Layer):
    def __init__(self, snr=10, name="Gaussian_noise_layer"):
        super(customGaussian, self).__init__(name=name)
        self.snr = snr
        
    def add_noise_each_x(self, x):

        m = x.shape[0]*x.shape[1]*x.shape[2]
        divisor = m*10**(self.snr/10)
        stddev = tf.math.sqrt(tf.math.divide(tf.math.pow(tf.norm(x, 'fro', axis= [0,1]),2), divisor))
        return x + tf.keras.backend.random_normal(shape=x.shape,mean=0,stddev=stddev, dtype=x.dtype)

    def call(self, input):
        input = tf.cast(input, tf.float64)      
        salida = tf.map_fn(self.add_noise_each_x,input)
        return salida


class CLAS_PR(tf.keras.Model):
    def __init__(self, shape, num_classes, snapshots, clasification_network, wave_length, dx, distance, tipo_muestreo, normalize_measurements, float_dtype, complex_dtype, snr):
        super(CLAS_PR, self).__init__()
        self.muestreo_layer = Muestreo(snapshots = snapshots, wave_length = wave_length, 
                dx = dx,distance = distance, tipo_muestreo = tipo_muestreo, 
                normalize_measurements = normalize_measurements, 
                float_dtype = float_dtype, complex_dtype=complex_dtype, codificate = False, name="Muestreo")

        if clasification_network == "mobilnetv2":
            self.classification_network = tf.keras.applications.MobileNetV2(input_shape=(*shape,3), classes=num_classes, weights=None, classifier_activation="softmax")
            self.conv_initial = tf.keras.layers.Conv2D(3, 3, padding="same", activation = None, name = "ConvInitial")
        #self.ruido = customGaussian(snr = snr)

    def build(self, input_shape):
        super(CLAS_PR, self).build(input_shape)
        self.S = input_shape
         
    def call(self, input):
        muestras,_,_ = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)
        muestras = self.conv_initial(muestras)
        clasificion = self.classification_network(muestras)
        return clasificion

    def get_medidas(self, input):
        muestras,_,_ = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)
        return muestras

    def model(self):
        x = tf.keras.Input(shape = self.S[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class CLAS_PR_BACK(tf.keras.Model):
    def __init__(self, shape, num_classes, snapshots, clasification_network, wave_length, dx, distance, tipo_muestreo, normalize_measurements, float_dtype, complex_dtype, snr):
        super(CLAS_PR_BACK, self).__init__()
        self.muestreo_layer = Muestreo(snapshots = snapshots, wave_length = wave_length, 
                dx = dx,distance = distance, tipo_muestreo = tipo_muestreo, 
                normalize_measurements = normalize_measurements, 
                float_dtype = float_dtype, complex_dtype=complex_dtype, name="Muestreo")

        if clasification_network == "mobilnetv2":
            self.classification_network = tf.keras.applications.MobileNetV2(input_shape=(*shape,3), classes=num_classes, weights=None, classifier_activation="softmax")
            self.conv_initial = tf.keras.layers.Conv2D(3, 3, padding="same", activation = None, name = "ConvInitial")
        #self.ruido = customGaussian(snr = snr)
        self.backpropagation_layer  = BackPropagationLayer(tipo_muestreo = tipo_muestreo, complex_dtype = complex_dtype, name = "BackPropagationLayer")

    def build(self, input_shape):
        super(CLAS_PR_BACK, self).build(input_shape)
        self.S = input_shape

    def call(self, input):
        muestras,mascara_usada,var_de_interes  = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)
        caracteristicas = self.backpropagation_layer([muestras,mascara_usada,var_de_interes])
        caracteristicas = self.conv_initial(caracteristicas)
        clasificion = self.classification_network(caracteristicas)
        return clasificion

    def model(self):
        x = tf.keras.Input(shape = self.S[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_medidas(self, input):
        muestras,_,_  = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)
        return muestras
    
    def get_init(self, input):
        muestras,mascara_usada,var_de_interes  = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)
        caracteristicas = self.backpropagation_layer([muestras,mascara_usada,var_de_interes])
        x_real, x_angle = tf.unstack(caracteristicas, num=2, axis = -1)
        return x_real, x_angle

class CLAS_PR_INIT(tf.keras.Model):
    def __init__(self, shape, num_classes, p, k_size, n_iterations, snapshots, clasification_network, wave_length, dx, distance, tipo_muestreo, normalize_measurements, float_dtype, complex_dtype, snr):
        super(CLAS_PR_INIT, self).__init__()
        self.p = p
        self.k_size = k_size
        self.n_iterations = n_iterations
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        self.muestreo_layer = Muestreo(snapshots = snapshots, wave_length = wave_length, 
                dx = dx,distance = distance, tipo_muestreo = tipo_muestreo, 
                normalize_measurements = normalize_measurements, 
                float_dtype = float_dtype, complex_dtype=complex_dtype, name="Muestreo")

        self.init_initialzation = FSI_Initial(p = p, float_dtype=float_dtype,complex_dtype=complex_dtype, name = "init_initialzation")
        self.Initialation = FSI_cell(p = p, k_size = k_size, tipo_muestreo = tipo_muestreo,float_dtype=float_dtype,complex_dtype=complex_dtype,  train_initialization = False, name = "initialization_cell")
        if clasification_network == "mobilnetv2":
            self.classification_network = tf.keras.applications.MobileNetV2(input_shape=(*shape,3), classes=num_classes, weights=None, classifier_activation="softmax")
            self.conv_initial = tf.keras.layers.Conv2D(3, 3, padding="same", activation = None, name = "ConvInitial")
        #self.ruido = customGaussian(snr = snr)

    def build(self, input_shape):
        super(CLAS_PR_INIT, self).build(input_shape)
        self.S = input_shape
        

    def call(self, input):
        muestras,mascara_usada,var_de_interes  = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)


        Ytr, Z = self.init_initialzation(muestras)

        for _ in range(self.n_iterations):
          Z = self.Initialation([Ytr, Z, mascara_usada, var_de_interes])

        Z = tf.transpose(Z,perm=[0,2,3,1])
        back_real = tf.math.real(Z)
        back_imag = tf.math.imag(Z)

        initialization_x = tf.concat([back_real, back_imag], axis=-1)

        caracteristicas = self.conv_initial(initialization_x)
        clasificion = self.classification_network(caracteristicas)
        return clasificion

    def model(self):
        x = tf.keras.Input(shape = self.S[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_medidas(self, input):
        muestras,_,_  = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)
        return muestras
    
    def get_init(self, input):
        muestras,mascara_usada,var_de_interes  = self.muestreo_layer(input)
        #muestras = self.ruido(muestras)


        Ytr, Z = self.init_initialzation(muestras)

        for i in range(self.n_iterations):
          Z = self.Initialation([Ytr, Z, mascara_usada, var_de_interes])

        Z = tf.transpose(Z,perm=[0,2,3,1])
        back_real = tf.math.real(Z)
        back_imag = tf.math.imag(Z)
        return back_real,back_imag