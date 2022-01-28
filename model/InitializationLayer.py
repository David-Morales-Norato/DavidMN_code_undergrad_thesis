import tensorflow as tf
from .Muestreos import *
import numpy as np
import cv2



class ComplexConv2D(tf.keras.layers.Layer):
    def __init__(self, k_size, kernel_real_init, kernel_imag_init, train_initialization, **kwargs):
        super(ComplexConv2D, self).__init__(**kwargs)
        kernel_real_init = tf.expand_dims(kernel_real_init, -1)
        kernel_imag_init = tf.expand_dims(kernel_imag_init, -1)
        self.kernel_real = self.add_weight(shape=(k_size, k_size, 1, 1),
                          initializer=tf.keras.initializers.Constant(kernel_real_init),
                          trainable=train_initialization, dtype = kernel_real_init.dtype, name = "kernel_real")
        self.kernel_imag = self.add_weight(shape=(k_size, k_size, 1, 1),
                          initializer=tf.keras.initializers.Constant(kernel_imag_init),
                          trainable=train_initialization, dtype = kernel_imag_init.dtype, name = "kernel_imag")
        self.conv_real = lambda input: tf.nn.conv2d(tf.cast(input, kernel_real_init.dtype), self.kernel_real, strides=1, padding="SAME", data_format='NHWC')
        self.conv_imag = lambda input: tf.nn.conv2d(tf.cast(input, kernel_imag_init.dtype), self.kernel_imag, strides=1, padding="SAME", data_format='NHWC')

    def call(self, input):
        z_real = tf.math.real(input)
        z_imag = tf.math.imag(input)
        
        Z_real_filt = self.conv_real(z_real) - self.conv_imag(z_imag)
        Z_imag_filt = self.conv_real(z_imag) + self.conv_imag(z_real)
        
        return tf.complex(Z_real_filt, Z_imag_filt)

    def get_kernel_initialization(self):
      return [self.kernel_real, self.kernel_imag]

class FSI_Initial(tf.keras.layers.Layer):
    def __init__(self, p, float_dtype, complex_dtype, **kwargs):
        super(FSI_Initial, self).__init__(**kwargs)
        self.p = p
        self.complex_dtype = complex_dtype
        self.float_dtype = float_dtype
        

    def build(self, input_shape):
        super(FSI_Initial, self).build(input_shape)
        self.S = input_shape
        self.M = tf.constant(self.S[1] * self.S[2] * self.S[3], dtype=self.float_dtype)
        self.R = tf.cast(tf.math.ceil(tf.math.divide(self.M, self.p)), dtype=self.float_dtype)

        Z0_abs = tf.random.normal(shape=(1,1, self.S[1], self.S[2]), mean=0.5, stddev=0.1)
        Z0_angle= tf.random.normal(shape=(1, 1,self.S[1], self.S[2]), mean=0.0, stddev=(0.5*np.pi**(1/2)))
        Z0_abs = tf.cast(Z0_abs,self.complex_dtype)
        Z0_angle = tf.cast(Z0_angle,self.complex_dtype)
        Z0 = tf.multiply(Z0_abs, tf.math.exp(tf.multiply(Z0_angle, tf.constant(1j,self.complex_dtype))))
        self.Z0 = tf.math.divide(Z0, tf.norm(Z0, ord='fro',axis=(2,3), keepdims=True))

        

    def call(self, input):

        Y = input
        
        # INitializations
        Y = tf.cast(Y, dtype=self.float_dtype); Y = tf.transpose(Y,perm=[0,3,1,2])# (self.S[0], self.L, self.S[1], self.S[2])
        Z = tf.cast(self.Z0,self.complex_dtype)

        # GET YTR
        S = tf.shape(Y)
        Y_S = tf.math.divide(Y,self.S[2])
        y_s = tf.reshape(Y_S, (S[0], S[1]*S[2]*S[3])) # Vectoriza

        y_s = tf.sort(y_s, axis = 1, direction='DESCENDING')
        aux  = tf.gather(y_s, indices=tf.cast(self.R-1, tf.int64), axis=1)
        threshold = tf.reshape(aux, (S[0], 1, 1, 1))
        Ytr = tf.cast(Y_S>=threshold, dtype=Y_S.dtype) 
        return Ytr, self.Z0

class FSI_cell(tf.keras.layers.Layer):
    def __init__(self, p, k_size, tipo_muestreo, float_dtype,complex_dtype, train_initialization, **kwargs):
        super(FSI_cell, self).__init__(**kwargs)
        self.p = p
        self.k_size = k_size
        self.tipo_muestreo = tipo_muestreo
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype

        k = cv2.getGaussianKernel(k_size,1)
        self.kernel = tf.constant(np.dot(k, k.T), shape=(k_size, k_size), dtype=self.float_dtype)
        self.kernel_real = self.kernel/(tf.reduce_max(self.kernel) - tf.reduce_min(self.kernel))        
        self.kernel_imag = self.kernel_real

        # self.conv_real = tf.keras.layers.Conv2D(1, self.k_size, padding="same", use_bias=False, activation = None, kernel_initializer=tf.keras.initializers.Constant(self.kernel_real), trainable=train_initialization,name="FILTRO_REAL_INITIALIZATION")
        # self.conv_imag = tf.keras.layers.Conv2D(1, self.k_size, padding="same", use_bias=False, activation = None, kernel_initializer=tf.keras.initializers.Constant(self.kernel_imag), trainable=train_initialization,name="FILTRO_IMAG_INITIALIZATION")
        self.complex_conv = ComplexConv2D(k_size, kernel_real_init = self.kernel_real, kernel_imag_init = self.kernel_imag, train_initialization = train_initialization)

    def build(self, input_shape):
        super(FSI_cell, self).build(input_shape[0])
        self.S = input_shape[0]
        if self.tipo_muestreo =="FRAN":
          self.A = lambda y, mask,ignore: A_Fran(y, mask)
          self.AT = lambda y, mask,ignore: AT_Fran(tf.cast(y, self.complex_dtype), mask)
        elif self.tipo_muestreo == "ASM":
          self.A = lambda y, mask,sftf: A_ASM_LAB(y, mask, sftf)
          self.AT = lambda y,mask,sftf: AT_ASM_LAB(tf.cast(y, self.complex_dtype), mask, sftf)
        elif self.tipo_muestreo == "FRESNEL":
          self.A = lambda y, mask,Q: A_FRESNEL(y, mask, Q)
          self.AT = lambda y, mask,Q: AT_FRESNEL(tf.cast(y, self.complex_dtype), mask, Q)
        else:
          raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

    def call(self, input):


        Ytr = tf.cast(input[0], self.float_dtype)
        Z = input[1]
        self.Masks = input[2]
        self.var_de_interes = input[3]

        Z = self.AT(tf.multiply(tf.cast(Ytr, Z.dtype), self.A(Z, self.Masks,self.var_de_interes)), self.Masks,self.var_de_interes)
        Z = tf.math.divide(Z,self.S[3]**2*self.S[1]**2*self.S[2]**2*self.p)
        Z = tf.expand_dims(Z,-1)

        Z = self.complex_conv(Z)

        Z = tf.transpose(Z, [0, 3, 1, 2])
        Z = tf.math.divide(Z, tf.norm(Z, ord='fro', axis=(2,3), keepdims=True))
        return Z
  
    def get_kernel_initialization(self):
      return self.complex_conv.get_kernel_initialization()#[self.conv_real.get_weights()[0], self.conv_imag.get_weights()[0]]
    

class BackPropagationLayer(tf.keras.layers.Layer):
  def __init__(self, tipo_muestreo, complex_dtype, **kwargs):
      super(BackPropagationLayer, self).__init__(**kwargs)  
      self.tipo_muestreo = tipo_muestreo  
      self.complex_dtype = complex_dtype

  def build(self, input_shape):
      super(BackPropagationLayer, self).build(input_shape)
      if self.tipo_muestreo =="FRAN":
        self.AT = lambda y, mask, ignore: AT_Fran(tf.cast(y, self.complex_dtype), mask)
      elif self.tipo_muestreo == "ASM":
        self.AT = lambda y,mask,sftf: AT_ASM_LAB(tf.cast(y, self.complex_dtype), mask, sftf)
      elif self.tipo_muestreo == "FRESNEL":
        self.AT = lambda y,mask,Q : AT_FRESNEL(tf.cast(y, self.complex_dtype), mask, Q)

      else:
        raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

  def call(self, input):

      Y = tf.cast(input[0], self.complex_dtype)
      self.Masks = input[1]
      self.var_de_interes = input[2]

      Y = tf.transpose(Y,perm=[0,3,1,2])
      Z0 = self.AT(Y,self.Masks,self.var_de_interes)
      
      Z0 = tf.expand_dims(Z0, -1)
      back_real = tf.math.real(Z0); 
      back_imag = tf.math.imag(Z0); 
      Z = tf.concat([back_real, back_imag], axis = -1)
      return Z
