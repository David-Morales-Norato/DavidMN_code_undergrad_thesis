#import tensorflow as tf
import numpy as np
from .Muestreos import *


class Muestreo(tf.keras.layers.Layer):
    def __init__(self, snapshots, wave_length ,dx ,distance, tipo_muestreo, normalize_measurements, float_dtype, complex_dtype, codificate = True, **kwargs):
        super(Muestreo, self).__init__(**kwargs)
        self.tipo_muestreo = tipo_muestreo
        self.L = snapshots
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        self.normalize_measurements = normalize_measurements
        self.wave_length = wave_length
        self.codificate = codificate
        self.dx = dx
        self.distance = distance

    def build(self, input_shape):
        super(Muestreo, self).build(input_shape)
        self.S = input_shape
        Masks_ang = tf.random.uniform((1,self.L, *self.S[1:-1]))*2*np.pi-np.pi
        Masks_ang = tf.cast(Masks_ang, dtype=self.float_dtype)
        if self.codificate == True:
          self.Masks_weights = self.add_weight(name='Masks', shape=[1,self.L, *self.S[1:-1]], initializer=tf.keras.initializers.Constant(Masks_ang), trainable=False)
        else:
          self.Masks_weights = self.add_weight(name='Masks', shape=[1,self.L, *self.S[1:-1]], initializer=tf.keras.initializers.Constant(tf.ones(shape = Masks_ang.shape, dtype = Masks_ang.dtype)), trainable=False)
        
        if self.tipo_muestreo =="FRAN":
          self.A = lambda y, mask: tf.math.abs(A_Fran(y, mask))
          self.var_de_interes = tf.ones(tf.shape(self.Masks_weights))

        elif self.tipo_muestreo == "ASM":
          X = np.arange(-self.S[1]//2, self.S[1]//2)
          Y = np.arange(-self.S[2]//2, self.S[2]//2)
          X, Y = np.meshgrid(X,Y)
          U = 1 - (self.wave_length**2)*((X/(self.dx*self.S[1]))**2 + (Y/(self.dx*self.S[2]))**2);
          SFTF = np.exp(1j*2*np.pi/self.wave_length*self.distance*np.sqrt(U))
          SFTF[U<0]=0
          self.SFTF = tf.broadcast_to(tf.convert_to_tensor(SFTF, dtype=self.complex_dtype), (1, 1, *self.Masks_weights.shape[-2:]))
          self.A = lambda y, mask: tf.math.abs(A_ASM_LAB(y, mask, self.SFTF))
          self.var_de_interes = self.SFTF

        elif self.tipo_muestreo == "FRESNEL":
          X = np.arange(-self.S[1]//2, self.S[1]//2)
          Y = np.arange(-self.S[2]//2, self.S[2]//2)
          X, Y = np.meshgrid(X,Y)
          THOR = (X**2 + Y**2)*self.dx
          Q1 = np.exp(-1j*(np.pi/self.wave_length/self.distance)*THOR**2)
          self.Q = tf.broadcast_to(tf.convert_to_tensor(Q1, dtype=self.complex_dtype), (1, 1, *self.Masks_weights.shape[-2:]))
          self.var_de_interes = self.Q
          self.A = lambda y, mask: tf.math.abs(A_FRESNEL(y, mask, self.var_de_interes))
        else:
          raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

    def call(self, input):
        input = tf.cast(input, dtype=self.float_dtype)
        real, imag = tf.unstack(input, num=2, axis=-1)
        self.Masks = tf.multiply(tf.ones(tf.shape(self.Masks_weights), dtype=self.complex_dtype), tf.math.exp(tf.multiply(tf.cast(self.Masks_weights, dtype = self.complex_dtype) , tf.constant(1j, dtype=self.complex_dtype))))
        Z = tf.complex(real, imag)
        Z = tf.expand_dims(Z,1)
        Y = self.A(Z, self.Masks)
        Y = tf.transpose(Y,perm=[0,2,3,1])  
        if(self.normalize_measurements):
          division = tf.math.reduce_max(Y, axis=[1,2], keepdims=True)
          Y = tf.math.divide(Y, division)
        return Y, self.Masks, self.var_de_interes
    

    def get_mask(self):
      return self.Masks_weights

    def get_var_de_interes(self):
      return self.var_de_interes


