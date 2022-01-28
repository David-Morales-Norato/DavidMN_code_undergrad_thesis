import tensorflow as tf

@tf.function
def A_Fran(y, Masks):
    return  tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(tf.multiply(Masks, y))))

@tf.function
def AT_Fran(y, Masks):
    y = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.fftshift(y)))
    mult_mass_z = tf.multiply(tf.math.conj(Masks),y)
    res = tf.reduce_sum(mult_mass_z, axis=1)
    return res

@tf.function
def A_ASM_LAB(y, Masks, SFTF):
    y = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.multiply(y, Masks))))
    y = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(tf.multiply(y, SFTF))))
    return  y

@tf.function
def AT_ASM_LAB(y, Masks, SFTF):
    y = tf.multiply(tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(y))), tf.math.conj(SFTF))
    y = tf.multiply(tf.signal.fftshift(tf.signal.ifft2d(tf.signal.fftshift(y))), tf.math.conj(Masks))
    res = tf.reduce_mean(y, axis=1)
    return res

@tf.function
def A_FRESNEL(x, Masks,Q):
    y = tf.multiply(Masks, tf.multiply(x, Q))
    res = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(y)))
    return  res
    
@tf.function
def AT_FRESNEL(x, Masks,Q):
    y = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.fftshift(x))) 
    res = tf.multiply(tf.math.conj(Masks), tf.multiply(y,tf.math.conj( Q)))
    res = tf.reduce_mean(res, axis=1)
    return  res

