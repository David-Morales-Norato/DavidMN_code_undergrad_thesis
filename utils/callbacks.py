
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, io

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_predictions(x_image, y_pred, medidas, init, y_true, num_plots = 5):

  if init is None:
    n_figs = 4
  else :
    n_figs = 6
  x_abs, x_angle = tf.unstack(x_image, num=2, axis=-1)

  conf_matrix = medidas
  fig = plt.figure(figsize=(30, 20))
  for indx_plot in range(num_plots):
    plt.subplot(num_plots, n_figs, n_figs*indx_plot + 1); plt.imshow(tf.squeeze(x_abs[indx_plot, ...])); plt.xticks([]); plt.yticks([]); plt.title("x_abs")
    plt.subplot(num_plots, n_figs, n_figs*indx_plot + 2); plt.imshow(tf.squeeze(x_angle[indx_plot, ...])); plt.xticks([]); plt.yticks([]); plt.title("x_angle")
    plt.subplot(num_plots, n_figs, n_figs*indx_plot + 3); plt.imshow(tf.squeeze(medidas[indx_plot, ...])); plt.xticks([]); plt.yticks([]); plt.title("medidas, etiqueta: " + str(y_true.numpy()[indx_plot, ...]) + ", predicción " + str(y_pred.numpy()[indx_plot, ...]))
    plt.subplot(num_plots, n_figs, n_figs*indx_plot + 4); plt.imshow(tf.squeeze(conf_matrix[indx_plot, ...])); plt.xticks([]); plt.yticks([]); plt.title("Matrix de confusión")
    if init is not None:
      init_abs, init_angle = init
      plt.subplot(num_plots, n_figs, n_figs*indx_plot + 5); plt.imshow(tf.squeeze(init_abs[indx_plot, ...])); plt.xticks([]); plt.yticks([]); plt.title("init_abs")
      plt.subplot(num_plots, n_figs, n_figs*indx_plot + 6); plt.imshow(tf.squeeze(init_angle[indx_plot, ...])); plt.xticks([]); plt.yticks([]); plt.title("init_angle")

  return fig
  
def log_predictions(epoch, logs, model, val_images, fw_results, num_plots = 5, plot_init = True):
  # Use the model to predict the values from the validation dataset.
  x_image, y_true = val_images
  y_pred = model(x_image)
  medidas = model.get_medidas(x_image)
  #x_image = x_image[:num_plots, ...]
  y_true = tf.argmax(y_true, axis = -1)#[:num_plots, ...]
  y_pred = tf.argmax(y_pred, axis = -1)#[:num_plots, ...]
  medidas = medidas#[:num_plots, ...]
  
  if plot_init:
    init_real, init_imag = model.get_init(x_image)
    init_real, init_imag = init_real, init_imag#[:num_plots, ...], init_imag#[:num_plots, ...]
    init = init_real, init_imag
  else:
    init = None

  
  # Log the results images as an image summary.
  figure_predictions = plot_predictions(x_image, y_pred, medidas, init, y_true, num_plots = num_plots)
  # Log the results images as an image summary.
  image = plot_to_image(figure_predictions)
  with fw_results.as_default():
    tf.summary.image("results images initializations", image, step=epoch)

