
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

def plot_confussion_matrix(y_true, y_pred, n_classes, epoch):
  cf_matrix = None
  return cf_matrix
  
def log_confusion_depth(epoch, logs, model, val_images, fw_results, save_dir):
  # Use the model to predict the values from the validation dataset.
  x_image, y_true = val_images
  y_pred = model(x_image)

  fw_results_init,fw_results_recons = fw_results

  
  # Log the results images as an image summary.
  figure_conf_matrx = plot_confussion_matrix(y_true, y_pred, epoch = epoch, name = "initializaitons")
  # Log the results images as an image summary.
  image = plot_to_image(figure_conf_matrx)
  with fw_results_init.as_default():
    tf.summary.image("results images initializations", image, step=epoch)

