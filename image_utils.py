# Helpful image processing tools

from PIL import Image
import numpy as np
import tensorflow as tf


# Convert image file into a tensorflow-readable input with defined shape
def to_array(path):
  max_dim = 512

  img = tf.io.read_file(path)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]

  return img


# Go from 0-255 to 0-1 scale
def normalize(img):
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


# Go from tensor output back to image
def to_image(tensor):
    # Scale from 0-1 to 0-255
    tensor = 255*tensor
    # Make numpy array of bytes
    tensor = np.array(tensor, dtype=np.uint8)

    # If the image is wrapped in an extra dimension, remove it
    if np.ndim(tensor) > 3:
        # Make sure this isn't a list of multiple images
        assert tensor.shape[0] == 1
        # Get rid of the extra dimension
        tensor = tensor[0]

    # Initialize PIL Image using modified tensor
    return Image.fromarray(tensor)


def save(img, name):
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)

    with open(name, 'wb') as file:
        Image.fromarray(img).save(file, 'jpeg')
