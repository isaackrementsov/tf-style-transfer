# VGG19 Style Transfer in Python
# By Isaac Krementsov, based on tutorial at https://www.tensorflow.org/tutorials/generative/style_transfer


import numpy as np

import image_utils as images
import ml_utils as ml

# Previous frame (for animations)
previous_frame = images.to_array('./assets/mixed_image.jpg')
# Image to base content on
base_image = images.to_array('./assets/previous_frame.jpg')
# Image to base artistic style on
style_reference_image = images.to_array('./assets/style_reference_image3.jpg')

# Network-generated image that shows content of base image in artistic style
mixed_image = ml.style_transfer(
    previous_frame=previous_frame,
    content_image=base_image,
    style_image=style_reference_image,
    content_layers=['block4_conv2'], # VGG19 layers used to determine content
    style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'], # VGG19 layers used to determine style
    weight_content=1e3, # How much the image content should matter
    weight_style=8e-1, # How much the artistic style should matter
    weight_denoise=1e-4, # How much noise needs to be removed
    weight_frame=7e-1, # How much to blend previous frame's style
    epochs=10, # How long to let the network train
    learning_rate=0.02, # How fast the network should descend gradients
    shift=1
)

# Save the image data
mixed_image.save('./assets/mixed_image4.jpg')
