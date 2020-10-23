# VGG19 Style Transfer in Python
# By Isaac Krementsov, based on tutorial at https://www.tensorflow.org/tutorials/generative/style_transfer


import numpy as np
import cv2
import os

import misc_utils as misc
import image_utils as images
import ml_utils as ml


path = 'C:\\Users\\isaac\\Documents\\python\\art_generation\\ann\\assets\\base_video.mp4'
video = cv2.VideoCapture(path)

count = 0
success = True

while success:
    success, image = video.read()
    cv2.imwrite('assets/frames/frame%d.jpg' % count, image)
    count += 1

imgs = [img for img in os.listdir('assets/frames')]
imgs.sort(key=misc.natural_keys)

for i in range(len(imgs)):
    if i > 155:
        image = imgs[i]

        # Image to base content on
        base_image = images.to_array('./assets/frames/' + image)

        style = 'style_reference_image3.jpg'
        if i < 100:
            style = 'style_reference_image2.jpg'

        # Image to base artistic style on
        style_reference_image = images.to_array('./assets/' + style)

        previous_frame = images.to_array('./assets/style_reference_image.jpg')
        weight_frame = 0
        if i > 0:
            # Previous frame for animations
            previous_frame = images.to_array('./assets/frames/frame' + str(i - 1) + '.jpg')
            weight_frame = 3e-1

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
            weight_frame=weight_frame, # How much to blend previous frame's style
            epochs=10, # How long to let the network train
            learning_rate=0.02, # How fast the network should descend gradients
            shift=1
        )

        mixed_image.save('./assets/frames/frame' + str(i) + '.jpg')
