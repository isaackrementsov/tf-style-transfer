# Neural network code and helper functions are located here

import tensorflow as tf
import keras.applications.vgg19 as vgg19
import numpy as np

import image_utils as images
import math_utils as math


# Main function for neural network training
def style_transfer(previous_frame, content_image, style_image,
        content_layers, style_layers,
        shift=1,
        weight_content=1, weight_style=1, weight_denoise=1, weight_frame=1,
        learning_rate=1, epochs=10):

        # Feature extractor to get activations from VGG19 layers
        extractor = StyleContentModel(style_layers, content_layers)

        # Features corresponding to style and image layers
        style_targets = extractor(style_image)['style']
        frame_targets = extractor(previous_frame)['style']
        content_targets = extractor(content_image)['content']

        # Start the mixed image out with the original content image
        image = tf.Variable(content_image)

        # Using Adam optimizer for loss reduction
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

        steps_per_epoch = 100

        # Complete the desired number of gradient steps
        for n in range(epochs):
            for m in range(steps_per_epoch):
                # Complete training steps and assign new values to the image each time
                image.assign(train_step(
                    image,
                    extractor,
                    optimizer,
                    {'style': weight_style, 'content': weight_content, 'denoise': weight_denoise, 'frame': weight_frame},
                    {'style': style_targets, 'content': content_targets, 'frame': frame_targets},
                    {'style': len(style_layers), 'content': len(content_layers)},
                    shift
                ))

        print('Done training!')

        # Convert image tensor to a PIL.Image object
        return images.to_image(image)


# Get combined style and content loss for mixed image
def style_content_loss(outputs, weight_style, weight_content, weight_frame, num_style_layers, num_content_layers, style_targets, content_targets, frame_targets):
    # Get extracted style and content activations
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    # Get error for style activations of the mixed image
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
    # Multiply by how important style is
    style_loss *= weight_style / num_style_layers

    # Get error for content activations of the mixed image
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
    # Multiply by how important content is
    content_loss *= weight_content / num_content_layers

    # Get style loss from previous frame
    frame_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - frame_targets[name])**2) for name in style_outputs.keys()])
    # Multiply by how import frame style is
    frame_loss *= weight_frame / num_content_layers

    # Combine both losses
    loss = style_loss + content_loss + frame_loss

    return loss


# Loss from image being noisy (high-frequency "jumps")
def denoise_loss(image, shift):
    # Get how much the image varies due to pixels shifted horizontally and vertically
    x_var = image[:,:,shift:,:] - image[:,:,:-shift,:]
    y_var = image[:,shift:,:,:] - image[:,:-shift,:,:]

    return tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))


# Descend the gradient
@tf.function()
def train_step(image, extractor, optimizer, weights, targets, num_layers, shift):
    with tf.GradientTape() as tape:
        # Get current activations
        outputs = extractor(image)

        loss = style_content_loss(
            outputs,
            weights['style'],
            weights['content'],
            weights['frame'],
            num_layers['style'],
            num_layers['content'],
            targets['style'],
            targets['content'],
            targets['frame']
        )

        # Add denoise loss
        loss += weights['denoise']*denoise_loss(image, shift)

    # Use selected optimizer to descend gradient
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])

    # Return the network's new values for the image and clip between 0 and 1
    return images.normalize(image)


# Get VGG19 layers using layer names
def vgg_layers(layer_names):
        # Initialize VGG19 network, don't train the weights
        vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        # Use the layers to create a new Keras model
        model = tf.keras.Model([vgg.input], outputs)
        return model


# Style & Content feature extractor
class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        # Use newly created TensorFlow model as VGG
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

        self.vgg.trainable = False

    # Extract features/activations
    def call(self, inputs):
        # Scale up from 0-1 to 0-255
        inputs = inputs*255.0
        # Let VGG19 prepare the input data
        preprocessed_input = vgg19.preprocess_input(inputs)

        # Call vgg and get style/content activations
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        # Use Gramian operator to quantify style features
        style_outputs = [math.gramian(style_output) for style_output in style_outputs]

        # Store layer values in a dictionary
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
