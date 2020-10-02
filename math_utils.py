# Helpful math tools

import tensorflow as tf

# Gramian matrix operator
def gramian(tensor):
    # Sum the outer product of each feature vector with itself
    total = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    shape = tf.shape(tensor)
    size = tf.cast(shape[1]*shape[2], tf.float32)

    # Average the sum over the tensor dimensions
    return total/size
