import numpy as np
import tensorflow as tf
import param

l_function = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# shape(real) = (batch_size, pad_size)
# shape(pred) = (batch_size, pad_size, tar_vocab_size)
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = l_function(real, pred) # shape(loss_) = (BATCH_SIZE, PAD_SIZE)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    # Return mean without the 0 elements
    return tf.math.reduce_sum(loss_) / tf.math.reduce_sum(mask)

# mask pred according to eos_id index
def masked_equal(real, pred, apply_argmax = False):
    if apply_argmax:
        pred = tf.argmax(pred, axis=-1)

    try:
        eos_index = list(pred.numpy()).index(param.EOS_ID)
        ones = np.ones(eos_index)
        zeros = np.zeros(len(pred) - eos_index)
        mask = np.append(ones, zeros)
        mask = tf.constant(mask)
        pred *= mask
    except:
        # eos_id not found
        pass
    return np.array_equal(real.numpy(), pred.numpy())

def cer(real, pred):
    # shape(real) = shape(pred) = (pad_sz)
    real = tf.cast(real, dtype='int32')
    pred = tf.cast(pred, dtype='int32')
    error = real != pred
    error = tf.cast(error, dtype='int32')
    return tf.reduce_sum(error) / len(real)

class PerformanceMetrics:
    def __init__(self):
        self.total_word = 0.0
        self.error_word_count = 0.0
        self.loss = tf.metrics.Mean(name='performance_loss')
        self.cer  = tf.metrics.Mean(name='character_error_rate')    

    def __call__(self, real_ref, pred):
        # shape(pred) = (pad_size)
        # shape(real_ref) = (None, pad_size)
        self.total_word += 1.0
        cer_list = [1.0]
        loss_list = [999.0]
        error = 0.0

        for real in real_ref:
            loss_list.append(loss_function(real, pred))
            tmp_pred = tf.argmax(pred, axis=-1) # do not change pred, shape error in next iteration
            cer_list.append(cer(real, tmp_pred))
            if not masked_equal(real, tmp_pred):
                error = 1.0
        self.error_word_count += error
        self.loss(min(loss_list))
        self.cer(min(cer_list))

    def reset_states(self):
        self.error_word_count = 0.0
        self.total_word = 0.0
        self.cer.reset_states()
        self.loss.reset_states()

    def result(self):
        return self.loss.result(), self.cer.result(), self.error_word_count / self.total_word