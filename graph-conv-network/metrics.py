import tensorflow as tf

def maskedSoftmaxCrossEntropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(preds, labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    correctPrediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracyAll = tf.cast(correctPrediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracyAll *= mask
    return tf.reduce_mean(accuracyAll)