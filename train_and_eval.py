import tensorflow as tf
import numpy as np

def train_classifier(classifier, data, labels, steps = 1000):
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=10)
    return classifier.train(
        input_fn=get_train_input_fn(data,labels),
        steps=steps,
        # hooks=[logging_hook]
    )

def eval_classifier(classifier, data, labels):
    return classifier.evaluate(input_fn=get_eval_input_fn(data, labels))


def get_train_input_fn(data, labels):
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(data)},
        y=np.array(labels),
        batch_size=10,
        num_epochs=None,
        shuffle=True)


def get_eval_input_fn(data, labels):
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(data)},
        y=np.array(labels),
        num_epochs=1,
        shuffle=False)
