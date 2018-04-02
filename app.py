
import numpy as np
import tensorflow as tf

from get_data import *
from model import *
from model_2 import *
from train_and_eval import *

tf.logging.set_verbosity(tf.logging.INFO)

# TODO: saving and loading, but maybe not needed
def load_or_get_new_classifier():
    return get_new_cifar_classifier()
#
# def save_classifier(classifier):
#     print("saving not implemented yet")

def main(argv):
    # batch1 = get_batch(1)
    # train_data = batch1[b'data']
    # train_labels = batch1[b'labels']
    test = get_test_data()
    test_data = test[b'data'][0:1000]
    test_labels = test[b'labels'][0:1000]

    classifier = load_or_get_new_classifier()

    for i in range(5):
        batch = get_batch(i + 1)
        train_data = batch[b'data']
        train_labels = batch[b'labels']

        print("Train")
        train_classifier(classifier, train_data, train_labels, 100)

        print("Eval")
        eval_results = eval_classifier(classifier, test_data, test_labels)
        print(eval_results)

    # save_classifier(classifier)
    print("Goodbye")

if __name__ == "__main__":
  tf.app.run()