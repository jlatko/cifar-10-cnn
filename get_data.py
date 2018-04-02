def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_test_data():
    return unpickle("./cifar-10-batches-py/test_batch")


# 1 to 5
def get_batch(number):
    if number < 1 or number > 5: raise ValueError("possible batch numbers 1 - 5")
    return unpickle("./cifar-10-batches-py/data_batch_{}".format(int(number)))
