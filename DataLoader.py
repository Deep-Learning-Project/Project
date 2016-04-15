from os import listdir
from os.path import join

import numpy as np
from keras.utils import np_utils


def load_data(data='cifar-10', red_size=1):
    if data == 'cifar-10':
        path_to_data = './cifar-10-batches-py'
        train_files = [join(path_to_data, f) for f in listdir(path_to_data) if 'data' in f]
        train_data = None
        train_labels = None
        for f in train_files:
            if f == train_files[0]:
                ret_val = unpickle(f)
                train_data = ret_val[b'data']
                train_labels = ret_val[b'labels']
            else:
                ret_val = unpickle(f)
                train_data = np.vstack((train_data, ret_val[b'data']))
                train_labels = np.hstack((train_labels, ret_val[b'labels']))

        test_file_path = join(path_to_data, 'test_batch')
        test_dict = unpickle(test_file_path)
        X_train = np.asarray(train_data)
        X_train = X_train.reshape((X_train.shape[0], 3, 32, 32))
        X_test = np.asarray(test_dict[b'data'])
        X_test = X_test.reshape((X_test.shape[0], 3, 32, 32))
        y_train = np.asarray(train_labels)
        y_test = np.asarray(test_dict[b'labels'])
        if red_size < 1:
            from sklearn.cross_validation import train_test_split
            X_train, X_train_rem, y_train, y_rem = train_test_split(X_train, y_train, train_size=red_size)
            X_test, X_test_rem, y_test, y_test_rem = train_test_split(X_test, y_test, train_size=red_size)
        return (X_train, np_utils.to_categorical(y_train, 10)), (X_test, np_utils.to_categorical(y_test, 10))


def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
