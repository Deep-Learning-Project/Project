import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential

from DataLoader import load_data


def convert_array_to_params(bit_array: np.ndarray):
    def array_to_int(x: np.ndarray):
        return int("".join(np.array_str(x)[1:-1].split()), 2) + 1

    # Encoding: Total max of 134 bits
    # # First 2 digits for number of convolution layers. => No more than 4 layers
    n_bit_c_lay = 2
    n_conv_lay = bit_array[:n_bit_c_lay]
    n_conv_lay = array_to_int(n_conv_lay)
    # # Next 2 digits for number of fully connected layers. => No more than 4 layers
    n_bit_f_lay = 2
    n_full_lay = bit_array[n_bit_c_lay:n_bit_c_lay + n_bit_f_lay]
    n_full_lay = array_to_int(n_full_lay)
    # # 5 bits for number of channels, 3 bits for kernel height, 3 bits for kernel width
    n_bit_c_wts = 11 * n_conv_lay
    con_wts = bit_array[n_bit_c_lay + n_bit_f_lay:n_bit_c_lay + n_bit_f_lay + n_bit_c_wts]
    con_wts = con_wts.reshape(-1, 11)
    con_wts_list = []
    for row in con_wts:
        n_chan = row[:5]
        n_chan = array_to_int(n_chan)
        w = row[5:8]
        w = array_to_int(w)
        h = row[8:]
        h = array_to_int(h)
        con_wts_list.append((n_chan, w, h))
    # # 2 bits for max pooling height, 2 bits for max pooling width
    n_bit_p_dim = 4 * n_conv_lay
    pool_dims = bit_array[n_bit_c_lay + n_bit_f_lay + n_bit_c_wts:n_bit_c_lay + n_bit_f_lay + n_bit_c_wts + n_bit_p_dim]
    pool_dims = pool_dims.reshape(-1, 4)
    pool_dims_list = []
    for row in pool_dims:
        w = row[:2]
        w = array_to_int(w)
        h = row[2:]
        h = array_to_int(h)
        pool_dims_list.append((w, h))
    # # 1 bit for activation function. 0 for relu, 1 for tanh
    n_bit_a_fun = 1 * n_conv_lay
    act_fun_bits = bit_array[n_bit_c_lay + n_bit_f_lay + n_bit_c_wts + n_bit_p_dim:
    n_bit_c_lay + n_bit_f_lay + n_bit_c_wts + n_bit_p_dim + n_bit_a_fun]
    act_fun_bool = act_fun_bits.astype(bool)
    act_funs = np.empty(act_fun_bits.size, dtype='<U4')
    act_funs[act_fun_bool] = 'tanh'
    act_funs[~act_fun_bool] = 'relu'
    # # 10 bits for Dense layer dims
    n_bit_d_dim = 10 * n_full_lay
    den_lay_bits = bit_array[n_bit_c_lay + n_bit_f_lay + n_bit_c_wts + n_bit_p_dim + n_bit_a_fun:
    n_bit_c_lay + n_bit_f_lay + n_bit_c_wts + n_bit_p_dim + n_bit_a_fun + n_bit_d_dim]
    den_lay_bits = den_lay_bits.reshape(-1, 10)
    den_lay = [array_to_int(row) for row in den_lay_bits]
    # # 1 bit for linear activation function. 0 for relu, 1 for tanh
    n_bit_l_fun = 1 * n_full_lay
    act_lin_bits = bit_array[n_bit_c_lay + n_bit_f_lay + n_bit_c_wts + n_bit_p_dim + n_bit_a_fun + n_bit_d_dim:
    n_bit_c_lay + n_bit_f_lay + n_bit_c_wts + n_bit_p_dim + n_bit_a_fun + n_bit_d_dim + n_bit_l_fun]
    act_lin_bool = act_lin_bits.astype(bool)
    act_lin_funs = np.empty(act_lin_bits.size, dtype='<U4')
    act_lin_funs[act_lin_bool] = 'tanh'
    act_lin_funs[~act_lin_bool] = 'relu'

    return (n_conv_lay, n_full_lay), list(zip(con_wts_list, act_funs, pool_dims_list)), list(zip(den_lay, act_lin_funs))


def build_network(bit_array: np.ndarray, input_shape, n_classes):
    (n_conv_lay, n_full_lay), params, dens_params = convert_array_to_params(bit_array)
    conv0, act0, pool0 = params[0]
    model = Sequential()
    model.add(Convolution2D(*conv0, activation=act0, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool0))
    for layer_params in params[1:]:
        conv, act, pool = layer_params
        model.add(Convolution2D(*conv, activation=act))
        model.add(MaxPooling2D(pool_size=pool))
    model.add(Flatten())
    for out_size, act in dens_params:
        model.add(Dense(output_dim=out_size, activation=act))
    model.add(Dense(n_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    bit_array = np.random.random_integers(0, 1, 134)
    print(convert_array_to_params(bit_array))

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = load_data(red_size=0.1)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = build_network(bit_array, (3, 32, 32), 10)
    print(model.summary())
