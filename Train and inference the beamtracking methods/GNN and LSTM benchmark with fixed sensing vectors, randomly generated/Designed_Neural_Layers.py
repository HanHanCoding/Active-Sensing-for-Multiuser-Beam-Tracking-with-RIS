import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras.layers import BatchNormalization,Dense,Layer

'RNN -- GRU variant'
# ----------------------------------------------------------------------------------------------------
# 'RNN layers'
# # 1. units are the dimensionality of the output space!
# #    for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
# #    so we only need to ensure that the batch size are consistant, means we need to use tf.concat([a,b], axis=1)! key: [axis = 1]
# # 2. Keras Dense Layer already includes weights and bias -> bias is set to default as true!

class GRU_model(Layer):
    def __init__(self):
        super(GRU_model, self).__init__()
        self.layer_Wz = Dense(units=512, activation='linear')
        self.layer_Uz = Dense(units=512, activation='linear')
        self.layer_Wr = Dense(units=512, activation='linear')
        self.layer_Ur = Dense(units=512, activation='linear')
        self.layer_Wh = Dense(units=512, activation='linear')
        self.layer_Uh = Dense(units=512, activation='linear')

    def call(self, input_x, h_old):
        z_t = tf.sigmoid(self.layer_Wz(input_x) + self.layer_Uz(h_old))
        r_t = tf.sigmoid(self.layer_Wr(input_x) + self.layer_Ur(h_old))
        h_t_candidate = tf.tanh(self.layer_Wh(input_x) + self.layer_Uh(r_t * h_old))
        h_t = z_t * h_old + (1 - z_t) * h_t_candidate
        return h_t

class LSTM_model(Layer):
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.layer_Ui = Dense(units=512, activation='linear')
        self.layer_Wi = Dense(units=512, activation='linear')
        self.layer_Uf = Dense(units=512, activation='linear')
        self.layer_Wf = Dense(units=512, activation='linear')
        self.layer_Uo = Dense(units=512, activation='linear')
        self.layer_Wo = Dense(units=512, activation='linear')
        self.layer_Uc = Dense(units=512, activation='linear')
        self.layer_Wc = Dense(units=512, activation='linear')

    def call(self, input_x, h_old, c_old):
        i_t = tf.sigmoid(self.layer_Ui(input_x) + self.layer_Wi(h_old))
        f_t = tf.sigmoid(self.layer_Uf(input_x) + self.layer_Wf(h_old))
        o_t = tf.sigmoid(self.layer_Uo(input_x) + self.layer_Wo(h_old))
        c_t = tf.tanh(self.layer_Uc(input_x) + self.layer_Wc(h_old))
        c_new = i_t * c_t + f_t * c_old
        h_new = o_t * tf.tanh(c_new)
        return h_new, c_new
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
'DNNs'
'Note that all DNNs last layers, the linear (output) layer, are defined in the main program because its easier to define the size of the output tensor this way!'
# ----------------------------------------------------------------------------------------------------
# 这是用来替代原来LSTM network中链接cell state的v mapping DNN
# class DNN_mapping_V(Layer):
#     def __init__(self):
#         super(DNN_mapping_V, self).__init__()
#         self.A1 = Dense(units=1024, activation='relu')
#         self.A2 = Dense(units=1024, activation='relu')
#         self.A3 = Dense(units=1024, activation='relu')
#
#     def call(self, h_RIS):
#         g1 = self.A1(h_RIS)
#         # g1 = BatchNormalization()(g1)
#         g2 = self.A2(g1)
#         # g2 = BatchNormalization()(g2)
#         g3 = self.A3(g2)
#         # g3 = BatchNormalization()(g3)
#         return g3
#
# class DNN_mapping_w(Layer):
#     def __init__(self):
#         super(DNN_mapping_w, self).__init__()
#         self.A1 = Dense(units=1024, activation='relu')
#         self.A2 = Dense(units=1024, activation='relu')
#         self.A3 = Dense(units=1024, activation='relu')
#
#     def call(self, h_RIS):
#         g1 = self.A1(h_RIS)
#         g1 = BatchNormalization()(g1)
#         g2 = self.A2(g1)
#         g2 = BatchNormalization()(g2)
#         g3 = self.A3(g2)
#         g3 = BatchNormalization()(g3)
#         return g3

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
'GNN -- Spatial Convolution'
# ----------------------------------------------------------------------------------------------------
class MLP_input_F(Layer):
    def __init__(self):
        super(MLP_input_F, self).__init__()
        self.linear_1 = Dense(units=1024, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLP_input_w(Layer):
    def __init__(self):
        super(MLP_input_w, self).__init__()
        self.linear_1 = Dense(units=1024, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x

# class MLP_input_V(Layer):
#     def __init__(self):
#         super(MLP_input_V, self).__init__()
#         self.linear_1 = Dense(units=1024, activation='relu')
#         self.linear_2 = Dense(units=512, activation='relu')
#
#     def call(self, inputs):
#         x = self.linear_1(inputs)
#         x = self.linear_2(x)
#         return x


class MLPBlock_w_node(Layer):
    def __init__(self):
        super(MLPBlock_w_node, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x

# class MLPBlock_V_node(Layer):
#     def __init__(self):
#         super(MLPBlock_V_node, self).__init__()
#         self.linear_1 = Dense(units=512, activation='relu')
#         self.linear_2 = Dense(units=512, activation='relu')
#
#     def call(self, inputs):
#         x = self.linear_1(inputs)
#         x = self.linear_2(x)
#         return x


class MLPBlock1(Layer):
    def __init__(self):
        super(MLPBlock1, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock2(Layer):
    def __init__(self):
        super(MLPBlock2, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock3(Layer):
    def __init__(self):
        super(MLPBlock3, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock4(Layer):
    def __init__(self):
        super(MLPBlock4, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x

# class MLPBlock5(Layer):
#     def __init__(self):
#         super(MLPBlock5, self).__init__()
#         self.linear_1 = Dense(units=512, activation='relu')
#         self.linear_2 = Dense(units=512, activation='relu')
#
#     def call(self, inputs):
#         x = self.linear_1(inputs)
#         x = self.linear_2(x)
#         return x
#
# class MLPBlock6(Layer):
#     def __init__(self):
#         super(MLPBlock6, self).__init__()
#         self.linear_1 = Dense(units=512, activation='relu')
#         self.linear_2 = Dense(units=512, activation='relu')
#
#     def call(self, inputs):
#         x = self.linear_1(inputs)
#         x = self.linear_2(x)
#         return x

# a single layer of GNN
def GNN_layer(x,w,MLP_w,MLP1,MLP2,MLP3,MLP4,num_user):
    # for input:
    # total K tensors in the list <-------> K UEs
    # hidden_size_LSTM <-------> size of the cell state of the LSTM
    # hidden_size_GNN <------->  output size of the GNN MLP layers

    # The reason we define dict and list here is b/c for tensor, 'Tensor' object does not support item assignment
    # i.e., [INVALID] h_old_init = tf.zeros([batch_size, hidden_size_LSTM,1000]) ----> h_old_init[:,:,0] = tf.zeros([batch_size, hidden_size_LSTM])
    """
        :param: x: list[tensor(batch_size, hidden_size_LSTM), tensor(batch_size, hidden_size_LSTM),..., tensor(batch_size, hidden_size_LSTM)]
                w: tensor(batch_size, hidden_size_LSTM)

        :return:x_out: dict{0:tensor(batch_size, hidden_size_GNN), 1:tensor(batch_size, hidden_size_GNN),..., K-1:tensor(batch_size, hidden_size_GNN)}
                w_out: tensor(batch_size, hidden_size_GNN)
    """
    w0 = MLP_w(w)
    x_out = {0:0} #define the output x as a dictionary!, corresponding to K different users!
    'update the user node k ---> (A+C) block 1'
    for ii in range(num_user):
        tmp = []
        for jj in range(num_user):
            if jj != ii:
                tmp.append(MLP1(x[jj]))
        x_max = tf.reduce_max(tmp, axis=0)
        if num_user == 1:
            x_concate = tf.concat([x[ii], w0], axis=1)
        else:
            x_concate = tf.concat([x[ii], x_max, w0], axis=1)
        # since MLPs are DENSE layers, the axis-1 of the input 2D tensor is automatically gotten rid of!
        x_out[ii] = MLP2(x_concate)

    'update the RIS node w ---> (A+C) block 2'
    tmp = []
    for ii in range(num_user):
        tmp.append(MLP3(x[ii]))
    x_mean = tf.reduce_mean(tmp, axis=0)
    x_concate = tf.concat([x_mean, w0], axis=1)
    w_out = MLP4(x_concate)

    return x_out, w_out


