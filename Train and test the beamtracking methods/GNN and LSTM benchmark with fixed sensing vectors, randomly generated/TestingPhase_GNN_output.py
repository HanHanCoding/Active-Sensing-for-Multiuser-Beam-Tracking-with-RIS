import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from Designed_Neural_Layers import GNN_layer
from Pilots_Generation import Contribution_receivedpilots, LS_Estimated_LowDim_Channel
from util_fun import compute_sum_rate, compute_min_rate, Generate_Beamformer

'This function is for Testing phase'
# This function is just a replication of the GNN part in the proposed network!
def GNN_output_proposed_approach(cell_UE_all, MLP_w_in, MLP_F_in,
                                 MLP_w_1, MLP1_1, MLP2_1, MLP3_1, MLP4_1,
                                 MLP_w_2, MLP1_2, MLP2_2, MLP3_2, MLP4_2,
                                 F_output, DNN_w_mapping_output,
                                 num_user, num_elements_irs, batch_size, num_antenna_bs, Pt_down_linear,
                                 tau_w, noise_sqrt_up, noise_power_linear_down, channel_combined_input):
    """
        :return: w_her_complex: tensor[batch_size, (num_elements_irs + 1), 1], dtype: Complex64
                 F_complex: tensor(batch_size, num_antenna_bs, 1, num_user), dtype: Complex64
    """

    '""""Based on the updated cell states of the LSTM, Spatial convolution on a GNN to design {w, F, V}"""""""""'
    # h_UE_all: list[tensor(batch_size, hidden_size_LSTM), tensor(batch_size, hidden_size_LSTM),..., tensor(batch_size, hidden_size_LSTM)]
    # cell_UE_all: list[tensor(batch_size, hidden_size_LSTM), tensor(batch_size, hidden_size_LSTM),..., tensor(batch_size, hidden_size_LSTM)]
    # w_RIS_down: tensor(batch_size, hidden_size_LSTM)

    'generate the INPUT representative vector for the user node'
    c_user = []
    for user in range(num_user):
        c_user.append(MLP_F_in(cell_UE_all[user]))

    input_RIS = tf.reduce_mean(c_user, axis=0)
    'generate the INPUT representative vector for the RIS node w'
    c_RIS_w = MLP_w_in(input_RIS)

    # Spatial convolution for the first time
    c_user, c_RIS_w = GNN_layer(c_user, c_RIS_w, MLP_w_1, MLP1_1, MLP2_1, MLP3_1, MLP4_1, num_user)
    # Spatial convolution for the second time
    F_UE_all, w_candidate = GNN_layer(c_user, c_RIS_w, MLP_w_2, MLP1_2, MLP2_2, MLP3_2, MLP4_2, num_user)

    '""""""""""""""""""""""""Liner Layer & Normalization for the GNN output!""""""""""""""""""""""""""""""""'
    # F_UE_all: dict{0:tensor(batch_size, hidden_size_LSTM),..., (num_user-1):tensor(batch_size, hidden_size_LSTM)}
    # RIS_phase_candidate: tensor(batch_size, hidden_size_LSTM)
    'RIS downlink reflection coefficients w'
    # w_her = DNN_w_mapping(RIS_phase_candidate)
    w_her = DNN_w_mapping_output(w_candidate)
    real_w0 = tf.reshape(w_her[:, 0:num_elements_irs], [batch_size, num_elements_irs])
    imag_w0 = tf.reshape(w_her[:, num_elements_irs:2 * num_elements_irs], [batch_size, num_elements_irs])
    real_w = real_w0 / tf.sqrt(tf.square(real_w0) + tf.square(imag_w0))
    imag_w = imag_w0 / tf.sqrt(tf.square(real_w0) + tf.square(imag_w0))
    w_her_complex = tf.complex(real_w, imag_w)
    # add extra one dimension corresponding to the direct link! current v_her_complex: array (batch_size, num_elements_irs)
    w_her_complex = tf.concat([tf.ones([batch_size, 1], tf.complex64), w_her_complex], 1)
    w_her_complex = tf.reshape(w_her_complex, [batch_size, (num_elements_irs + 1), 1])

    'Downlink Beamformer F at AP'
    # Note that at this point, for power allocation and the optimal Lagrangian multiplier lambda, we directly used a linear layer for output, without adding another DNN.
    for user in range(num_user):
        p_lambda_k_tilde = tf.abs(
            F_output(F_UE_all[user]))  # to make sure that the output power allocations are positive!
        p_lambda_k_tilde = tf.expand_dims(p_lambda_k_tilde, 2)  # p_k_tilde: tensor [batch_size, 2, 1]
        if user == 0:
            p_tilde = p_lambda_k_tilde[:, 0:1, :]
            lagrangian_lambda_tilde = p_lambda_k_tilde[:, 1:2, :]
        else:
            p_tilde = tf.concat([p_tilde, p_lambda_k_tilde[:, 0:1, :]], axis=2)  # p_tilde: tensor(batch_size, 1, K)
            lagrangian_lambda_tilde = tf.concat([lagrangian_lambda_tilde, p_lambda_k_tilde[:, 1:2, :]], axis=2)  # lagrangian_lambda_tilde: tensor(batch_size, 1, K)

    # to make sure p and 'virtual uplink power - lambda' satisfying the total power constraint
    # here, the axis can't be (1,2) like 2-norm, since the 1-norm for a matrix is equal to the maximum of L1 norm of a column of the matrix.
    p_allocation_linear = Pt_down_linear * (p_tilde / tf.norm(p_tilde, ord=1, axis=2, keepdims=True))  # p_allocation_linear: tensor(batch_size, 1, K)
    p_allocation_sqrt = tf.sqrt(p_allocation_linear)

    lambda_allocation_linear = Pt_down_linear * (lagrangian_lambda_tilde / tf.norm(lagrangian_lambda_tilde, ord=1, axis=2, keepdims=True))  # lambda_allocation_linear: tensor(batch_size, 1, K)

    '""""""""""""""""""""""""Using the designed w_her_complex as the UL sensing vector, and estimate Low-dim overall channel!""""""""""""""""""""""""""""""""'
    # 'Using the LS estimator to compute the online estimated overall low-dim channel'
    Estimated_MISO_allUE, mean_Estimated_error, Estimated_MISO_allUE_Tensor = LS_Estimated_LowDim_Channel(batch_size, num_user, num_antenna_bs, tau_w, channel_combined_input, w_her_complex, noise_sqrt_up)

    # 'Generate the Beamformer using the transmit MMSE (Regularized ZF) beamforming with the Estimated low-dim MISO channel'
    F_complex = Generate_Beamformer(Estimated_MISO_allUE, p_allocation_sqrt, lambda_allocation_linear, num_user,num_antenna_bs, Pt_down_linear, noise_power_linear_down)

    return w_her_complex, F_complex, mean_Estimated_error, Estimated_MISO_allUE_Tensor
