import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# compute the sumrate for each block for a certain frame!
# def compute_sum_rate(combined_A, w_her_complex, F_complex, batch_size, num_user, num_antenna_bs, N_w_using, noise_power_linear_down, Coefficient_pilot_overhead):
#     """
#         :param  combined_A: tensor(num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, N_w_using)
#                 w_her_complex: tensor[batch_size, (num_elements_irs + 1), 1], dtype = complex64
#                 F_complex: tensor[batch_size, num_antenna_bs, 1, num_user], dtype = complex64
#
#         :return: Sumrate_perframe: float
#                  Sumrate_count: list[float, ..., float], N_w_using items
#     """
#     Sumrate_perframe = tf.constant(0, dtype=tf.float32)
#     Sumrate_count = []
#     for block in range(N_w_using):
#         for user_1 in range(num_user):
#             # 'Tensor' object does not support item assignment, but tensor can perform slicing!!!!!!
#             channel_gain = tf.transpose(tf.matmul(combined_A[:, :, :, user_1, block], w_her_complex), [0, 2, 1]) # [batch_size, 1, num_antenna_bs]
#             signal_power = []
#             for user_2 in range(num_user):
#                 'Here we compute the signal strength using Hermitian not regular Transpose!'
#                 Gain_k = tf.squeeze(tf.matmul(tf.conj(channel_gain), F_complex[:, :, :, user_2]), axis=2) # [batch_size, 1]
#                 signal_power.append(tf.abs(Gain_k) ** 2)
#             # 'calculate the downlink rate for each UE per block'
#             SINR = signal_power[user_1] / (tf.reduce_sum(signal_power, axis=0) - signal_power[user_1] + noise_power_linear_down)
#             # 'calculate the downlink sum rate per block'
#             if user_1 == 0:
#                 Sumrate_perblock = tf.log(1 + SINR) / tf.log(2.0)
#             else:
#                 Sumrate_perblock = Sumrate_perblock + tf.log(1 + SINR) / tf.log(2.0)
#         # 'calculate the sum of the downlink sum rate per block over N blocks'
#         Sumrate_perframe += Sumrate_perblock
#         'record the computed sum rate per block'
#         rate_test_block = tf.reduce_mean(Sumrate_perblock) * Coefficient_pilot_overhead
#         Sumrate_count.append(rate_test_block)
#     # 'Average over the batch as the approximate the expectation'
#     Sumrate_perframe = tf.reduce_mean(Sumrate_perframe) * Coefficient_pilot_overhead
#
#     return Sumrate_perframe, Sumrate_count




# compute the sumrate for each block for a certain frame!
def compute_sum_rate(combined_A, w_her_complex, F_complex, batch_size, num_user, num_antenna_bs, N_w_using, noise_power_linear_down):
    """
        :param  combined_A: tensor(num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, N_w_using)
                w_her_complex: tensor[batch_size, (num_elements_irs + 1), 1], dtype = complex64
                F_complex: tensor[batch_size, num_antenna_bs, 1, num_user], dtype = complex64

        :return: Sumrate_perframe: float
                 Sumrate_count: list[float, ..., float], N_w_using items
    """
    Sumrate_perframe = tf.constant(0, dtype=tf.float32)
    Sumrate_count = []
    for block in range(N_w_using):
        for user_1 in range(num_user):
            # 'Tensor' object does not support item assignment, but tensor can perform slicing!!!!!!
            channel_gain = tf.transpose(tf.matmul(combined_A[:, :, :, user_1, block], w_her_complex), [0, 2, 1]) # [batch_size, 1, num_antenna_bs]
            signal_power = []
            for user_2 in range(num_user):
                'Here we compute the signal strength using Hermitian not regular Transpose!'
                'We assume Hermitian to represent the uplink/downlink reciprocity'
                Gain_k = tf.squeeze(tf.matmul(tf.conj(channel_gain), F_complex[:, :, :, user_2]), axis=2) # [batch_size, 1]
                signal_power.append(tf.abs(Gain_k) ** 2)
            # 'calculate the downlink rate for each UE per block'
            SINR = signal_power[user_1] / (tf.reduce_sum(signal_power, axis=0) - signal_power[user_1] + noise_power_linear_down)
            # 'calculate the downlink sum rate per block'
            if user_1 == 0:
                Sumrate_perblock = tf.log(1 + SINR) / tf.log(2.0)
            else:
                Sumrate_perblock = Sumrate_perblock + tf.log(1 + SINR) / tf.log(2.0)
        # 'calculate the sum of the downlink sum rate per block over N blocks'
        Sumrate_perframe += Sumrate_perblock
        'record the computed sum rate per block'
        rate_test_block = tf.reduce_mean(Sumrate_perblock)
        Sumrate_count.append(rate_test_block)
    # 'Average over the batch as the approximate the expectation'
    Sumrate_perframe = tf.reduce_mean(Sumrate_perframe)

    return Sumrate_perframe, Sumrate_count


# compute the min rate among the K users for each block for a certain frame!
def compute_min_rate(combined_A, w_her_complex, F_complex, batch_size, num_user, num_antenna_bs, N_w_using, noise_power_linear_down):
    """
        :param  combined_A: tensor(batch_size, num_antenna_bs, (num_elements_irs + 1), num_user, N_w_using)
                w_her_complex: tensor[batch_size, (num_elements_irs + 1), 1], dtype = complex64
                F_complex: tensor[batch_size, num_antenna_bs, 1, num_user], dtype = complex64

        :return: Minrate_perframe: float
                 Minrate_count: list[float, ..., float], N_w_using items
    """
    Minrate_perframe_adding = tf.constant(0, dtype=tf.float32)
    Minrate_count = []
    for block in range(N_w_using):
        for user_1 in range(num_user):
            # 'Tensor' object does not support item assignment, but tensor can perform slicing!!!!!!
            'Here we compute the signal strength using Hermitian not regular Transpose!'
            'We assume Hermitian to represent the uplink/downlink reciprocity'
            channel_gain = tf.transpose(tf.matmul(combined_A[:, :, :, user_1, block], w_her_complex), [0, 2, 1], conjugate=True) # [batch_size, 1, num_antenna_bs]
            signal_power = []
            for user_2 in range(num_user):
                Gain_k = tf.squeeze(tf.matmul(channel_gain, F_complex[:, :, :, user_2]), axis=2) # [batch_size, 1]
                signal_power.append(tf.abs(Gain_k) ** 2)
            # 'calculate the downlink rate for each UE per block'
            SINR = signal_power[user_1] / (tf.reduce_sum(signal_power, axis=0) - signal_power[user_1] + noise_power_linear_down) # SINR: tensor[batch_size, 1]
            rate_kk = tf.log(1 + SINR) / tf.log(2.0) # rate_kk: tensor[batch_size, 1]
            # record the rate for the K users
            if user_1 == 0:
                rate_K_users_perblock = rate_kk
            else:
                rate_K_users_perblock = tf.concat([rate_K_users_perblock, rate_kk], axis=1)

        # rate_K_users_perblock: tensor[batch_size, K]
        Minrate_block = tf.reduce_min(rate_K_users_perblock, axis=1) # tensor[batch_size, ]
        Minrate_perframe_adding += Minrate_block

        # record the min rate per block
        Minrate_count.append(tf.reduce_mean(Minrate_block))

    # Averaging the min rate over the N blocks each frame
    Minrate_perframe = Minrate_perframe_adding / N_w_using
    Minrate_perframe = tf.reduce_mean(Minrate_perframe)

    return Minrate_perframe, Minrate_count




# Generate the Beamformer using the transmit MMSE (Regularized ZF) beamforming for a certain frame!
def Generate_Beamformer(Estimated_MISO_allUE, p_allocation_sqrt, lambda_allocation_linear, num_user, num_antenna_bs, Pt_down_linear, noise_power_linear_down):
    """
        :param  Estimated_MISO_allUE: dict{0:tensor[num_samples, num_antenna_bs, 1],..., K-1:tensor[num_samples, num_antenna_bs, 1]}
                p_allocation_sqrt: tensor(batch_size, 1, K), non-negative float
                lambda_allocation_linear: tensor(batch_size, 1, K), non-negative float

        :return: B_complex: tensor[batch_size, num_antenna_bs, 1, num_user], dtype=complex64
    """
    'Generate the Combined MISO channel and compute the inner summation of the transmit MMSE beamforming'
    for ii in range(num_user):
        # make use of the broadcasting rules in matrix multiplication!
        g_MISO_k = Estimated_MISO_allUE[ii] # g_MISO_k: tensor[batch_size, num_antenna_bs, 1], dtype=complex64

        #compute the inner summation
        g_MISO_k_Hermitian = tf.transpose(g_MISO_k, [0, 2, 1], conjugate=True) # tensor[batch_size, 1, num_antenna_bs], dtype=complex64
        priority = tf.expand_dims(lambda_allocation_linear[:, :, ii] / noise_power_linear_down, 2) # priority: tensor[batch_size, 1, 1], dtype=float32

        Inner_summation_ii_real = priority * tf.real(tf.matmul(g_MISO_k, g_MISO_k_Hermitian))
        Inner_summation_ii_imag = priority * tf.imag(tf.matmul(g_MISO_k, g_MISO_k_Hermitian))
        Inner_summation_ii = tf.complex(Inner_summation_ii_real, Inner_summation_ii_imag)  # tensor[batch_size, num_antenna_bs, num_antenna_bs], dtype=complex64

        if ii == 0:
            Inner_summation = Inner_summation_ii
        else:
            Inner_summation = Inner_summation + Inner_summation_ii # tensor[batch_size, num_antenna_bs, num_antenna_bs], dtype=complex64

    'Generate the Beamformer using the transmit MMSE beamforming'
    p_allocation_sqrt = tf.expand_dims(p_allocation_sqrt, 2) # p_allocation_sqrt: tensor [batch_size, 1, 1, K]
    # make use of the broadcasting rules in matrix multiplication/inversion!
    Identity_matrix = tf.eye(num_antenna_bs, dtype=tf.complex64)
    for kk in range(num_user):
        # for linalg.inv, The input is a tensor of shape [..., M, M] whose inner-most 2 dimensions form square matrices (allow Broadcasting!)
        Matrix_inverse = tf.linalg.inv(Identity_matrix + Inner_summation) # tensor[batch_size, num_antenna_bs, num_antenna_bs], dtype=complex64
        Numerator_k = tf.matmul(Matrix_inverse, Estimated_MISO_allUE[kk]) # Numerator_k: tensor[batch_size, num_antenna_bs, 1], dtype=complex64
        Denominator_k = tf.norm(Numerator_k, ord='euclidean', axis=(1, 2), keepdims=True) # Denominator_k: tensor[batch_size, 1, 1], dtype=complex64
        Beamforming_direction_k = Numerator_k / Denominator_k # Beamforming_direction: tensor[batch_size, num_antenna_bs, 1], dtype=complex64

        # since in Tensorflow, the real number cannot be directly multiplied by complex vector
        Beamformer_k_real = p_allocation_sqrt[:, :, :, kk] * tf.real(Beamforming_direction_k)
        Beamformer_k_imag = p_allocation_sqrt[:, :, :, kk] * tf.imag(Beamforming_direction_k)
        Beamformer_k = tf.complex(Beamformer_k_real, Beamformer_k_imag) # tensor[batch_size, num_antenna_bs, 1], dtype=complex64

        # record the generated beamformers
        if kk == 0:
            B_complex = tf.expand_dims(Beamformer_k, 3) # tensor[batch_size, num_antenna_bs, 1, 1], dtype=complex64
        else:
            B_complex = tf.concat([B_complex, tf.expand_dims(Beamformer_k, 3)], axis=3)

    return B_complex



def compute_pattern_gain_UL(combined_A_range_all, V_collect):
    """
        :param  combined_A_range_all: array [num_antenna_bs, (num_elements_irs + 1)]
                V_collect: array [(num_elements_irs + 1), 1]

        :return: RIS_patternGain_UL: float
    """
    RIS_patternGain_UL_tmp = combined_A_range_all @ V_collect
    RIS_patternGain_UL = (np.linalg.norm(RIS_patternGain_UL_tmp)) ** 2

    return RIS_patternGain_UL


def compute_pattern_gain_DL(combined_A_range_all, w_collect, F_collect, num_user, num_antenna_bs):
    """
        :param  combined_A_range_all: array (num_antenna_bs, (num_elements_irs + 1))
                w_collect: array [(num_elements_irs + 1), 1]
                F_collect: array [num_antenna_bs, 1, num_user]

        :return: RIS_patternGain_DL: float
    """
    channel_gain = np.reshape(combined_A_range_all @ w_collect, [1, num_antenna_bs])
    RIS_patternGain_DL_tmp = 0.0
    for user in range(num_user):
        RIS_patternGain_DL_tmp += channel_gain @ F_collect[:, :, user]

    RIS_patternGain_DL = (np.abs(RIS_patternGain_DL_tmp)) ** 2

    return RIS_patternGain_DL

def compute_array_response_irs(a_bs_irs, a_irs_user, V_collect_no_direct):
    """
        :param  a_bs_irs: array (num_elements_irs, 1)
                a_irs_user: array [num_elements_irs, 1]
                V_collect_no_direct: array [num_elements_irs, 1]

        :return: Array_response_V: float
    """
    tmp = V_collect_no_direct * a_bs_irs.conjugate() # same as a_bs_irs.conjugate() @ diag(V_collect_no_direct)
    tmp = tmp.reshape((1, -1)) @ a_irs_user

    Array_response_V = np.abs(tmp)

    return Array_response_V


def compute_array_response_Beamformer_allUE(F_collect, num_user, num_antenna_bs, N_angles):
    """
        :param  F_collect: array [num_antenna_bs, 1, num_user]

        :return: Array_response_F: [N_angles, num_user], N_angles = 1000
    """
    Array_response_F = np.empty([N_angles, num_user])

    phi_1 = np.linspace(0, np.pi, N_angles).reshape([N_angles, 1])
    BS_antenna = np.arange(num_antenna_bs).reshape([1, num_antenna_bs])
    for kk in range(num_user):
        a_BS_k = np.exp(1j * np.pi * np.matmul(np.cos(phi_1), BS_antenna)) # [N_angles, num_antenna_bs]
        tmp = np.squeeze(np.matmul(np.conj(a_BS_k), F_collect[:, :, kk])) # [N_angles,]
        Array_response_F[:, kk] = np.abs(tmp)

    return Array_response_F


# compute the SINR for the UE k within the how range! ---> Since visualization, only one sample!
def compute_SINR_all_range(combined_A_range_all, w_collect, F_collect, num_user, noise_power_linear_down):
    """
        :param  combined_A_range_all: array (num_antenna_bs, (num_elements_irs + 1))
                w_collect: array [(num_elements_irs + 1), 1]
                F_collect: array [num_antenna_bs, 1, num_user]

        :return: SINR_allUE_perLocation_perFrame: array [num_user]
    """
    'the SINR for the whole range!'
    SINR_allUE_perLocation_perFrame = np.empty([num_user])
    for user_1 in range(num_user):
        channel_gain = np.conj(np.transpose(combined_A_range_all @ w_collect)) # [1, num_antenna_bs]
        signal_power = np.empty([num_user])
        for user_2 in range(num_user):
            Gain_k = np.squeeze(channel_gain @ F_collect[:, :, user_2]) # []
            signal_power[user_2] = (np.abs(Gain_k)) ** 2
        # 'calculate the downlink rate for each UE
        SINR_k = signal_power[user_1] / (np.sum(signal_power) - signal_power[user_1] + noise_power_linear_down)

        SINR_allUE_perLocation_perFrame[user_1] = 10 * np.log10(SINR_k)
        # rate_allUE_perLocation_perFrame[user_1] = np.log2(SINR_k)

    return SINR_allUE_perLocation_perFrame



# compute the SINR for the UE k within the how range! ---> Since visualization, only one sample!
def compute_SNR_SINR_table(combined_A_UE_location, w_collect, F_collect, num_user, noise_power_linear_down):
    """
        :param  w_collect: array [(num_elements_irs + 1), 1]
                F_collect: array [num_antenna_bs, 1, num_user]
                combined_A_UE_location: [num_antenna_bs, (num_elements_irs + 1), num_user]

        :return: SNR_SINR_table: array [num_user+1, num_user]
    """

    'the signal power for each channel(at UEs true location)/beamformer combination'
    SNR_SINR_table = np.empty([num_user+1, num_user])
    for user_1 in range(num_user): # columns -- i
        channel_gain = np.conj(np.transpose(combined_A_UE_location[:, :, user_1] @ w_collect))  # [1, num_antenna_bs]
        signal_power = np.empty([num_user])
        for user_2 in range(num_user): # rows -- j
            Gain_k = np.squeeze(channel_gain @ F_collect[:, :, user_2])  # []
            signal_power[user_2] = (np.abs(Gain_k)) ** 2

            # RECORD THE SNR in dB scale
            SNR_SINR_table[user_2, user_1] = 10 * np.log10((np.abs(Gain_k)) ** 2)

        # 'calculate the downlink SINR for each UE
        SINR_k = signal_power[user_1] / (np.sum(signal_power) - signal_power[user_1] + noise_power_linear_down)

        SNR_SINR_table[-1, user_1] = 10 * np.log10(SINR_k)


    return SNR_SINR_table
















