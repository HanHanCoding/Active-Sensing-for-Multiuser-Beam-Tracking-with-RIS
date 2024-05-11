import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import scipy.io as sio
import os

import time

def DFT_matrix(N):
    """
        :param  N: int

        :return: W: array (N, N)
    """
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(- 2 * np.pi * 1j / N)
    # remember we do not need the normalization factor (1/sqrt(N))
    # W = np.power(omega, i * j) / np.sqrt(N)
    W = np.power(omega, i * j, dtype=np.complex64)
    return W


# generate the contribution from each UE in a single UL phase!
def Contribution_receivedpilots(num_samples, num_user, num_antenna_bs, subblocks_L, U_repeat, combined_A, sensing_vector_V_all, noise_sqrt_up):
    """
        :param: num_user, num_antenna_bs, subblocks_L, U_repeat: int
                noise_sqrt_up: float
                num_samples: = batch_size
                combined_A: tensor(batch_size, num_antenna_bs, (num_elements_irs + 1), num_user), dtype = complex64 ---------------Notice, we only input a certain block of combined_A in the main program
                sensing_vector_V_all: tensor(batch_size, (num_elements_irs + 1), subblocks_L)

        :return: Ybar_all: list[tensor(batch_size, num_antenna_bs, subblocks_L, U_repeat),..., tensor(batch_size, num_antenna_bs, subblocks_L, U_repeat)], in total num_user items

    """
    len_pilot_per_subblock = num_user  # len_pilot_per_subblock: int (equivalent to num_user in the frame structure)
    'Generate orthogonal pilot sequences x_k at each sub-block'
    # pilot sequence is generated based on the DFT matrix!
    pilotseq_UE_all = DFT_matrix(len_pilot_per_subblock)  # notice, the pilot sequence will be sent repeatedly

    'decorrelation!'
    Ybar_all = []
    for user in range(num_user):
        # the decorrelated noiseless contribution for each UE
        ybar_k_noiseless = tf.matmul(combined_A[:, :, :, user], sensing_vector_V_all)  # tensor(batch_size, num_antenna_bs, subblocks_L)
        pilotseq_UE_k = tf.expand_dims(pilotseq_UE_all[user, :], 1) # [len_pilot_per_subblock, 1]
        for uu in range(U_repeat): #eq. (4) in the note
            # 'generate the equivalent noise for each UE'
            for ll in range(subblocks_L): #eq. (3) in the note
                noise_l_u = tf.complex(tf.random_normal([num_samples, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5), tf.random_normal([num_samples, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5))
                noise_bar_k_l_u = (1 / len_pilot_per_subblock) * tf.matmul(noise_l_u, pilotseq_UE_k) # tensor [num_samples, num_antenna_bs, 1]
                # noise_bar_k_u: tensor [num_samples, num_antenna_bs, subblocks_L]
                if ll == 0:
                    noise_bar_k_u = noise_bar_k_l_u
                else:
                    noise_bar_k_u = tf.concat([noise_bar_k_u, noise_bar_k_l_u], axis=2)

            Y_bar_k_u = tf.expand_dims((ybar_k_noiseless + noise_sqrt_up * noise_bar_k_u), axis=3)
            # Y_bar_k: tensor [num_samples, num_antenna_bs, subblocks_L, U_repeat]
            if uu == 0:
                Y_bar_k = Y_bar_k_u
            else:
                Y_bar_k = tf.concat([Y_bar_k, Y_bar_k_u], axis=3)

        Ybar_all.append(Y_bar_k)

    return Ybar_all


# this function is to compute the online estimated overall low-dim channel at the 0-th block of each frame'
def LS_Estimated_LowDim_Channel(num_samples, num_user, num_antenna_bs, tau_w, combined_A, w_her_complex, noise_sqrt_up):
    """
        [the online received pilots]
        :param: tau_w: int ---> The pilot length in phase II
                num_user, num_antenna_bs, subblocks_L, U_repeat: int
                noise_sqrt_up: float
                combined_A: tensor(num_samples, num_antenna_bs, (num_elements_irs + 1), num_user), dtype = complex64 ---------------Notice, we only input a certain block of combined_A in the main program
                w_her_complex: tensor(num_samples, (num_elements_irs + 1), 1)


        :return: Estimated_MISO_allUE: dict{0:tensor[num_samples, num_antenna_bs, 1],..., K-1:tensor[num_samples, num_antenna_bs, 1]}
                 Estimated_MISO_allUE_Tensor: tensor(batch_size_test_1block, num_antenna_bs, 1, num_user)
    """

    subblocks_L = 1
    U_repeat = 1

    'The received "online" pilots'
    # the difference btw here and the 'Stats_LMMSE_offline' function is the different input: combined_A
    # here is the online actual channel, which is unknown in testing

    len_pilot_per_subblock = tau_w  # len_pilot_per_subblock: int
    'Generate orthogonal pilot sequences x_k at each sub-block'
    # pilot sequence is generated based on the DFT matrix!
    pilotseq_UE_all = DFT_matrix(len_pilot_per_subblock)  # notice, the pilot sequence will be sent repeatedly

    Estimated_MISO_allUE = {0:0}
    for user in range(num_user):
        '------------------------------------------for each batch, decorrelation contribution for each UE (channel_k) -----------------------------------------------'
        # the decorrelated noiseless contribution for each UE
        ybar_k_noiseless = tf.matmul(combined_A[:, :, :, user], w_her_complex)  # tensor(num_samples, num_antenna_bs, 1)
        pilotseq_UE_k = tf.expand_dims(pilotseq_UE_all[user, :], 1) # [len_pilot_per_subblock, 1]
        for uu in range(U_repeat): #eq. (4) in the note
            # 'generate the equivalent noise for each UE'
            for ll in range(subblocks_L): #eq. (3) in the note
                noise_l_u = tf.complex(tf.random_normal([num_samples, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5), tf.random_normal([num_samples, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5))
                noise_bar_k_l_u = (1 / len_pilot_per_subblock) * tf.matmul(noise_l_u, pilotseq_UE_k) # tensor [num_samples, num_antenna_bs, 1]
                # noise_bar_k_u: tensor [num_samples, num_antenna_bs, subblocks_L]
                if ll == 0:
                    noise_bar_k_u = noise_bar_k_l_u
                else:
                    noise_bar_k_u = tf.concat([noise_bar_k_u, noise_bar_k_l_u], axis=2)

            Y_bar_k_u = tf.expand_dims((ybar_k_noiseless + noise_sqrt_up * noise_bar_k_u), axis=3)

            if uu == 0:
                Y_bar_k = Y_bar_k_u
            else:
                Y_bar_k = tf.concat([Y_bar_k, Y_bar_k_u], axis=3) # Y_bar_k: tensor [num_samples, num_antenna_bs, 1, 1]

        Y_bar_k = tf.squeeze(Y_bar_k, axis=3) # Y_bar_k: tensor(num_samples, num_antenna_bs, 1)

        '------------------------------------------ compute the online estimated overall low-dim channel g_MISO_k for user k -----------------------------------------------'
        # by using the LS estimator here (y = h + z), the estimated channel k is just the contribution from UE k.
        Estimated_MISO_allUE[user] = Y_bar_k

        # Compute the Estimated error and also, record the tensor-representation of the estimated channel
        Estimated_error_k = (tf.abs(tf.norm(tf.squeeze(Y_bar_k, axis=2) - tf.squeeze(ybar_k_noiseless, axis=2), ord=2, axis=1)))**2 / (tf.abs(tf.norm(tf.squeeze(Y_bar_k, axis=2), ord=2, axis=1)))**2
        if user == 0:
            mean_Estimated_error = tf.reduce_mean(Estimated_error_k)
            Estimated_MISO_allUE_Tensor = tf.expand_dims(Y_bar_k, axis=3)
        else:
            mean_Estimated_error = mean_Estimated_error + tf.reduce_mean(Estimated_error_k)
            Estimated_MISO_allUE_Tensor = tf.concat([Estimated_MISO_allUE_Tensor, tf.expand_dims(Y_bar_k, axis=3)], axis=3)

    mean_Estimated_error = mean_Estimated_error / num_user


    return Estimated_MISO_allUE, mean_Estimated_error, Estimated_MISO_allUE_Tensor


'This function is for generating the offline channel stats for LMMSE estimation'
# this function is for generating the contribution from each UE offline for LMMSE estimation!
# each sensing vector 'w' in every batch will corresponding to 10000 realizations of the offline generated combined channel A!
def Stats_LMMSE_offline(num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs, combined_A, w_her_complex, noise_bar_allUE, noise_sqrt_up):
    """
        :param: num_user, num_antenna_bs, num_samples, batch_size_LMMSE_1block: int
                noise_sqrt_up: float

                -------Notice, here, ***num_samples*** means corresponding to larger batch size either in training/testing-----
                noise_bar_allUE: tensor [***num_samples*** ,batch_size_LMMSE_1block, num_antenna_bs, K].

                combined_A: tensor(batch_size_LMMSE_1block, num_antenna_bs, (num_elements_irs + 1), num_user), dtype = complex64 ---------------Notice, we only input a certain block of combined_A in the main program
                w_her_complex: tensor(num_samples, (num_elements_irs + 1), 1)


        :return: mean_Y_bar_batch: tensor(num_samples, num_antenna_bs, 1, K)
                 mean_LowDim_overallChannal_batch: tensor(num_samples, num_antenna_bs, 1, K)
                 C_yh_batch: tensor(num_samples, num_antenna_bs, num_antenna_bs, K)
                 C_yy_batch: tensor(num_samples, num_antenna_bs, num_antenna_bs, K)
    """

    len_pilot_per_subblock = tau_w  # len_pilot_per_subblock: int
    'Generate orthogonal pilot sequences x_k at each sub-block'
    # pilot sequence is generated based on the DFT matrix!
    pilotseq_UE_all = DFT_matrix(len_pilot_per_subblock)  # notice, the pilot sequence will be sent repeatedly


    for user in range(num_user):
        '------------------------------------------for the whole training/testing batch, decorrelation contribution for each UE (channel_k)-----------------------------------------------'
        # the decorrelated noiseless contribution for each UE
        # make use of the broadcasting rule (still applicable with two batch dimensions)!
        ybar_k_noiseless = tf.matmul(tf.expand_dims(combined_A[:, :, :, user], axis=0), tf.expand_dims(w_her_complex, axis=1))  # ybar_k_noiseless: tensor(num_samples, batch_size_LMMSE_1block, num_antenna_bs, 1)

        # the noise for each UE ---> slicing the needed portion out of the larger batch size
        noise_bar_k = tf.expand_dims(noise_bar_allUE[0:num_samples, :, :, user], axis=3) # tensor [num_samples, batch_size_LMMSE_1block, num_antenna_bs, 1]

        # pilotseq_UE_k = tf.expand_dims(pilotseq_UE_all[user, :], 1) # [len_pilot_per_subblock, 1]
        # noise_k = tf.complex(tf.random_normal([num_samples, batch_size_LMMSE_1block, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5),tf.random_normal([num_samples, batch_size_LMMSE_1block, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5))
        # noise_bar_k = (1 / len_pilot_per_subblock) * tf.matmul(noise_k, pilotseq_UE_k)  # tensor [num_samples, batch_size_LMMSE_1block, num_antenna_bs, 1]

        # The contribution from each UE (with noise)
        Y_bar_k = ybar_k_noiseless + noise_sqrt_up * noise_bar_k # tensor [num_samples, batch_size_LMMSE_1block, num_antenna_bs, 1]

        '------------------------------------------ for the whole training/testing batch, compute its corresponding LMMSE stats for each UE (channel_k) -----------------------------------------------'
        LowDim_overallChannal_k = ybar_k_noiseless  # low_dim_overallChannal_k: tensor(num_samples, batch_size_LMMSE_1block, num_antenna_bs, 1)

        mean_Y_bar_k = tf.reduce_mean(Y_bar_k, axis=1, keepdims=True) # tensor(num_samples, 1, num_antenna_bs, 1)
        mean_LowDim_overallChannal_k = tf.reduce_mean(LowDim_overallChannal_k, axis=1, keepdims=True) # tensor(num_samples, 1, num_antenna_bs, 1)

        ZeroMean_Y_bar_k = Y_bar_k - mean_Y_bar_k # tensor(num_samples, batch_size_LMMSE_1block, num_antenna_bs, 1)
        ZeroMean_LowDim_overallChannal_k = LowDim_overallChannal_k - mean_LowDim_overallChannal_k # tensor(num_samples, batch_size_LMMSE_1block, num_antenna_bs, 1)

        '# compute the covariance matrix'
        # tf.linalg.matrix_transpose: Transposes last two dimensions of the input tensor
        C_yh = tf.reduce_mean(tf.matmul(ZeroMean_Y_bar_k, tf.matrix_transpose(ZeroMean_LowDim_overallChannal_k, conjugate=True)), axis=1) # tensor(num_samples, num_antenna_bs, num_antenna_bs)
        C_yy = tf.reduce_mean(tf.matmul(ZeroMean_Y_bar_k, tf.matrix_transpose(ZeroMean_Y_bar_k, conjugate=True)), axis=1) # tensor(num_samples, num_antenna_bs, num_antenna_bs)

        # add one dim to the outputs ---> prepare for recording from all UEs
        mean_Y_bar_k = tf.expand_dims(tf.squeeze(mean_Y_bar_k, axis=1), axis=3) # tensor(num_samples, num_antenna_bs, 1, 1)
        mean_LowDim_overallChannal_k = tf.expand_dims(tf.squeeze(mean_LowDim_overallChannal_k, axis=1), axis=3) # tensor(num_samples, num_antenna_bs, 1, 1)
        C_yh = tf.expand_dims(C_yh, axis=3) # tensor(num_samples, num_antenna_bs, num_antenna_bs, 1)
        C_yy = tf.expand_dims(C_yy, axis=3)# tensor(num_samples, num_antenna_bs, num_antenna_bs, 1)

        'record the stats for all UE'
        if user == 0:
            mean_Y_bar_batch = mean_Y_bar_k
            mean_LowDim_overallChannal_batch = mean_LowDim_overallChannal_k
            C_yh_batch = C_yh
            C_yy_batch = C_yy
        else:
            mean_Y_bar_batch = tf.concat([mean_Y_bar_batch, mean_Y_bar_k], axis=3) # # tensor(num_samples, num_antenna_bs, 1, K)
            mean_LowDim_overallChannal_batch = tf.concat([mean_LowDim_overallChannal_batch, mean_LowDim_overallChannal_k], axis=3) # tensor(num_samples, num_antenna_bs, 1, K)
            C_yh_batch = tf.concat([C_yh_batch, C_yh], axis=3) # tensor(num_samples, num_antenna_bs, num_antenna_bs, K)
            C_yy_batch = tf.concat([C_yy_batch, C_yy], axis=3) # tensor(num_samples, num_antenna_bs, num_antenna_bs, K)

    return mean_Y_bar_batch, mean_LowDim_overallChannal_batch, C_yh_batch, C_yy_batch



# this function is to compute the online estimated overall low-dim channel at the 0-th block of each frame'
def Estimated_LowDim_Channel(mean_Y_bar_batch, mean_LowDim_overallChannal_batch, C_yh_batch, C_yy_batch, num_samples, num_user, num_antenna_bs, tau_w, combined_A, w_her_complex, noise_sqrt_up):
    """
        [LMMSE offline STATS]
        :param: mean_Y_bar_batch: tensor(num_samples, num_antenna_bs, 1, K)
                mean_LowDim_overallChannal_batch: tensor(num_samples, num_antenna_bs, 1, K)
                C_yh_batch: tensor(num_samples, num_antenna_bs, num_antenna_bs, K)
                C_yy_batch: tensor(num_samples, num_antenna_bs, num_antenna_bs, K)

        [the online received pilots]
        :param: num_user, num_antenna_bs, subblocks_L, U_repeat: int
                noise_sqrt_up: float
                combined_A: tensor(num_samples, num_antenna_bs, (num_elements_irs + 1), num_user), dtype = complex64 ---------------Notice, we only input a certain block of combined_A in the main program
                w_her_complex: tensor(num_samples, (num_elements_irs + 1), 1)


        :return: Estimated_MISO_allUE: dict{0:tensor[num_samples, num_antenna_bs, 1],..., K-1:tensor[num_samples, num_antenna_bs, 1]}
                 Estimated_MISO_allUE_Tensor: tensor(batch_size_test_1block, num_antenna_bs, 1, num_user)
    """

    'Since here is the sensing phase-II, only one RIS sensing vector: w'
    subblocks_L = 1
    U_repeat = 1

    'The received "online" pilots'
    # the difference btw here and the 'Stats_LMMSE_offline' function is the different input: combined_A
    # here is the online actual channel, which is unknown in testing

    len_pilot_per_subblock = tau_w  # len_pilot_per_subblock: int
    'Generate orthogonal pilot sequences x_k at each sub-block'
    # pilot sequence is generated based on the DFT matrix!
    pilotseq_UE_all = DFT_matrix(len_pilot_per_subblock)  # notice, the pilot sequence will be sent repeatedly

    Estimated_MISO_allUE = {0:0}
    for user in range(num_user):
        '------------------------------------------for each batch, decorrelation contribution for each UE (channel_k) -----------------------------------------------'
        # the decorrelated noiseless contribution for each UE
        ybar_k_noiseless = tf.matmul(combined_A[:, :, :, user], w_her_complex)  # tensor(num_samples, num_antenna_bs, 1)
        pilotseq_UE_k = tf.expand_dims(pilotseq_UE_all[user, :], 1) # [len_pilot_per_subblock, 1]

        for uu in range(U_repeat): #eq. (4) in the note
            # 'generate the equivalent noise for each UE'
            for ll in range(subblocks_L): #eq. (3) in the note
                noise_l_u = tf.complex(tf.random_normal([num_samples, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5), tf.random_normal([num_samples, num_antenna_bs, len_pilot_per_subblock], mean=0.0, stddev=0.5))
                noise_bar_k_l_u = (1 / len_pilot_per_subblock) * tf.matmul(noise_l_u, pilotseq_UE_k) # tensor [num_samples, num_antenna_bs, 1]
                # noise_bar_k_u: tensor [num_samples, num_antenna_bs, subblocks_L]
                if ll == 0:
                    noise_bar_k_u = noise_bar_k_l_u
                else:
                    noise_bar_k_u = tf.concat([noise_bar_k_u, noise_bar_k_l_u], axis=2)

            Y_bar_k_u = tf.expand_dims((ybar_k_noiseless + noise_sqrt_up * noise_bar_k_u), axis=3)

            if uu == 0:
                Y_bar_k = Y_bar_k_u
            else:
                Y_bar_k = tf.concat([Y_bar_k, Y_bar_k_u], axis=3) # Y_bar_k: tensor [num_samples, num_antenna_bs, 1, 1]

        Y_bar_k = tf.squeeze(Y_bar_k, axis=3) # Y_bar_k: tensor(num_samples, num_antenna_bs, 1)

        '------------------------------------------ compute the online estimated overall low-dim channel g_MISO_k for user k -----------------------------------------------'
        # # for linalg.inv, The input is a tensor of shape [..., M, M] whose inner-most 2 dimensions form square matrices (allow Broadcasting!)
        Inv_C_yy_batch_UEk = tf.linalg.inv(C_yy_batch[:, :, :, user])
        Estimated_MISO_allUE[user] = mean_LowDim_overallChannal_batch[:, :, :, user] + tf.matmul(tf.matmul(C_yh_batch[:, :, :, user], Inv_C_yy_batch_UEk), (Y_bar_k - mean_Y_bar_batch[:, :, :, user]))

        # Compute the Estimated error and also, record the tensor-representation of the estimated channel
        Estimated_error_k = (tf.abs(tf.norm(tf.squeeze(Y_bar_k, axis=2) - tf.squeeze(ybar_k_noiseless, axis=2), ord=2, axis=1)))**2 / (tf.abs(tf.norm(tf.squeeze(Y_bar_k, axis=2), ord=2, axis=1)))**2
        if user == 0:
            mean_Estimated_error = tf.reduce_mean(Estimated_error_k)
            Estimated_MISO_allUE_Tensor = tf.expand_dims(Y_bar_k, axis=3)
        else:
            mean_Estimated_error = mean_Estimated_error + tf.reduce_mean(Estimated_error_k)
            Estimated_MISO_allUE_Tensor = tf.concat([Estimated_MISO_allUE_Tensor, tf.expand_dims(Y_bar_k, axis=3)], axis=3)

    mean_Estimated_error = mean_Estimated_error / num_user


    return Estimated_MISO_allUE, mean_Estimated_error, Estimated_MISO_allUE_Tensor




# Compute the uplink effective noise for each UE offline to avoid the huge noise matrix size causing the overflow in GPU memory!
# this will also be a place holder ---> this computation won't be included in the training process
def Uplink_effectiveNoise_offline_forLMMSE(params_system):
    'Notice, in order to facilitate the different batch size in training and testing, we always generate the noise\
    corresponding to the larger one. Then, when using this placeholder, just slicing the needed portion.\
    Notice: "noise_bar_allUE " here does not include the influence on UL SNR, we need to multiply the noise power later on'

    """
        :param: num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs: int
        :param: num_samples: usually = batch_size_test_1block

        :return: noise_bar_allUE: tensor [num_samples, batch_size_LMMSE_1block, num_antenna_bs, K]
    """

    (num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs) = params_system

    len_pilot_per_subblock = tau_w  # len_pilot_per_subblock: int
    'Generate orthogonal pilot sequences x_k at each sub-block'
    # pilot sequence is generated based on the DFT matrix!
    pilotseq_UE_all = DFT_matrix(len_pilot_per_subblock)  # notice, the pilot sequence will be sent repeatedly

    'Here we have double loop is for saving the CPU memory in constructing noise_k using np.random.normal'
    for sample in range(num_samples):
        # start_time = time.time()
        # the noise for each UE
        pilotseq_UE_all = pilotseq_UE_all[:, 0:num_user]  # [len_pilot_per_subblock, num_user]

        noise_k_real = np.random.normal(loc=0.0, scale=0.5, size=(batch_size_LMMSE_1block, num_antenna_bs, len_pilot_per_subblock))
        noise_k_imag = np.random.normal(loc=0.0, scale=0.5, size=(batch_size_LMMSE_1block, num_antenna_bs, len_pilot_per_subblock))
        noise_k = noise_k_real + 1j * noise_k_imag

        # Compute the uplink effective noise for each UE
        noise_bar_allUE_sample = (1 / len_pilot_per_subblock) * np.matmul(noise_k, pilotseq_UE_all)  # tensor [batch_size_LMMSE_1block, num_antenna_bs, num_user]
        noise_bar_allUE_sample = np.expand_dims(noise_bar_allUE_sample, axis=0)

        if sample == 0:
            noise_bar_allUE = noise_bar_allUE_sample
        else:
            noise_bar_allUE = np.concatenate((noise_bar_allUE, noise_bar_allUE_sample), axis=0)  # tensor [num_samples, batch_size_LMMSE_1block, num_antenna_bs, K]
        # print("Already training for %s seconds" % (time.time() - start_time))

    return noise_bar_allUE


def main_Uplink_effectiveNoise_offline_forLMMSE(num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs, isTesting=False):
    """
        :param:  isTesting: [True] save the generated {noise_bar_allUE} and reusing {noise_bar_allUE} in testing
                            [False] randomly generate {noise_bar_allUE} for all the blocks each time calling this function

        :return: noise_bar_allUE: tensor [num_samples, batch_size_LMMSE_1block, num_antenna_bs, K]
    """
    params_system = (num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs)
    if isTesting:
        if not os.path.exists('Rician_channel'):
            os.makedirs('Rician_channel')
        params_all = (num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs)
        file_test_ULnoise = './Rician_channel/ULNoise' + str(params_all) + '.mat'
        if os.path.exists(file_test_ULnoise):
            data_test = sio.loadmat(file_test_ULnoise)
            noise_bar_allUE = data_test['noise_bar_allUE']
        else:
            noise_bar_allUE = Uplink_effectiveNoise_offline_forLMMSE(params_system)

            sio.savemat(file_test_ULnoise, {'noise_bar_allUE': noise_bar_allUE})
    else:
        noise_bar_allUE = Uplink_effectiveNoise_offline_forLMMSE(params_system)

    return noise_bar_allUE


# ==================================================================================================================================
# ==================================================================================================================================
# This is for generating the needed Noise for LMMSE estimation in the uplink ----> save the .mat file offline to save training/testing time!
'# Main generate function!'
def main():
    num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs = 256, 10000, 1000, 3, 8 # 10, 10, 20, 3, 8

    params_all = (num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs)
    # generate testing channel for the BCD algorithm!
    file_path = './Rician_channel/ULNoise' + str(params_all) + '.mat'
    if not os.path.exists(file_path):
        noise_bar_allUE = main_Uplink_effectiveNoise_offline_forLMMSE(num_samples, batch_size_LMMSE_1block, tau_w, num_user, num_antenna_bs, isTesting=True)
    else:
        print('Testing noise data exists')


'# generate the channel data'
if __name__ == '__main__':
    main()















