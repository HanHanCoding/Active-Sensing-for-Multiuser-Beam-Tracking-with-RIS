import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


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


























