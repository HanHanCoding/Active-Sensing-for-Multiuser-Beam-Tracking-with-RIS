import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras.layers import Dense
import time
import scipy.io as sio

from generate_channel import main_generate_channel_forTesting_and_Refining, main_generate_channel
from Designed_Neural_Layers import MLP_input_w,MLP_input_F,MLPBlock0,MLPBlock1,MLPBlock2,MLPBlock3,MLPBlock4,GNN_layer,LSTM_model
from Pilots_Generation import Contribution_receivedpilots, LS_Estimated_LowDim_Channel
from util_fun import compute_sum_rate, compute_min_rate, Generate_Beamformer
from TestingPhase_GNN_output import GNN_output_proposed_approach

'System information'
num_elements_irs = 100
num_antenna_bs = 8 #1
num_user = 3 #1
irs_Nh = 10 # Number of elements on the x-axis of rectangle RIS
Rician_factor = 10
# enlarge this scale factor will never cause effect on the optimal rate
scale_factor = 100
N_G = 8 # strong paths in channel G!

frames = 30 # training frame number (equavalent to the LSTM cell number (including the 0th cell. i.e., frames = M+1 in ICASSP paper notation, M is the training RNN length))
frames_train = 21
# number of the sub_blocks per frame (uplink pilot phase)
subblocks_L = 1
# number of pilots each UE transmitted in each subblock
tau_train = 3
tau = num_user
# number of pilot sequence repeatedly send w.r.t. the same sensing vector
U_repeat = 3
# for each designed w, it has the potential to be used in multiple frames due to the slow change in the LOS part of the channel
# define the block number that a w could be used in is N_w_using
N_w_using = 3 # notice, here, we assume the channel reciprocity!
# since in training, we don't calculate the Minrate for frame 0 (the first LSTM cell)
total_blocks = N_w_using * frames
# feed channel dimension
total_blocks_inTesting = N_w_using

#####################################################
'Compute the pilot overhead and its effect on the DL sum rate'
T_c = 5e-3
T_s = 2e-6
pilots_duration_perFrame = tau * subblocks_L * T_s
frame_duration = N_w_using * T_c + pilots_duration_perFrame
Coefficient_pilot_overhead = (1 - pilots_duration_perFrame/frame_duration)

#####################################################
'Channel Information'
# uplink transmitting power and noise
# transmitting power and noise power in dBm
Pt_up = 5 #dBm
# we want to enlarge this term to make the network hard to estimate
noise_power_db_up = -70 #dBm
noise_power_linear_up = 10 ** ((noise_power_db_up - Pt_up + scale_factor) / 10)
noise_sqrt_up = np.sqrt(noise_power_linear_up)
SNR_test_indB_up = Pt_up - noise_power_db_up - scale_factor
print('raw SNR in uplink (dB):', SNR_test_indB_up)

# downlink transmitting power and noise
Pt_down = 10 #dBm
noise_power_db_down = -90 #dBm
SNR_test_indB_down = Pt_down - noise_power_db_down - scale_factor
Pt_down_linear = 10 ** (SNR_test_indB_down / 10)
Pt_sqrt_down = np.sqrt(Pt_down_linear)
# modify this noise power will cause effect on the optimal rate and the worst rate
noise_power_linear_down = 1
noise_sqrt_down = np.sqrt(noise_power_linear_down)
print('raw SNR in downlink (dB):', SNR_test_indB_down)

#####################################################
'Learning Parameters'
initial_run = 0  # 0: Continue training; 1: Starts from the scratch
n_epochs = 70  # Num of epochs
learning_rate = 0.0001 # 0.0001  # Learning rate
batch_per_epoch = 100  # Number of mini batches per epoch
batch_size_test_1block = 2000 # in testing case in a pretrained model, it should be 10000
batch_size_train_1block = batch_size_test_1block

#####################################################
# for the LS estimator
tau_w = 1000 # pilot length for LS

######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
# notice! though the generated 'combined_A' is complex128 type, but the placeholder here force its type becoming complex 64!
channel_combined_input = tf.placeholder(tf.complex64, shape=(None, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks_inTesting), name="channel_combined_input")
# channel_combined_input_forLMMSE = tf.placeholder(tf.complex64, shape=(None, num_antenna_bs, (num_elements_irs + 1), num_user), name="channel_combined_input_forLMMSE")
# noise_bar_input_forLMMSE = tf.placeholder(tf.complex64, shape=(None, batch_size_LMMSE_1block, num_antenna_bs, num_user), name="noise_bar_input_forLMMSE")

# in benchmark, the batch_size = batch_size_test_1frame * frames
# here, different from the benchmark setting
batch_size = tf.shape(channel_combined_input)[0]

hidden_size_LSTM = 512
previous_hidden_state = tf.placeholder(tf.float32, shape=(None, hidden_size_LSTM, num_user), name="previous_hidden_state")
previous_cell_state = tf.placeholder(tf.float32, shape=(None, hidden_size_LSTM, num_user), name="previous_cell_state")
V_subblocks_all_current = tf.placeholder(tf.complex64, shape=(None, (num_elements_irs + 1), subblocks_L), name="V_subblocks_all_current")

##########################################################################################################
with tf.name_scope("channel_sensing"):
    '""""""""""""""""""""""""""""Define the Neural Network layers"""""""""""""""""""""""""""'
    'RNN layers (LSTM variation)' # three different LSTM for three different users
    LSTM_user = LSTM_model()

    'GNN layers for {w, F}'
    'The GNN has two layers for spatial convolution'
    # first layer
    MLP0_1, MLP1_1, MLP2_1, MLP3_1, MLP4_1 = MLPBlock0(), MLPBlock1(), MLPBlock2(), MLPBlock3(), MLPBlock4()
    # second layer
    MLP0_2, MLP1_2, MLP2_2, MLP3_2, MLP4_2 = MLPBlock0(), MLPBlock1(), MLPBlock2(), MLPBlock3(), MLPBlock4()
    'The layer to generate the INPUT representative vector for the user node and RIS node!'
    MLP_w_in, MLP_F_in = MLP_input_w(), MLP_input_F()

    'DNN for calculating the DL RIS reflection coefficients w for each frame'
    # DNN_w_mapping = DNN_mapping_w()
    DNN_w_mapping_output = Dense(units=2 * num_elements_irs, activation='linear')  # the linear(ouput) layer of DNN_w_mapping

    'Linear Layers for outputting F'
    # here, we want the GNN only output the power allocation for the beamformers (since we embedded the beamformer's structure into the solution!)
    F_output = Dense(units=2, activation='linear')
    #######################################################################################################################

    '""""""""""""""""""""""""""""Initialization"""""""""""""""""""""""""""'
    # The dimension of the input pilots after vectorization.
    width_input_Y_k = 2 * num_antenna_bs * subblocks_L * U_repeat  # here, 2 means y has the real and imag part
    # since 'Tensor' object does not support item assignment, we create a list for hidden and cell state w.r.t. all UEs
    h_UE_all_init = []
    cell_UE_all_init = []
    # 'the proposed structure'
    for user in range(num_user):
        y_real_init = tf.ones([batch_size, width_input_Y_k])
        #Here we first attempt a LSTM network that is identical for three different users.
        h_old_init, cell_old_init = LSTM_user(y_real_init, previous_hidden_state[:, :, user], previous_cell_state[:, :, user])
        # 'Initialize the hidden/cell states for the K users separately'
        h_UE_all_init.append(h_old_init) # list[(batch_size, hidden_size_LSTM),...], num_user items
        cell_UE_all_init.append(cell_old_init)  # list[(batch_size, hidden_size_LSTM),...], num_user items
        'fetch the updated initial hidden/cell state'
        h_old_init_fetch = tf.reshape(h_old_init, [batch_size, hidden_size_LSTM, 1])
        cell_old_init_fetch = tf.reshape(cell_old_init, [batch_size, hidden_size_LSTM, 1])
        if user == 0:
            h_old_init_fetch_all = h_old_init_fetch
            cell_old_init_fetch_all = cell_old_init_fetch
        else:
            h_old_init_fetch_all = tf.concat([h_old_init_fetch_all, h_old_init_fetch], axis=2)
            cell_old_init_fetch_all = tf.concat([cell_old_init_fetch_all, cell_old_init_fetch], axis=2)

    'fetch the initial V_subblocks_all'
    V_subblocks_all_init, w_her_complex_init, F_complex_init, _, _ = GNN_output_proposed_approach(cell_UE_all_init,
                                                                                               MLP_w_in, MLP_F_in,
                                                                                               MLP0_1, MLP1_1, MLP2_1, MLP3_1, MLP4_1,
                                                                                               MLP0_2, MLP1_2, MLP2_2, MLP3_2, MLP4_2,
                                                                                               F_output, DNN_w_mapping_output,
                                                                                               num_user, num_elements_irs, batch_size, num_antenna_bs, Pt_down_linear,
                                                                                               tau_w, noise_sqrt_up, noise_power_linear_down, channel_combined_input[:, :, :, :, 0])



    '""""""""""""""""""""""""""""After initialization!"""""""""""""""""""""""""""'
    '""""""""""""""""""""""""""""RNN (LSTM) update its hidden state for each UE"""""""""""""""""""""""""""'
    'AP receiving pilots and decorrlate the contribution based on designed V at each UL phase'
    h_UE_all = []
    cell_UE_all = []
    # Ybar_all: list[tensor(batch_size, num_antenna_bs, subblocks_L, U_repeat),..., tensor(batch_size, num_antenna_bs, subblocks_L, U_repeat)], in total num_user items, dtype=complex64
    Ybar_all = Contribution_receivedpilots(batch_size, num_user, num_antenna_bs, subblocks_L, U_repeat, channel_combined_input[:, :, :, :, 0], V_subblocks_all_current, noise_sqrt_up)
    'Update hidden state based on the newly received pilots'
    for user in range(num_user):
        Ybar_all_k = tf.reshape(Ybar_all[user], [batch_size, num_antenna_bs * subblocks_L * U_repeat])
        # separate the real and imag part
        Ybar_all_vectorization_k = tf.concat([tf.real(Ybar_all_k), tf.imag(Ybar_all_k)], axis=1) #Ybar_all_vectorization_k: tensor(batch_size, 2 * num_antenna_bs * subblocks_L)
        # 'Update hidden state based on the newly received pilots'
        h_UE_all_k, cell_UE_all_k = LSTM_user(Ybar_all_vectorization_k, previous_hidden_state[:, :, user], previous_cell_state[:, :, user])

        # 'Initialize the hidden/cell states for the K users separately'
        h_UE_all.append(h_UE_all_k)  # list[(batch_size, hidden_size_LSTM),...], num_user items
        cell_UE_all.append(cell_UE_all_k)  # list[(batch_size, hidden_size_LSTM),...], num_user items
        'fetch the updated hidden/cell state'
        h_old_fetch = tf.reshape(h_UE_all_k, [batch_size, hidden_size_LSTM, 1])
        cell_old_fetch = tf.reshape(cell_UE_all_k, [batch_size, hidden_size_LSTM, 1])
        if user == 0:
            h_old_fetch_all = h_old_fetch
            cell_old_fetch_all = cell_old_fetch
        else:
            h_old_fetch_all = tf.concat([h_old_fetch_all, h_old_fetch], axis=2)
            cell_old_fetch_all = tf.concat([cell_old_fetch_all, cell_old_fetch], axis=2)

    'fetch the V_subblocks_all'
    V_subblocks_all, w_her_complex, F_complex, mean_Estimated_error, Estimated_MISO_allUE_Tensor = GNN_output_proposed_approach(cell_UE_all,
                                                                                                                               MLP_w_in, MLP_F_in,
                                                                                                                               MLP0_1, MLP1_1, MLP2_1, MLP3_1, MLP4_1,
                                                                                                                               MLP0_2, MLP1_2, MLP2_2, MLP3_2, MLP4_2,
                                                                                                                               F_output, DNN_w_mapping_output,
                                                                                                                               num_user, num_elements_irs, batch_size, num_antenna_bs, Pt_down_linear,
                                                                                                                               tau_w, noise_sqrt_up, noise_power_linear_down, channel_combined_input[:, :, :, :, 0])

    F_complex_testing = tf.squeeze(F_complex)
    # Things to be fetched in testing phase at frame > 0:
    # h_old_fetch_all: tensor(batch_size, hidden_size_LSTM, num_user), dtype: float32
    # cell_old_fetch_all: tensor(batch_size, hidden_size_LSTM, num_user), dtype: float32
    # V_subblocks_all: tensor(batch_size, (num_elements_irs + 1), subblocks_L), dtype: Complex64

    '"""""""""""""Calculate the downlink sum rate for this frame based on designed {F, w}""""""""""""""""""""'
    # Notice, w and F are used in each frame. i.e., consecutive N blocks!
    'Attention! we dont use F^(0) and w^(0) to compute rate (followed by the frame structure.)'
    Minrate_perframe, Minrate_count = compute_min_rate(channel_combined_input, w_her_complex, F_complex, batch_size, num_user, num_antenna_bs, N_w_using, noise_power_linear_down)


##################################################################################################################
'Loss Function'
loss = -Minrate_perframe
#loss = -1 * tf.reduce_mean(received_power)
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#############################################################################################  testing set (validation)
'we can use this part to generate the testing set!'
# cascaded_A: array (num_samples, num_antenna_bs, num_elements_irs, num_user, total_blocks), dtype=complex128
combined_A_test = main_generate_channel_forTesting_and_Refining(N_G, num_antenna_bs, num_elements_irs, num_user, irs_Nh,
                                                         batch_size_test_1block, Rician_factor, total_blocks, scale_factor,
                                                         rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
                                                         location_irs=np.array([0, 0, 0]), isTesting=True)

###########################################################################################################################
'testing the trained performance'
Minrate_all = []

# these are for sensing phase II, we use designed w^(t) as the sensing vector to estimate the low-dim channel
w_noDirect_all = np.empty([batch_size_test_1block, num_elements_irs, 1, frames], dtype=complex)
w_withDirect_all = np.empty([batch_size_test_1block, (num_elements_irs+1), 1, frames], dtype=complex) # this w is used to multiply the estimated channel at the output stage
F_all = np.empty([batch_size_test_1block, num_antenna_bs, num_user, frames], dtype=complex)
H_lmmse_designed_w = np.empty([batch_size_test_1block, num_antenna_bs, 1, num_user, frames], dtype=complex)

for frame in range(frames):
    'initialize hidden state and the sensing vectors to be feed -- get h_fetch and V_fetch'
    # previous_hidden_state = tf.placeholder(tf.float32, shape=(None, hidden_size_LSTM, num_user), name="previous_hidden_state")
    # V_subblocks_all_current = tf.placeholder(tf.complex64, shape=(None, (num_elements_irs + 1), subblocks_L), name="V_subblocks_all_current")
    if frame == 0:
        h_init = np.zeros([batch_size_test_1block, hidden_size_LSTM, num_user], dtype=float)
        cell_init = np.zeros([batch_size_test_1block, hidden_size_LSTM, num_user], dtype=float)
        # here, this V is just used to run the program, with no true meaning!!
        # V_init = np.ones([batch_size_test_1block, (num_elements_irs + 1), subblocks_L], dtype=complex)
        'Notice, V_init is not necessary to be fed into the architecture, since we do not fetch the second GNN output. We only need to feed the placeholder to compute the needed output'
        'However, channel_combined_input is necessary to be fed, since batch_size is a function of its dimensions'
        feed_dict_val = {channel_combined_input: combined_A_test[:, :, :, :, frame * N_w_using:(frame + 1) * N_w_using], previous_hidden_state: h_init, previous_cell_state: cell_init}

        snr_temp = SNR_test_indB_up
        with tf.Session() as sess:
            if initial_run == 1:
                init.run()
            else:
                saver.restore(sess,'./params/addaptive_reflection_snr_' + str(int(snr_temp)) + '_' + str(tau_train) + '_' + str(frames_train))

            h_fetch, cell_fetch, V_fetch = sess.run([h_old_init_fetch_all, cell_old_init_fetch_all, V_subblocks_all_init], feed_dict=feed_dict_val)

    # here, we won't feed h_val frame per frame, but directly feed it as a whole
    feed_dict_val = {channel_combined_input: combined_A_test[:, :, :, :, frame * N_w_using:(frame + 1) * N_w_using], previous_hidden_state: h_fetch, previous_cell_state: cell_fetch, V_subblocks_all_current: V_fetch}

    #########################################################################
    ###########  Training:
    # Recording the uplink SNR
    snr_temp = SNR_test_indB_up
    with tf.Session() as sess:
        if initial_run == 1:
            init.run()
        else:
            saver.restore(sess, './params/addaptive_reflection_snr_'+str(int(snr_temp))+'_'+str(tau_train) + '_' + str(frames_train))

        'Notice! Now, the posterior_dict is not used in visulization and results performing!'
        best_loss, Minrate_block_test, h_fetch, cell_fetch, V_fetch, w_fetch, F_fetch, Estimated_MISO_fetch = sess.run([loss, Minrate_count, h_old_fetch_all, cell_old_fetch_all, V_subblocks_all, w_her_complex, F_complex_testing, Estimated_MISO_allUE_Tensor], feed_dict=feed_dict_val)
        Minrate_all.append(Minrate_block_test)

        # # these are for sensing phase II, we use designed w^(t) as the sensing vector to estimate the low-dim channel
        w_noDirect_all[:, :, :, frame] = w_fetch[:, 1:(num_elements_irs + 1), :] # w_fetch : tensor[batch_size, (num_elements_irs + 1), 1], dtype: Complex64
        w_withDirect_all[:, :, :, frame] = w_fetch
        F_all[:, :, :, frame] = F_fetch
        H_lmmse_designed_w[:, :, :, :, frame] = Estimated_MISO_fetch


###########################################################################################################################
# Show/Record the computed min use rate without estimating low-dim channel and re-design beamformer F.
Proposed_Trained_Minrate_all_2 = np.array(Minrate_all).reshape([frames * N_w_using, 1])

# save the data
print('Proposed_trained_rate_perframe_2',Proposed_Trained_Minrate_all_2)

# save the recorded data
file_path_Proposed_trained_Min_rate_all_2 = 'Proposed_Trained_Minrate_all_2.mat'
sio.savemat(file_path_Proposed_trained_Min_rate_all_2, {'Proposed_Trained_Minrate_all_2': Proposed_Trained_Minrate_all_2})


###########################################################################################################################
'Save the estimated channel (the channel gain computed together with the designed DL RIS reflection coefficients)'
# we also save the designed w_noDirect_all to compute the downlink rate w.r.t. the perfect CSI
params_system = (num_antenna_bs, num_elements_irs, num_user)
file_path = './Rician_channel/Estimated_channel_gain' + str(params_system) + '_' + str(Rician_factor) + '.mat'

# H_lmmse_designed_w = [batch_size_test_1block, num_antenna_bs, 1, num_user, frames]
# w_noDirect_all = [batch_size_test_1block, num_elements_irs, 1, frames]
raw_SNR_DL = float(Pt_down - noise_power_db_down - scale_factor)
sio.savemat(file_path,
            {'H_lmmse_combine_designed_theta': H_lmmse_designed_w,
             'theta_noDirect_all': w_noDirect_all,
             'W_GNN_only': F_all,
             'raw_SNR_DL': raw_SNR_DL})


























