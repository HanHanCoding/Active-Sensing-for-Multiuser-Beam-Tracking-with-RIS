import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras.layers import Dense
import time
import scipy.io as sio

from generate_channel_visualization import main_generate_channel
from util_fun import compute_sum_rate, compute_min_rate, compute_pattern_gain_UL, compute_pattern_gain_DL, compute_array_response_irs, compute_array_response_Beamformer_allUE


'sampling points on the range'
sample_num_x = 25
sample_num_y = 25
N_angles = 1000

'System information'
num_elements_irs = 100
num_antenna_bs = 8 #1
num_user = 3 #1
irs_Nh = 10
Rician_factor = 10
# enlarge this scale factor will never cause effect on the optimal rate
scale_factor = 100
N_G = 8 # strong paths in channel G!

frames = 40 # training frame number (equavalent to the LSTM cell number (including the 0th cell. i.e., frames = M+1 in ICASSP paper notation, M is the training RNN length))
frames_train = 21
# number of the sub_blocks per frame (uplink pilot phase)
subblocks_L = 3
# number of pilots each UE transmitted in each subblock
tau = num_user
# number of pilot sequence repeatedly send w.r.t. the same sensing vector
U_repeat = 1
# for each designed w, it has the potential to be used in multiple frames due to the slow change in the LOS part of the channel
# define the block number that a w could be used in is N_w_using
N_w_using = 3 # notice, here, we assume the channel reciprocity!
# since in training, we don't calculate the Minrate for frame 0 (the first LSTM cell)
total_blocks = N_w_using * frames
# feed channel dimension
total_blocks_inTesting = N_w_using

batch_size_test_1block = 1
params_system = (num_antenna_bs, num_elements_irs, num_user)
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


#############################################################################################
'Load the generated channel data'
# combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks)

# user_corrdinate_all: array [moving_frames, num_user, 3]
# range_coordinate_all_without_z: array [sample_num_x * sample_num_y, 1, 2]
# combined_A_range_all: array [sample_num_x * sample_num_y, 1, num_antenna_bs, (num_elements_irs + 1), 1, moving_frames]
# a_irs_bs_range_all: array [sample_num_x * sample_num_y, num_elements_irs, 1]
# a_irs_user_range_all: array [sample_num_x * sample_num_y, num_elements_irs, 1]

combined_A_test, user_corrdinate_all, range_coordinate_all_without_z, combined_A_range_all, a_irs_bs_range_all, a_irs_user_range_all = main_generate_channel(N_G, frames, N_w_using, sample_num_x, sample_num_y, num_antenna_bs,
                                                                                                                                                              num_elements_irs, num_user, irs_Nh, batch_size_test_1block,
                                                                                                                                                              Rician_factor, total_blocks,
                                                                                                                                                              scale_factor, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995,
                                                                                                                                                              location_bs=np.array([100, -100, 0]),
                                                                                                                                                              location_irs=np.array([0, 0, 0]), isTesting=True)

'we visualize {V,w,F} (notice, F means B in this paper)'
#############################################################################################
'Load the designed V'
file_path_V = './Rician_channel/Estimated_channel_gain' + str(params_system) + '_' + str(Rician_factor) + '.mat'
Designed_V_data = sio.loadmat(file_path_V)
V_collect_no_direct = Designed_V_data['V_collect_no_direct'] # (1, num_elements_irs, 1, subblocks_L, frames)


'Load the designed w and F'
file_path_w_F = './results_data/refined_w_B' + '_' + str(params_system) + '_' + str(Rician_factor) + '.mat'
Designed_w_F_data = sio.loadmat(file_path_w_F)

w_collect_input = Designed_w_F_data['DL_RIS_w'] # (frames, (N+1))
F_collect_input = Designed_w_F_data['DL_beamformer'] # (frames, M, K)

w_collect = np.empty([1, (num_elements_irs + 1), 1, frames], dtype=complex)
w_collect_no_direct = np.empty([1, num_elements_irs, 1, frames], dtype=complex)
F_collect = np.empty([1, num_antenna_bs, 1, num_user, frames], dtype=complex)

for frame in range(frames):
    w_collect[0, :, 0, frame] = w_collect_input[frame, :]
    w_collect_no_direct[0, :, 0, frame] = w_collect_input[frame, 1:(num_elements_irs + 1)]
    F_collect[0, :, 0, :, frame] = F_collect_input[frame, :, :]

###########################################################################################################################
'# compute the beam pattern gain in both UL/DL for the whole range'
# Since we have subblocks_L number of sensing vectors -----> Here, subblocks_L = 3
Array_response_V_range_all_1 = np.empty([sample_num_x * sample_num_y, 1, frames])
Array_response_V_range_all_2 = np.empty([sample_num_x * sample_num_y, 1, frames])
Array_response_V_range_all_3 = np.empty([sample_num_x * sample_num_y, 1, frames])

Array_response_w_range_all = np.empty([sample_num_x * sample_num_y, 1, frames])

Array_response_F = np.empty([N_angles, num_user, frames]) # for beamformer, no need to consider the whole moving range!

RIS_patternGain_DL_range_all = np.empty([sample_num_x * sample_num_y, 1, frames])

# combined_A_range_all: array [sample_num_x * sample_num_y, 1, num_antenna_bs, (num_elements_irs + 1), 1, moving_frames]
# "combined_A_range_all" is the channel w.r.t. the first blocks at each transmission frame
for frame in range(frames):
    'Array Response for the DL Beamformer'
    Array_response_F[:, :, frame] = compute_array_response_Beamformer_allUE(F_collect[0, :, :, :, frame], num_user, num_antenna_bs, N_angles)

    # for x coordinate on the moving range
    for jj in range(sample_num_x):
        # for y coordinate on the moving range
        for kk in range(sample_num_y):
            'Array Response for the UL RIS sensing vector'
            Array_response_V_range_all_1[sample_num_y * jj + kk, 0, frame] = compute_array_response_irs(a_irs_bs_range_all[sample_num_y * jj + kk, :, :], a_irs_user_range_all[sample_num_y * jj + kk, :, :], V_collect_no_direct[0, :, :, 0, frame])
            Array_response_V_range_all_2[sample_num_y * jj + kk, 0, frame] = compute_array_response_irs(a_irs_bs_range_all[sample_num_y * jj + kk, :, :], a_irs_user_range_all[sample_num_y * jj + kk, :, :], V_collect_no_direct[0, :, :, 1, frame])
            Array_response_V_range_all_3[sample_num_y * jj + kk, 0, frame] = compute_array_response_irs(a_irs_bs_range_all[sample_num_y * jj + kk, :, :], a_irs_user_range_all[sample_num_y * jj + kk, :, :], V_collect_no_direct[0, :, :, 2, frame])
            'Array Response for the DL RIS reflection coefficients'
            Array_response_w_range_all[sample_num_y * jj + kk, 0, frame] = compute_array_response_irs(a_irs_bs_range_all[sample_num_y * jj + kk, :, :], a_irs_user_range_all[sample_num_y * jj + kk, :, :], w_collect_no_direct[0, :, :, frame])
            'Beam pattern gain for the Combined BS and RIS response in downlink data transmission'
            RIS_patternGain_DL_range_all[sample_num_y * jj + kk, 0, frame] =compute_pattern_gain_DL(combined_A_range_all[sample_num_y * jj + kk, 0, :, :, 0, frame], w_collect[0, :, :, frame], F_collect[0, :, :, :, frame], num_user, num_antenna_bs)

path_Array_response_F = './Generate_MatFile_forVisualization/Array_response_F.mat'
sio.savemat(path_Array_response_F, {'Array_response_F': Array_response_F})

path_Array_response_V_range_all_1 = './Generate_MatFile_forVisualization/Array_response_V_range_all_1.mat'
sio.savemat(path_Array_response_V_range_all_1, {'Array_response_V_range_all_1': Array_response_V_range_all_1})
path_Array_response_V_range_all_2 = './Generate_MatFile_forVisualization/Array_response_V_range_all_2.mat'
sio.savemat(path_Array_response_V_range_all_2, {'Array_response_V_range_all_2': Array_response_V_range_all_2})
path_Array_response_V_range_all_3 = './Generate_MatFile_forVisualization/Array_response_V_range_all_3.mat'
sio.savemat(path_Array_response_V_range_all_3, {'Array_response_V_range_all_3': Array_response_V_range_all_3})

path_Array_response_w_range_all = './Generate_MatFile_forVisualization/Array_response_w_range_all.mat'
sio.savemat(path_Array_response_w_range_all, {'Array_response_w_range_all': Array_response_w_range_all})

path_RIS_patternGain_DL_range_all = './Generate_MatFile_forVisualization/RIS_patternGain_DL_range_all.mat'
sio.savemat(path_RIS_patternGain_DL_range_all, {'RIS_patternGain_DL_range_all': RIS_patternGain_DL_range_all})

###########################################################################################################################
# user_corrdinate_all: array [moving_frames, num_user, 3]
'get the x and y axis of the users coordinate for the selected blocks'
user_coor_x = np.zeros([frames, 1, num_user])
user_coor_y = np.zeros([frames, 1, num_user])

for frame in range(frames):
    user_coor_x[frame, 0, :] = user_corrdinate_all[frame, :, 0]
    user_coor_y[frame, 0, :] = user_corrdinate_all[frame, :, 1]

# save the user coordinate change (this will be used as the center (reference point) to the rate visualization graph)
path_user_coor_x = './Generate_MatFile_forVisualization/user_coor_x.mat'
path_user_coor_y = './Generate_MatFile_forVisualization/user_coor_y.mat'
sio.savemat(path_user_coor_x, {'user_coor_x': user_coor_x})
sio.savemat(path_user_coor_y, {'user_coor_y': user_coor_y})


#########################################################################
'get the x and y axis of the testing range [same for all the blocks]'
range_coordinate_x_all = np.zeros([sample_num_x * sample_num_y, 1])
range_coordinate_y_all = np.zeros([sample_num_x * sample_num_y, 1])

range_coordinate_x = np.zeros([sample_num_x, 1])
range_coordinate_y = range_coordinate_y_all[0:sample_num_y, :]

for ii in range(sample_num_x * sample_num_y):
    range_coordinate_x_all[ii, 0] = range_coordinate_all_without_z[ii, 0, 0]
    range_coordinate_y_all[ii, 0] = range_coordinate_all_without_z[ii, 0, 1]

for kk in range(sample_num_x):
  range_coordinate_x[kk, 0] = range_coordinate_x_all[kk * sample_num_y, 0]


path_range_coordinate_x = './Generate_MatFile_forVisualization/range_coordinate_x.mat'
path_range_coordinate_y = './Generate_MatFile_forVisualization/range_coordinate_y.mat'
sio.savemat(path_range_coordinate_x, {'range_coordinate_x': range_coordinate_x})
sio.savemat(path_range_coordinate_y, {'range_coordinate_y': range_coordinate_y})








