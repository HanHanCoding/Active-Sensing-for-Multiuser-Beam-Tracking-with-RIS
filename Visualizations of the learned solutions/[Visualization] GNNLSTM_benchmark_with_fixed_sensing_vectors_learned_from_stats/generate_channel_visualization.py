import numpy as np
import scipy.io as sio
from scipy.linalg import dft
import os
import time

'This code is for generating channels according to the Rician Model. '
'for generate_channel_{G, hr}, we directly generate channels in all the total_blocks'

'generate the initial coordinate and moving directions for the mobile UEs'
def initial_moving_states_visualizing_traj(num_user):
    """
        :intermediate variables: Coordinate_direction_all: array (num_samples, num_user, 4)

        ============ Notice, in visualizing the trajectory, the num_sample = 1! ================
        :return: coordinate_UE_init: (1, num_user, 3)
                 init_direction_state: (1, num_user)
    """
    # By default, we assume there only be 3 UEs moving in the range
    Coordinate_direction_all = np.empty([1, num_user, 4]) # here, 1 means the num_sample

    'UE 1 -- upper left'
    Coordinate_direction_all[0, 0, 0] = 15  # x-axis init
    Coordinate_direction_all[0, 0, 1] = -10  # y-axis init
    Coordinate_direction_all[0, 0, 2] = -20.0 # z-axis init
    Coordinate_direction_all[0, 0, 3] = np.radians(45) # initial moving direction

    'UE 2 -- upper right'
    Coordinate_direction_all[0, 1, 0] = 15  # x-axis init
    Coordinate_direction_all[0, 1, 1] = 10  # y-axis init
    Coordinate_direction_all[0, 1, 2] = -20.0  # z-axis init
    Coordinate_direction_all[0, 1, 3] = np.radians(150)  # initial moving direction

    'UE 3 -- lower middle'
    Coordinate_direction_all[0, 2, 0] = 30  # x-axis init
    Coordinate_direction_all[0, 2, 1] = -5.0  # y-axis init
    Coordinate_direction_all[0, 2, 2] = -20.0  # z-axis init
    Coordinate_direction_all[0, 2, 3] = np.radians(330)  # initial moving direction


    'Record the generated initial moving states'
    coordinate_UE_init = Coordinate_direction_all[:, :, 0:3]
    init_direction_state = Coordinate_direction_all[:, :, 3]

    return coordinate_UE_init, init_direction_state


'update the coordinate and moving directions for the mobile UEs'
'We check whether the user hit the wall or not ---> make sure that the user is moving inside the rectangular area like a cleaning robot'
def generate_location(prev_direction_state, coordinate_UE_prev, num_samples, num_user):
    """
    :param: num_samples: int
            num_user: int
            prev_direction_state: array (num_samples,num_user)
            coordinate_UE_prev: array (num_samples,num_user,3)

    :return:curr_direction_state: array (num_samples,num_user)
            coordinate_UE_curr: array (num_samples,num_user,3)
    """

    'update the direction state'
    angle_increment = np.random.uniform(-1*np.deg2rad(10), np.deg2rad(10), size=[num_samples, num_user])
    curr_direction_state = prev_direction_state + angle_increment

    'find the current UE coordinate'
    coordinate_UE_curr = np.empty([num_samples, num_user, 3])
    # define the moving distance of the UE at each block i.e., 8m/s ~ 10m/s
    distance = 0.15
    # x-coordinate
    coordinate_UE_curr[:, :, 0] = coordinate_UE_prev[:, :, 0] + distance * np.sin(curr_direction_state)
    # y-coordinate
    coordinate_UE_curr[:, :, 1] = coordinate_UE_prev[:, :, 1] + distance * np.cos(curr_direction_state)
    # z-coordinate
    coordinate_UE_curr[:, :, 2] = coordinate_UE_prev[:, :, 2]

    # check if any UE has hit the wall
    for i in range(num_samples):
        for j in range(num_user):
            # for x-coordinate
            if coordinate_UE_curr[i, j, 0] > 45.0:
                coordinate_UE_curr[i, j, 0] = 45.0 - (coordinate_UE_curr[i, j, 0] - 45.0)
                curr_direction_state[i, j] = 2*np.pi - curr_direction_state[i, j]
            elif coordinate_UE_curr[i, j, 0] < 5.0:
                coordinate_UE_curr[i, j, 0] = 5.0 + (5.0 - coordinate_UE_curr[i, j, 0])
                curr_direction_state[i, j] = 2*np.pi - curr_direction_state[i, j]
            # for y-coordinate
            if coordinate_UE_curr[i, j, 1] > 35.0:
                coordinate_UE_curr[i, j, 1] = 35.0 - (coordinate_UE_curr[i, j, 1] - 35.0)
                curr_direction_state[i, j] = np.pi - curr_direction_state[i, j]
            elif coordinate_UE_curr[i, j, 1] < -35.0:
                coordinate_UE_curr[i, j, 1] = -35.0 + (-35.0 - coordinate_UE_curr[i, j, 1])
                curr_direction_state[i, j] = np.pi - curr_direction_state[i, j]

    return curr_direction_state, coordinate_UE_curr

def path_loss_r(d):
    loss = 28.0 + 16.9 * np.log10(d)
    return loss


def path_loss_d(d):  # this is for the BS-UE link
    loss = 32.0 + 43.3 * np.log10(d)
    return loss


def generate_pathloss_aoa_aod(location_user, location_bs, location_irs):
    """
    :param location_user: array (num_samples,num_user,3)
    :param location_bs: array (3,)
    :param location_irs: array (3,)

    :return:pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            pathloss_irs_bs: float
            pathloss_irs_user: array (num_samples,num_user)
            pathloss_bs_user: array (num_samples,num_user)

            上行信道steering vector建模
            aoa_bs, aod_irs_y, aod_irs_z: float
            aoa_irs_y, aoa_irs_z: array (num_samples,num_user)

    """
    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs)
    pathloss_irs_bs = path_loss_r(d0)
    aoa_bs = (location_irs[0] - location_bs[0]) / d0
    aod_irs_y = (location_bs[1] - location_irs[1]) / d0
    aod_irs_z = (location_bs[2] - location_irs[2]) / d0 # In our geometry, this will always be 0!

    # =========irs-user=============
    # make use of the broadcasting rule!
    dist = np.linalg.norm((location_user - location_irs), axis=2)
    pathloss_irs_user = path_loss_r(dist) # array: (num_samples, num_user)
    aoa_irs_y = (location_user[:, :, 1] - location_irs[1]) / dist
    aoa_irs_z = (location_user[:, :, 2] - location_irs[2]) / dist

    # =========bs-user=============
    'notice, h_d is of i.i.d Raylaigh （Guassian）fading'
    # make use of the broadcasting rule!
    d_k = np.linalg.norm((location_user - location_bs), axis=2)
    pathloss_bs_user = path_loss_d(d_k)

    # pathloss = (pathloss_irs_bs, np.array(pathloss_irs_user), np.array(pathloss_bs_user))
    pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
    aoa_aod = (aoa_bs, aoa_irs_y, aoa_irs_z)
    return pathloss, aoa_aod


# channel between IRS and BS
'In this program, to ensure that the Channel G supports the spatial multiplexing, we assume that G will have multiple paths - not necessarily the LOS path!!'
def generate_G_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, location_user, num_samples, N_G):
    """
        :param: N_G: integer ---> # of LOS links in the model of channel G
        :param: location_user: array (num_samples,num_user,3)
        :param: location_bs: array (3,)
        :param: location_irs: array (3,)

        [上行信道steering vector建模]
        aod_irs_y_highRank, aoa_bs_highRank: array (num_samples, N_G)

        :return: los_irs_bs: array (num_antenna_bs, num_elements_irs)
                 pathloss_irs_bs: float
                 aod_irs_y_highRank: array (num_samples, N_G)
    """
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    'find the path losses'
    pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
    (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
    (aoa_bs, aoa_irs_y, aoa_irs_z) = aoa_aod

    pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
    pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

    'compute the LOS part ------> Assuming that there only be 1 LOS link between AP and UE, so that the AOA and AOD (\phi_1 and \phi_2) are fixed by the location of the AP and the RIS'
    # a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
    # a_bs = np.reshape(a_bs, [num_antenna_bs, 1])
    #
    # i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
    # i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)
    # a_irs_bs = np.exp(1j * np.pi * (i1 * aod_irs_y + i2 * aod_irs_z))
    # a_irs_bs = np.reshape(a_irs_bs, [num_elements_irs, 1])
    # los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate()) #'for all the samples (moving instance), the LOS part of channel G remains unchanged. '

    'compute the LOS part -------->  Here, Not like the regular Rician fading channel that only has one LOS link, we assume that there is in total N_G paths'
    # here, we first generate the \phi_1 and \phi_2 different from the above model
    phi_1 = np.random.uniform(0, 2*np.pi, size=[num_samples, N_G]) # AOA at AP in the Uplink model
    phi_2 = np.random.uniform(-1 * 0.5*np.pi, 0, size=[num_samples, N_G]) # AOD from RIS in the Uplink model
    # we then generate the corresponding AOAs and AODs --------------> Notice that aod_irs_z always be 0 since the z-axis of RIS and AP are the same. So we ignore this term!
    aod_irs_y_highRank = np.sin(phi_2) # since cos(\theta_2) = 1
    aoa_bs_highRank = np.cos(phi_1) # since cos(\theta_1) = 1

    i1 = np.expand_dims(np.mod(np.arange(num_elements_irs), irs_Nh), axis=0) # i1: (1, num_elements_irs)
    # i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)
    # Compute the steering vector for the ii-th LOS link, and sum them together!
    for ii in range(N_G):
        a_bs_ii = np.expand_dims(np.exp(1j * np.pi * np.reshape(aoa_bs_highRank[:, ii], [-1, 1]) * np.reshape(np.arange(num_antenna_bs), [1, -1])), axis=2) # a_bs_ii: (num_samples, num_antenna_bs, 1)
        # since sin(\theta_2) = aod_irs_z = 0
        a_irs_bs_ii = np.expand_dims(np.exp(1j * np.pi * (np.reshape(aod_irs_y_highRank[:, ii], [-1, 1]) * i1)), axis=1) # a_irs_bs_ii: (num_samples, 1, num_elements_irs)
        # Apply the broadcasting role!
        los_irs_bs_ii = a_bs_ii @ a_irs_bs_ii.conjugate()
        # Sum over all the Paths
        if ii == 0:
            los_irs_bs = los_irs_bs_ii
        else:
            los_irs_bs = los_irs_bs + los_irs_bs_ii

    return los_irs_bs, pathloss_irs_bs, aod_irs_y_highRank


# LOS channel between IRS and user
# pathloss of the channel hd
def generate_hrLOS_hdPathloss(params_system, irs_Nh, location_bs, location_irs, scale_factor, location_user, num_samples):
    """
        :param: pathloss_irs_user: array (num_samples,num_user)
                pathloss_bs_user: array (num_samples,num_user)
                aoa_irs_y, aoa_irs_z: array (num_samples,num_user)
                location_user: array (num_samples,num_user,3)
                location_irs: array (3,)

        :return: los_irs_user: array (num_samples, num_elements_irs, num_user)
                 pathloss_irs_user: array (num_samples, num_user)
                 pathloss_bs_user: array (num_samples, num_user)
    """
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    'find the path loss'
    pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
    (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
    (aoa_bs, aoa_irs_y, aoa_irs_z) = aoa_aod
    # hr
    pathloss_irs_user = pathloss_irs_user - scale_factor / 2
    pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
    # hd
    pathloss_bs_user = pathloss_bs_user - scale_factor
    pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
    # pathloss_bs_user = np.zeros([num_samples, num_user])

    'compute the LOS part of hr'
    i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
    i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)

    los_irs_user = np.empty([num_samples, num_elements_irs, num_user], dtype=complex)
    for element in range(num_elements_irs):
        los_irs_user[:, element, :] = np.exp(1j * np.pi * (i1[element] * aoa_irs_y + i2[element] * aoa_irs_z))

    return los_irs_user, pathloss_irs_user, pathloss_bs_user

# this is for plotting the array response of the RIS in the UL
def generate_spatial_signature(params_system, location_user, location_bs, location_irs, irs_Nh, aod_irs_y_highRank, N_G):
    """
    :params: 上行信道steering vector建模
            aoa_bs: float (In fact, should be [1,N_G], but here we don't need this parameter so ignore this small mistake :-))
            aoa_irs_y, aoa_irs_z: array (num_samples, num_user) ----> (1,1) here (The reason is that we treat each small block on the moving range as a virtual 'user', in total 25*25 blocks)
            aod_irs_y_highRank: array (num_samples, N_G) ---------> (1,N_G) here

    :return: a_irs_bs: array (num_elements_irs, 1)
             a_irs_user: array (num_elements_irs, 1)
    """
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    # find the aoa_aod
    pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)

    (aoa_bs, aoa_irs_y, aoa_irs_z) = aoa_aod
    'compute the array response'
    i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
    i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)

    # Compute the steering vector for the only path between RIS-UE
    a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y + i2 * aoa_irs_z))

    # Compute the steering vector for the ii-th path between RIS-AP, and sum them together!
    for ii in range(N_G):
        a_irs_bs_ii = np.exp(1j * np.pi * (np.reshape(aod_irs_y_highRank[:, ii], [-1, 1]) * i1)) # a_irs_bs_ii: (num_samples, num_elements_irs) ---------> Here, (1, num_elements_irs)
        # Sum over all the Paths
        if ii == 0:
            a_irs_bs = a_irs_bs_ii
        else:
            a_irs_bs = a_irs_bs + a_irs_bs_ii

    return a_irs_bs.reshape((-1, 1)), a_irs_user.reshape((-1, 1))


# generate the cascaded channel A_k for all the users (w.r.t. all the blocks!)
def generate_channel(moving_frames, N_w_using, sample_num_x, sample_num_y, params_system, irs_Nh, total_blocks, location_bs, location_irs, Rician_factor, scale_factor, num_samples, rho_G, rho_hr, rho_hd, N_G):
    """
        :param  los_irs_bs (G_LOS): array (num_antenna_bs, num_elements_irs)
                pathloss_irs_bs: float

                los_irs_user: array (num_samples, num_elements_irs, num_user)
                pathloss_irs_user: array (num_samples, num_user)

                pathloss_bs_user: array (num_samples, num_user)

        :intermediate variables: channel_G: array (num_samples, num_antenna_bs, num_elements_irs, total_blocks)
                                 channel_hr: array (num_samples, num_elements_irs, num_user, total_blocks)
                                 channel_hd: array (num_samples, num_antenna_bs, num_user, total_blocks)

                                 ============= used for visualization =============
                                 channel_hr_NLOS: array (num_samples, num_elements_irs, 1, moving_frames)
                                 channel_hd_NLOS: array (num_samples, num_antenna_bs, 1, moving_frames)

        :return: combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks)

                 ============= used for visualization =============
                 user_corrdinate_all: array [moving_frames, num_user, 3]
                 range_coordinate_all_without_z: array [sample_num_x * sample_num_y, 1, 2]
                 combined_A_range_all: array [sample_num_x * sample_num_y, 1, num_antenna_bs, (num_elements_irs + 1), 1, moving_frames]
                 a_irs_bs_range_all: array [sample_num_x * sample_num_y, num_elements_irs, 1]
                 a_irs_user_range_all: array [sample_num_x * sample_num_y, num_elements_irs, 1]
    """
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_G = np.empty([num_samples, num_antenna_bs, num_elements_irs, total_blocks], dtype=complex)
    channel_hr = np.empty([num_samples, num_elements_irs, num_user, total_blocks], dtype=complex)
    channel_hd = np.empty([num_samples, num_antenna_bs, num_user, total_blocks], dtype=complex)
    combined_A = np.empty([num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks], dtype=complex)


    # =========== Remember, though each transmission frame contains N_w_using blocks, we only need to visualize one block per frame ========================
    'record the coordinate (Trajectory) of all the users'
    user_corrdinate_all = np.empty([moving_frames, num_user, 3])

    'record the NLOS part of the channels --> notice! Here we do not record NLOS w.r.t. three users, we average them up and only record one'
    # here, num_samples = 1
    channel_hr_NLOS = np.empty([num_samples, num_elements_irs, 1, moving_frames], dtype=complex)
    channel_hd_NLOS = np.empty([num_samples, num_antenna_bs, 1, moving_frames], dtype=complex)

    'The channels for the whole moving range'
    range_coordinate_all_without_z = np.empty([sample_num_x * sample_num_y, 1, 2])

    channel_G_range_all = np.empty([sample_num_x * sample_num_y, num_antenna_bs, num_elements_irs, moving_frames], dtype=complex)
    channel_hr_range_all = np.empty([sample_num_x * sample_num_y, num_elements_irs, 1, moving_frames], dtype=complex)
    channel_hd_range_all = np.empty([sample_num_x * sample_num_y, num_antenna_bs, 1, moving_frames], dtype=complex)
    combined_A_range_all = np.empty([sample_num_x * sample_num_y, 1, num_antenna_bs, (num_elements_irs + 1), 1, moving_frames], dtype=complex)

    'Plotting the array response'
    # steering vectors
    a_irs_bs_range_all = np.empty([sample_num_x * sample_num_y, num_elements_irs, 1], dtype=complex)
    a_irs_user_range_all = np.empty([sample_num_x * sample_num_y, num_elements_irs, 1], dtype=complex)

    # generate channel
    for block in range(total_blocks):
        # each moving instance got the same 'MARKOV' movement model
        if block == 0:
            'generate the initial coordinate and moving directions for the mobile UEs'
            # coordinate_UE_curr: (num_samples, num_user, 3); curr_direction_state: (num_samples, num_user)
            # we assume the whole moving range is a 40*70 rectangular, the upper-left corner coordinate: (x,y,z) = (5,-35,-20)
            coordinate_UE_curr, curr_direction_state = initial_moving_states_visualizing_traj(num_user)

            'we only need to generate LOS part (G_LOS, pathloss_irs_bs) for G once'
            # for LOS part of G
            G_LOS, pathloss_irs_bs, aod_irs_y_highRank = generate_G_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, coordinate_UE_curr, num_samples, N_G)

            '[initialize] deal with the Gauss-Marov evolution of the NLOS part for G'
            NLOS_part_G_real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, num_elements_irs])
            NLOS_part_G_imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, num_elements_irs])

            '[initialize] deal with the Gauss-Marov evolution of the NLOS part for h_r'
            NLOS_part_hr_real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_elements_irs, num_user])
            NLOS_part_hr_imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_elements_irs, num_user])

            '[initialize] direct link h_d'
            tilde_hd_real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, num_user])
            tilde_hd_imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, num_user])
        else:
            'find the current UE coordinate and update the direction state matrix'
            # curr_direction_state: array (num_samples,num_user); coordinate_UE_curr: array (num_samples,num_user,3)
            curr_direction_state, coordinate_UE_curr = generate_location(curr_direction_state, coordinate_UE_curr, num_samples, num_user)

            'deal with the Gauss-Marov evolution of the NLOS part for both G and h_r'
            # innovation part Gauss-Markov evolution model [for NLOS part of the channel] # /2 because of we devide the real and imag part!
            xi_t = np.random.normal(loc=0, scale=np.sqrt((1 - rho_G ** 2) / 2), size=[num_samples, num_antenna_bs, num_elements_irs])
            xi_r = np.random.normal(loc=0, scale=np.sqrt((1 - rho_hr ** 2) / 2), size=[num_samples, num_elements_irs, num_user])
            xi_d = np.random.normal(loc=0, scale=np.sqrt((1 - rho_hd ** 2) / 2), size=[num_samples, num_antenna_bs, num_user])

            NLOS_part_G_real = rho_G * NLOS_part_G_real + xi_t
            NLOS_part_G_imag = rho_G * NLOS_part_G_imag + xi_t

            NLOS_part_hr_real = rho_hr * NLOS_part_hr_real + xi_r
            NLOS_part_hr_imag = rho_hr * NLOS_part_hr_imag + xi_r

            'deal with the Gauss-Marov evolution of the direct link h_d'
            tilde_hd_real = rho_hd * tilde_hd_real + xi_d
            tilde_hd_imag = rho_hd * tilde_hd_imag + xi_d

        'channel_G: ndarray, (num_samples, num_antenna_bs, num_elements_irs, total_blocks), channel between IRS and BS'
        # for NLOS part of G
        NLOS_part_oneSample_G = NLOS_part_G_real + 1j * NLOS_part_G_imag
        # channel_G: array (num_samples, num_antenna_bs, num_elements_irs, total_blocks)
        channel_G[:, :, :, block] = (np.sqrt(Rician_factor / (1 + Rician_factor)) * G_LOS + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_G) * pathloss_irs_bs

        'channel_hr: ndarray, (num_samples, num_elements_irs, num_user, total_blocks), channel between IRS and user'
        'channel_hd: (num_samples, num_antenna_bs, num_user, total_blocks), channel between BS and user'
        # for LOS part of hr
        hr_LOS, pathloss_irs_user, pathloss_bs_user = generate_hrLOS_hdPathloss(params_system, irs_Nh, location_bs, location_irs, scale_factor, coordinate_UE_curr, num_samples)
        # for NLOS part of hr
        NLOS_part_oneSample_hr = NLOS_part_hr_real + 1j * NLOS_part_hr_imag

        # Hadamard multiplication can be done by broadcasting
        pathloss_irs_user = np.expand_dims(pathloss_irs_user, axis=1)
        pathloss_bs_user = np.expand_dims(pathloss_bs_user, axis=1)

        # channel_hr: array (num_samples, num_elements_irs, num_user, total_blocks)
        channel_hr[:, :, :, block] = (np.sqrt(Rician_factor / (1 + Rician_factor)) * hr_LOS + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_hr) * pathloss_irs_user
        # channel_hd: array (num_samples, num_antenna_bs, num_user, total_blocks)
        channel_hd[:, :, :, block] = (tilde_hd_real + 1j * tilde_hd_imag) * pathloss_bs_user

        'Compute the overall channel combined_A'
        # we try to reduce the computational complexity by avoiding the loops w.r.t. the num_samples
        for kk in range(num_user):
            channel_hr_tmp = np.expand_dims(channel_hr[:, :, kk, block], axis=1)
            # Broadcasting to perform the multiplication between G and diag(h_r)
            cascaded_RIS_channel_kk = channel_G[:, :, :, block] * channel_hr_tmp
            # The order of concatenation is IMP！！！！！！！！！！！
            channel_hd_tmp = np.expand_dims(channel_hd[:, :, kk, block], axis=2)  # channel_hd_tmp: array (num_samples, num_antenna_bs, 1)
            combined_A[:, :, :, kk, block] = np.concatenate((channel_hd_tmp, cascaded_RIS_channel_kk), axis=2)

        # ===================================================================================================
        # ===================================================================================================
        'For visualization ---> only need to visualize the first block per frame'
        if (block % N_w_using) == 0:
            # 'record the user location at the chosen block'
            user_corrdinate_all[block//N_w_using, :, :] = coordinate_UE_curr

            # 'record the NLOS part of the channels at the chosen block'
            channel_hr_NLOS[:, :, :, block//N_w_using] = np.mean(NLOS_part_oneSample_hr, axis=2, keepdims=True)
            channel_hd_NLOS[:, :, :, block//N_w_using] = np.mean(tilde_hd_real + 1j * tilde_hd_imag, axis=2, keepdims=True)

            'Generate the channel for the whole moving range by sampling the points on the range'
            # since the sampling point is equivalent to num_user = 1
            params_system_sampling = (num_antenna_bs, num_elements_irs, 1)
            # x-coordinate
            for jj in range(sample_num_x):
                # y-coordinate
                for kk in range(sample_num_y):
                    # 40*70 range
                    range_coordinate_x = 5.0 + jj * (40.0 / sample_num_x)
                    range_coordinate_y = -35.0 + kk * (70.0 / sample_num_y)
                    range_coordinate_z = -20.0
                    range_coordinate = [range_coordinate_x, range_coordinate_y, range_coordinate_z]
                    range_coordinate = np.array(range_coordinate).reshape([1, 1, 3]) # since the input to the LOS generate function: location_user: array(num_samples, num_user, 3)
                    # 'record the x,y-coordinates for range_coordinate_all_without_z'
                    range_coordinate_all_without_z[sample_num_y * jj + kk, :, 0] = range_coordinate_x
                    range_coordinate_all_without_z[sample_num_y * jj + kk, :, 1] = range_coordinate_y

                    'Generate the channels on the whole range'
                    # 'channel_G_range_all: array [sample_num_x * sample_num_y, num_antenna_bs, num_elements_irs, moving_frames]'
                    channel_G_range_all[sample_num_y * jj + kk, :, :, block//N_w_using] = channel_G[0, :, :, block] # since here, the num_sample = 1

                    # 'channel_hr_range_all: array [sample_num_x * sample_num_y, num_elements_irs, 1, moving_frames]'
                    # 'channel_hd_range_all: array [sample_num_x * sample_num_y, num_antenna_bs, 1, moving_frames]'
                    hr_LOS_range, pathloss_irs_user_range, pathloss_bs_user_range = generate_hrLOS_hdPathloss(params_system_sampling, irs_Nh, location_bs, location_irs, scale_factor, range_coordinate, num_samples)

                    pathloss_irs_user_range = np.squeeze(pathloss_irs_user_range)
                    pathloss_bs_user_range = np.squeeze(pathloss_bs_user_range)

                    channel_hr_range_all[sample_num_y * jj + kk, :, :, block//N_w_using] = (np.sqrt(Rician_factor / (1 + Rician_factor)) * hr_LOS_range[0, :, :] + np.sqrt(1 / (1 + Rician_factor)) * channel_hr_NLOS[0, :, :, block//N_w_using]) * pathloss_irs_user_range
                    channel_hd_range_all[sample_num_y * jj + kk, :, :, block//N_w_using] = channel_hd_NLOS[0, :, :, block//N_w_using] * pathloss_bs_user_range

                    'Compute the overall channel combined_A for the sampling point'
                    channel_hr_range_all_tmp = np.expand_dims(channel_hr_range_all[sample_num_y * jj + kk, :, 0, block//N_w_using], axis=0) # array(1, num_elements_irs)
                    # Broadcasting to perform the multiplication between G and diag(h_r)
                    cascaded_RIS_channel_range_tmp = channel_G_range_all[sample_num_y * jj + kk, :, :, block//N_w_using] * channel_hr_range_all_tmp # cascaded_RIS_channel_range_tmp: array (num_antenna_bs, num_elements_irs)
                    channel_hd_range_all_tmp = channel_hd_range_all[sample_num_y * jj + kk, :, :, block//N_w_using]  # channel_hd_range_all_tmp: array (num_antenna_bs, 1)
                    # The order of concatenation is IMP！！！！！！！！！！！
                    combined_A_range_all[sample_num_y * jj + kk, 0, :, :, 0, block//N_w_using] = np.concatenate((channel_hd_range_all_tmp, cascaded_RIS_channel_range_tmp), axis=1)

                    'For plotting the array response of the sensing vector in the UL'
                    a_irs_bs_range_all[sample_num_y * jj + kk, :, :], a_irs_user_range_all[sample_num_y * jj + kk, :, :] = generate_spatial_signature(params_system_sampling, range_coordinate, location_bs, location_irs, irs_Nh, aod_irs_y_highRank, N_G)

    return channel_G, channel_hr, channel_hd, combined_A, user_corrdinate_all, range_coordinate_all_without_z, combined_A_range_all, a_irs_bs_range_all, a_irs_user_range_all



def main_generate_channel(N_G, moving_frames, N_w_using, sample_num_x, sample_num_y, num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, total_blocks,
                          scale_factor, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
                          location_irs=np.array([0, 0, 0]), isTesting=False):
    """
        :param:  isTesting: [True] save the generated {combined_A} and reusing {combined_A} in testing
                            [False] randomly generate {combined_A} for all the blocks each time calling this function

        :return: combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks)

                 ============= used for visualization =============
                 user_corrdinate_all: array [moving_frames, num_user, 3]
                 range_coordinate_all_without_z: array [sample_num_x * sample_num_y, 1, 2]
                 combined_A_range_all: array [sample_num_x * sample_num_y, 1, num_antenna_bs, (num_elements_irs + 1), 1, moving_frames]
    """

    params_system = (num_antenna_bs, num_elements_irs, num_user)
    if isTesting:
        if not os.path.exists('Rician_channel'):
            os.makedirs('Rician_channel')
        params_all = (num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, scale_factor, total_blocks)
        file_test_channel = './Rician_channel/channel' + str(params_all) + '.mat'
        if os.path.exists(file_test_channel):
            data_test = sio.loadmat(file_test_channel)
            channel_G, channel_hr, channel_hd, combined_A, user_corrdinate_all, range_coordinate_all_without_z, combined_A_range_all, a_irs_bs_range_all, a_irs_user_range_all = data_test['channel_G'], data_test['channel_hr'], data_test['channel_hd'], data_test['combined_A'], data_test['user_corrdinate_all'], data_test['range_coordinate_all_without_z'], data_test['combined_A_range_all'], data_test['a_irs_bs_range_all'], data_test['a_irs_user_range_all']
        else:
            channel_G, channel_hr, channel_hd, combined_A, user_corrdinate_all, range_coordinate_all_without_z, combined_A_range_all, a_irs_bs_range_all, a_irs_user_range_all = generate_channel(moving_frames, N_w_using, sample_num_x, sample_num_y, params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
                                                                                                                                                              location_bs=location_bs,
                                                                                                                                                              location_irs=location_irs,
                                                                                                                                                              Rician_factor=Rician_factor,
                                                                                                                                                              scale_factor=scale_factor,
                                                                                                                                                              num_samples=num_samples, rho_G=rho_G,
                                                                                                                                                              rho_hr=rho_hr, rho_hd=rho_hd, N_G=N_G)

            sio.savemat(file_test_channel, {'channel_G': channel_G, 'channel_hr': channel_hr, 'channel_hd': channel_hd, 'combined_A': combined_A, 'user_corrdinate_all': user_corrdinate_all, 'range_coordinate_all_without_z': range_coordinate_all_without_z, 'combined_A_range_all': combined_A_range_all, 'a_irs_bs_range_all': a_irs_bs_range_all, 'a_irs_user_range_all': a_irs_user_range_all})
    else:
        channel_G, channel_hr, channel_hd, combined_A, user_corrdinate_all, range_coordinate_all_without_z, combined_A_range_all, a_irs_bs_range_all, a_irs_user_range_all = generate_channel(moving_frames, N_w_using, sample_num_x, sample_num_y, params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
                                                                                                                                                          location_bs=location_bs,
                                                                                                                                                          location_irs=location_irs,
                                                                                                                                                          Rician_factor=Rician_factor,
                                                                                                                                                          scale_factor=scale_factor,
                                                                                                                                                          num_samples=num_samples, rho_G=rho_G,
                                                                                                                                                          rho_hr=rho_hr, rho_hd=rho_hd, N_G=N_G)

    '=============================================================================='
    'This part is only for the MATLAB program to do refining'
    params_all = (num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, scale_factor, total_blocks)
    file_test_channel_forRefining = './Rician_channel/channel_Test' + str(params_all) + '.mat'

    sio.savemat(file_test_channel_forRefining, {'channel_bs_irs': channel_G, 'channel_irs_user': channel_hr, 'channel_bs_user': channel_hd})


    return combined_A, user_corrdinate_all, range_coordinate_all_without_z, combined_A_range_all, a_irs_bs_range_all, a_irs_user_range_all




'This stats can be used in both training/testing/visualizing, so after generated in the training process, just copy this file into testing/visualizing folder!'
def main_generate_channel_LMMSE_stats(N_G, num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, total_blocks,
                          scale_factor, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
                          location_irs=np.array([0, 0, 0]), isTesting=False):
    """
        :param:  isTesting: [True] save the generated {combined_A} and reusing {combined_A} in testing
                            [False] randomly generate {combined_A} for all the blocks each time calling this function

        :return: combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks), complex 128
    """

    params_all = (num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, scale_factor, total_blocks)
    file_test_channel = './Rician_channel/LMMSE_channel_Stats' + str(params_all) + '.mat'

    data_test = sio.loadmat(file_test_channel)
    channel_G, channel_hr, channel_hd, combined_A = data_test['channel_G'], data_test['channel_hr'], data_test['channel_hd'], data_test['combined_A']

    # params_system = (num_antenna_bs, num_elements_irs, num_user)
    # if isTesting:
    #     if not os.path.exists('Rician_channel'):
    #         os.makedirs('Rician_channel')
    #     params_all = (num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, scale_factor, total_blocks)
    #     file_test_channel = './Rician_channel/LMMSE_channel_Stats' + str(params_all) + '.mat'
    #     if os.path.exists(file_test_channel):
    #         data_test = sio.loadmat(file_test_channel)
    #         channel_G, channel_hr, channel_hd, combined_A = data_test['channel_G'], data_test['channel_hr'], data_test['channel_hd'], data_test['combined_A']
    #     else:
    #         channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
    #                                       location_bs=location_bs,
    #                                       location_irs=location_irs,
    #                                       Rician_factor=Rician_factor,
    #                                       scale_factor=scale_factor,
    #                                       num_samples=num_samples, rho_G=rho_G,
    #                                       rho_hr=rho_hr, rho_hd=rho_hd, N_G=N_G)
    #
    #         sio.savemat(file_test_channel, {'channel_G': channel_G, 'channel_hr': channel_hr, 'channel_hd': channel_hd, 'combined_A': combined_A})
    # else:
    #     channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
    #                                   location_bs=location_bs,
    #                                   location_irs=location_irs,
    #                                   Rician_factor=Rician_factor,
    #                                   scale_factor=scale_factor,
    #                                   num_samples=num_samples, rho_G=rho_G,
    #                                   rho_hr=rho_hr, rho_hd=rho_hd, N_G=N_G)
    return combined_A


# '============Testing Scripts for the channel generation=================='
# frames = 12#10 # we assume that the user will change 10 times to test!
# tau = 3#10
# # for each designed w, it has the potential to be used in multiple frames due to the slow change in the LOS part of the channel
# # define the frames number that a w could be used in is N_w_using
# N_w_using = 20#30 # notice, here, we assume the channel reciprocity!
# total_blocks = N_w_using * frames # this is 300 instead of 10 anymore!
#
# sample_num_x = 25
# sample_num_y = 25
#
# num_ris_elements = 100
# num_antenna_bs = 8 #1
# num_user = 3 #1
# irs_Nh = 10
# Rician_factor = 10
# # enlarge this scale factor will never cause effect on the optimal rate
# scale_factor = 100
# 'in visualization, the sample size should be 1'
# batch_size_test_1block = 1
# '-------------------------------------------------------------------------'
# '-------------------Testing the generate speed----------------------------'
# start_time = time.time()
#
# combined_A, user_corrdinate_all, range_coordinate_all_without_z, combined_A_range_all, a_irs_bs_range_all, a_irs_user_range_all = main_generate_channel(frames, N_w_using, sample_num_x, sample_num_y, num_antenna_bs, num_ris_elements, num_user, irs_Nh, batch_size_test_1block, Rician_factor, total_blocks,
#                           scale_factor, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
#                           location_irs=np.array([0, 0, 0]), isTesting=False)
#
# print("Already training for %s seconds" % (time.time() - start_time))
#
# # '---------------------------------------------------------------------------'
# # '-------------------Testing the channel strength----------------------------'
# # params_system = (num_antenna_bs, num_ris_elements, num_user)
# # channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
# #                                   location_bs=np.array([100, -100, 0]),
# #                                   location_irs=np.array([0, 0, 0]),
# #                                   Rician_factor=Rician_factor,
# #                                   scale_factor=scale_factor,
# #                                   num_samples=batch_size_test_1block, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995)
# #
# # # combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks)
# # combined_A_mean = np.mean(combined_A, axis=0)
# # norm_A_block_start = np.linalg.norm(combined_A_mean[:, :, :, 0])
# # norm_A_block_final = np.linalg.norm(combined_A_mean[:, :, :, (total_blocks - 1)])
# #
# # # channel_hd: array (num_samples, num_antenna_bs, num_user, total_blocks)
# # channel_hd_maen = np.mean(channel_hd, axis=0)
# # norm_channel_hd_block_start = np.linalg.norm(channel_hd_maen[:, :, 0])
# # norm_channel_hd_block_final = np.linalg.norm(channel_hd_maen[:, :, (total_blocks - 1)])
# #
# # # channel_hr:
# # channel_hr_maen = np.mean(channel_hd, axis=0)
# # norm_channel_hr_block_start = np.linalg.norm(channel_hr_maen[:, :, 0])
# # norm_channel_hr_block_final = np.linalg.norm(channel_hr_maen[:, :, (total_blocks - 1)])
# #
# # # channel_G:
# # channel_G_maen = np.mean(channel_G, axis=0)
# # norm_channel_G_block_start = np.linalg.norm(channel_G_maen[:, :, 0])
# # norm_channel_G_block_final = np.linalg.norm(channel_G_maen[:, :, (total_blocks - 1)])
#
# check = combined_A