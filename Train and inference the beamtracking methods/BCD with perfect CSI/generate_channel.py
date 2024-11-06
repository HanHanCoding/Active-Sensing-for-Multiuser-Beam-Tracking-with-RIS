import numpy as np
import scipy.io as sio
from scipy.linalg import dft
import os
import time

'This code is for generating channels according to the Rician Model. '
'for generate_channel_{G, hr}, we directly generate channels in all the total_blocks'

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

'generate the initial coordinate and moving directions for the mobile UEs'
def initial_moving_states(num_samples, num_user):
    """
        :intermediate variables: Coordinate_direction_all: array (num_samples, num_user, 4)

        :return: coordinate_UE_init: (num_samples, num_user, 3)
                 init_direction_state: (num_samples, num_user)
    """
    Coordinate_direction_all = np.empty([num_samples, num_user, 4])
    # x-axis init
    Coordinate_direction_all[:, :, 0] = np.random.uniform(5.0, 45.0, size=[num_samples, num_user])
    # y-axis init
    Coordinate_direction_all[:, :, 1] = np.random.uniform(-35.0, 35.0, size=[num_samples, num_user])
    # z-axis init
    Coordinate_direction_all[:, :, 2] = -20.0 * np.ones([num_samples, num_user])
    # initial moving direction
    Coordinate_direction_all[:, :, 3] = np.random.uniform(0, 2*np.pi, size=[num_samples, num_user])

    coordinate_UE_init = Coordinate_direction_all[:, :, 0:3]
    init_direction_state = Coordinate_direction_all[:, :, 3]

    return coordinate_UE_init, init_direction_state


def path_loss_r(d):
    loss = 28.0 + 16.9 * np.log10(d)
    return loss


def path_loss_d(d):  # this is for the BS-UE link
    loss = 32.0 + 43.3 * np.log10(d)
    return loss


def generate_pathloss_aoa_aod(location_user, location_bs, location_irs, num_samples, num_user):
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
    aoa_aod = (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z)
    return pathloss, aoa_aod


# channel between IRS and BS
'In this program, to ensure that the Channel G supports the spatial multiplexing, we assume that G will have multiple paths - not necessarily the LOS path!!'
def generate_G_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, location_user, num_samples, N_G):
    """
        :param: N_G: integer ---> # of LOS links in the model of channel G
        :param: location_user: array (num_samples,num_user,3)
        :param: location_bs: array (3,)
        :param: location_irs: array (3,)

        [Modeling of the uplink channel steering vector]
        aod_irs_y_highRank, aod_irs_z_highRank, aoa_bs_highRank: array (num_samples, N_G)

        :return: los_irs_bs: array (num_antenna_bs, num_elements_irs)
                 pathloss_irs_bs: float
    """
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    'find the path losses'
    pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs, num_samples, num_user)
    (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod

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

    return los_irs_bs, pathloss_irs_bs


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
    pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs, num_samples, num_user)
    (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod
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


# generate the cascaded channel A_k for all the users (w.r.t. all the blocks!)
def generate_channel(params_system, irs_Nh, total_blocks, location_bs, location_irs, Rician_factor, scale_factor, num_samples, rho_G, rho_hr, rho_hd, N_G):
    """
        :param  los_irs_bs (G_LOS): array (num_antenna_bs, num_elements_irs)
                pathloss_irs_bs: float

                los_irs_user: array (num_samples, num_elements_irs, num_user)
                pathloss_irs_user: array (num_samples, num_user)

                pathloss_bs_user: array (num_samples, num_user)

        :intermediate variables: channel_G: array (num_samples, num_antenna_bs, num_elements_irs, total_blocks)
                                 channel_hr: array (num_samples, num_elements_irs, num_user, total_blocks)
                                 channel_hd: array (num_samples, num_antenna_bs, num_user, total_blocks)

        :return:channel_G: array (num_samples, num_antenna_bs, num_elements_irs, total_blocks)
                channel_hr: array (num_samples, num_elements_irs, num_user, total_blocks)
                channel_hd: array (num_samples, num_antenna_bs, num_user, total_blocks)
                combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks)
    """
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_G = np.empty([num_samples, num_antenna_bs, num_elements_irs, total_blocks], dtype=complex)
    channel_hr = np.empty([num_samples, num_elements_irs, num_user, total_blocks], dtype=complex)
    channel_hd = np.empty([num_samples, num_antenna_bs, num_user, total_blocks], dtype=complex)
    combined_A = np.empty([num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks], dtype=complex)

    # generate channel
    for block in range(total_blocks):
        # each moving instance got the same 'MARKOV' movement model
        if block == 0:
            'generate the initial coordinate and moving directions for the mobile UEs'
            # coordinate_UE_curr: (num_samples, num_user, 3); curr_direction_state: (num_samples, num_user)
            # we assume the whole moving range is a 40*70 rectangular, the upper-left corner coordinate: (x,y,z) = (5,-35,-20)
            coordinate_UE_curr, curr_direction_state = initial_moving_states(num_samples, num_user)

            'we only need to generate LOS part (G_LOS, pathloss_irs_bs) for G once'
            # for LOS part of G
            G_LOS, pathloss_irs_bs = generate_G_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, coordinate_UE_curr, num_samples, N_G)

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

    return channel_G, channel_hr, channel_hd, combined_A



def main_generate_channel(N_G, num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, total_blocks,
                          scale_factor, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
                          location_irs=np.array([0, 0, 0]), isTesting=False):
    """
        :param:  isTesting: [True] save the generated {combined_A} and reusing {combined_A} in testing
                            [False] randomly generate {combined_A} for all the blocks each time calling this function

        :return: combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks), complex 128
    """

    params_system = (num_antenna_bs, num_elements_irs, num_user)
    if isTesting:
        if not os.path.exists('Rician_channel'):
            os.makedirs('Rician_channel')
        params_all = (num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, scale_factor, total_blocks)
        file_test_channel = './Rician_channel/channel' + str(params_all) + '.mat'
        if os.path.exists(file_test_channel):
            data_test = sio.loadmat(file_test_channel)
            channel_G, channel_hr, channel_hd, combined_A = data_test['channel_G'], data_test['channel_hr'], data_test['channel_hd'], data_test['combined_A']
        else:
            channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
                                          location_bs=location_bs,
                                          location_irs=location_irs,
                                          Rician_factor=Rician_factor,
                                          scale_factor=scale_factor,
                                          num_samples=num_samples, rho_G=rho_G,
                                          rho_hr=rho_hr, rho_hd=rho_hd, N_G=N_G)

            sio.savemat(file_test_channel, {'channel_G': channel_G, 'channel_hr': channel_hr, 'channel_hd': channel_hd, 'combined_A': combined_A})
    else:
        channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
                                      location_bs=location_bs,
                                      location_irs=location_irs,
                                      Rician_factor=Rician_factor,
                                      scale_factor=scale_factor,
                                      num_samples=num_samples, rho_G=rho_G,
                                      rho_hr=rho_hr, rho_hd=rho_hd, N_G=N_G)
    return combined_A


'Since the batch size is too large in the testing phase (batch size > 1000), so we do not save the generated channel '
# 'scipy.io.matlab.miobase.MatWriteError: Matrix too large to save with Matlab 5 format'
# this function obly used for testing the rate vs frames, in visualizing w and V program, we don't use this one since we need to save the channel (in this case, batch_size =1)
def main_generate_channel_testing_rate(N_G, num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, total_blocks,
                          scale_factor, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
                          location_irs=np.array([0, 0, 0])):
    """
        :param:  isTesting: [True] save the generated {combined_A} and reusing {combined_A} in testing
                            [False] randomly generate {combined_A} for all the blocks each time calling this function

        :return: combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks), complex 128
    """

    params_system = (num_antenna_bs, num_elements_irs, num_user)

    channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
                                  location_bs=location_bs,
                                  location_irs=location_irs,
                                  Rician_factor=Rician_factor,
                                  scale_factor=scale_factor,
                                  num_samples=num_samples, rho_G=rho_G,
                                  rho_hr=rho_hr, rho_hd=rho_hd, N_G=N_G)
    return combined_A





#
# 'This block should be commented when it is not in use'
# '------------------------------------------------------------------------------------------------------------------'
# '------------------------------------------------------------------------------------------------------------------'
# 'Generate the testing channel data for computing the upperbound/lowerbound of the sum-rate maximization problem'
# def generate_testing_channel_forBCD_algorithm(file_path, raw_SNR_DL, moving_frames, N_w_using, num_antenna_bs, num_elements_irs, num_user, irs_Nh,
#                                               num_samples, Rician_factor, total_blocks, scale_factor, rho_G=0.9995,
#                                               rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
#                                               location_irs=np.array([0, 0, 0])):
#     """
#         :return:channel_G: array (num_samples, num_antenna_bs, num_elements_irs, moving_frames)
#                 channel_hr: array (num_samples, num_elements_irs, num_user, moving_frames)
#                 channel_hd: array (num_samples, num_antenna_bs, num_user, moving_frames)
#     """
#
#     params_system = (num_antenna_bs, num_elements_irs, num_user)
#
#     start_time = time.time()
#     channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh,
#                                                                      total_blocks=total_blocks,
#                                                                      location_bs=location_bs,
#                                                                      location_irs=location_irs,
#                                                                      Rician_factor=Rician_factor,
#                                                                      scale_factor=scale_factor,
#                                                                      num_samples=num_samples, rho_G=rho_G,
#                                                                      rho_hr=rho_hr, rho_hd=rho_hd)
#     print("Already training for %s seconds" % (time.time() - start_time))
#
#     'Since each frame contains N_w_using blocks (not include the 0th block for sensing), we only test the optimal/rand rate for the 1st block of each frame'
#     channel_G_test = np.empty([num_samples, num_antenna_bs, num_elements_irs, moving_frames], dtype=complex)
#     channel_hr_test = np.empty([num_samples, num_elements_irs, num_user, moving_frames], dtype=complex)
#     channel_hd_test = np.empty([num_samples, num_antenna_bs, num_user, moving_frames], dtype=complex)
#
#     for frame in range(moving_frames):
#         channel_G_test[:, :, :, frame] = channel_G[:, :, :, frame*N_w_using]
#         channel_hr_test[:, :, :, frame] = channel_hr[:, :, :, frame*N_w_using]
#         channel_hd_test[:, :, :, frame] = channel_hd[:, :, :, frame*N_w_using]
#
#
#     'save the channels for the MATLAB code'
#     sio.savemat(file_path,
#                 {'channel_bs_user': channel_hd_test,
#                  'channel_irs_user': channel_hr_test,
#                  'channel_bs_irs': channel_G_test,
#                  'raw_SNR_DL': raw_SNR_DL})
#
#
# '# Main generate function!'
# def main():
#     num_antenna_bs, num_elements_irs, num_user = 8, 100, 3
#     N_w_using = 30
#     moving_frames = 12
#     total_blocks = N_w_using * moving_frames
#     num_samples = 1080
#
#     irs_Nh = 10
#     params_system = (num_antenna_bs, num_elements_irs, num_user)
#     Rician_factor = 10
#     scale_factor = 120 # this is for compute the channel statistics with better numerical stability
#
#     # compute the RAW SNR in the downlink!
#     Pt_down = 20  # dBm
#     noise_power_db_down = -85  # -100 - 15 #dBm
#     raw_SNR_DL = float(Pt_down - noise_power_db_down - scale_factor)
#
#     # generate testing channel for the BCD algorithm!
#     file_path = './channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
#     if not os.path.exists(file_path):
#         generate_testing_channel_forBCD_algorithm(file_path, raw_SNR_DL, moving_frames, N_w_using, num_antenna_bs, num_elements_irs, num_user, irs_Nh,
#                                                   num_samples, Rician_factor, total_blocks, scale_factor, rho_G=0.9995,
#                                                   rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
#                                                   location_irs=np.array([0, 0, 0]))
#     else:
#         print('Testing channel data exists')
#
# '# generate the channel data'
# if __name__ == '__main__':
#     main()
# '------------------------------------------------------------------------------------------------------------------'
# '------------------------------------------------------------------------------------------------------------------'


# '============Testing Scripts for the channel generation=================='
# frames = 15#10 # we assume that the user will change 10 times to test!
# tau = 3#10
# # for each designed w, it has the potential to be used in multiple frames due to the slow change in the LOS part of the channel
# # define the frames number that a w could be used in is N_w_using
# N_w_using = 40#30 # notice, here, we assume the channel reciprocity!
# total_blocks = N_w_using * frames # this is 300 instead of 10 anymore!
# 
# num_ris_elements = 100
# num_antenna_bs = 8 #1
# num_user = 3 #1
# irs_Nh = 10 # Number of elements on the x-axis of rectangle RIS
# Rician_factor = 10
# # enlarge this scale factor will never cause effect on the optimal rate
# scale_factor = 120
# 'This size must be a multiple of 9！！！！！！！！'
# batch_size_test_1block = 450
# '-------------------------------------------------------------------------'
# '-------------------Testing the generate speed----------------------------'
# # start_time = time.time()
# # combined_A = main_generate_channel(num_antenna_bs, num_ris_elements, num_user, irs_Nh,
# #                                                          batch_size_test_1block, Rician_factor, total_blocks, scale_factor,
# #                                                          rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995, location_bs=np.array([100, -100, 0]),
# #                                                          location_irs=np.array([0, 0, 0]), isTesting=False)
# # print("Already training for %s seconds" % (time.time() - start_time))
# 
# '---------------------------------------------------------------------------'
# '-------------------Testing the channel strength----------------------------'
# params_system = (num_antenna_bs, num_ris_elements, num_user)
# channel_G, channel_hr, channel_hd, combined_A = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
#                                   location_bs=np.array([100, -100, 0]),
#                                   location_irs=np.array([0, 0, 0]),
#                                   Rician_factor=Rician_factor,
#                                   scale_factor=scale_factor,
#                                   num_samples=batch_size_test_1block, rho_G=0.9995, rho_hr=0.9995, rho_hd=0.9995)
# 
# # combined_A: array (num_samples, num_antenna_bs, (num_elements_irs + 1), num_user, total_blocks)
# combined_A_mean = np.mean(combined_A, axis=0)
# norm_A_block_start = np.linalg.norm(combined_A_mean[:, :, :, 0])
# norm_A_block_final = np.linalg.norm(combined_A_mean[:, :, :, (total_blocks - 1)])
# 
# # channel_hd: array (num_samples, num_antenna_bs, num_user, total_blocks)
# channel_hd_maen = np.mean(channel_hd, axis=0)
# norm_channel_hd_block_start = np.linalg.norm(channel_hd_maen[:, :, 0])
# norm_channel_hd_block_final = np.linalg.norm(channel_hd_maen[:, :, (total_blocks - 1)])
# 
# # channel_hr:
# channel_hr_maen = np.mean(channel_hd, axis=0)
# norm_channel_hr_block_start = np.linalg.norm(channel_hr_maen[:, :, 0])
# norm_channel_hr_block_final = np.linalg.norm(channel_hr_maen[:, :, (total_blocks - 1)])
# 
# # channel_G:
# channel_G_maen = np.mean(channel_G, axis=0)
# norm_channel_G_block_start = np.linalg.norm(channel_G_maen[:, :, 0])
# norm_channel_G_block_final = np.linalg.norm(channel_G_maen[:, :, (total_blocks - 1)])
# 
# check = combined_A