function [g_opt, tau_opt] = compute_beamforming_vec(H, Pt)
    % this function is used for computing the optimal beamforming vector g and the optimal power coefficients tau
    
    % input H: M*K

    % output g_opt: K * M
    % output tau_opt: 1*1 R++ number

    [num_antenna_bs, num_user] = size(H);
    % initialize the beamforming vector g
    g_opt = zeros(num_user, num_antenna_bs);

    q_opt = find_q_fixed_point(H, Pt);
    m_opt = compute_m(H, q_opt);
    % compute tau_opt to prepare for computing the optimal power allowcation 'p'
    tau_opt = compute_tau(H, m_opt, Pt);
    
    % compute the optimal beamforming vector g_opt
    for k = 1:num_user
        h_k = H(:, k); % h_k: M*1
        m_k = squeeze(m_opt(k, :, :)); % after squeeze, m_k: num_antenna_bs*num_antenna_bs
        m_inv = inv(m_k); % m must be full-ranked since we added the eye-martix
        g_k_tmp = m_inv * h_k;
        g_k = g_k_tmp / norm(g_k_tmp); % g_k: M*1
        g_opt(k, :) = g_k.';
    end
end
