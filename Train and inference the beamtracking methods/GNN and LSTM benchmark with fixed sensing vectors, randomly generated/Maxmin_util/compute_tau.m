function [ tau ] = compute_tau(H, m, Pt)
    % this function is used for compute the minimum SINR under OLP: tau
    
    % input H: M*K
    % input m: (num_user, num_antenna_bs, num_antenna_bs)
    
    % output tau: 1*1 R++ number

    [~, num_user] = size(H);

    tmp = 0;
    for k = 1:num_user
        h_k = H(:, k); % h_k: M*1
        m_k = squeeze(m(k, :, :)); % after squeeze, m_k: num_antenna_bs*num_antenna_bs
        m_inv = inv(m_k); % m must be full-ranked since we added the eye-martix
        hmh = real(h_k' * m_inv * h_k / num_user); % hmh: 1*1
        % add w.r.t. all the users
        tmp = tmp + 1 / hmh;
    end
    tau = num_user * Pt / tmp;
end
