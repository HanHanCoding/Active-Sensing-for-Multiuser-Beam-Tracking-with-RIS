function [ q ] = find_q_fixed_point(H, Pt)
    % this function is used for computing the unique positive solution of 'q' by the fixed-point equation 
    
    % input H: M*K
    % output q: K * 1

    [~, num_user] = size(H);
    
    % in [Desmond '2011], it states that the fixed point solution should be initialized as the positive real numbers
    q = abs(randn(num_user, 1)); % initialization
    % q = randn(num_user, 1);
    iter_max = 100;
    for ii = 1:iter_max
        m = compute_m(H, q);
        tau = compute_tau(H, m, Pt);
        q_old = q;
        for k = 1:num_user
            h_k = H(:, k); % h_k: M*1
            m_k = squeeze(m(k, :, :)); % after squeeze, m_k: num_antenna_bs*num_antenna_bs
            m_inv = inv(m_k); % m must be full-ranked since we added the eye-martix
            q_k = real(h_k' * m_inv * h_k / num_user);
            q(k) = tau / q_k;
        end
        err = norm(q - q_old);
        if err <= 1e-5
            break
        end
    end
end
