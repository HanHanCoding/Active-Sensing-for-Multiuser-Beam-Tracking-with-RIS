function [ p_opt ] = compute_power_allocation(H, g_opt, tau_opt)
    % this function is used for computing the optimal power allocation vector p
    
    % input H: M * K
    % input g_opt: K * M
    % input tau_opt: 1*1 R++ number

    % output p_opt: K * 1
    
    [~, num_user] = size(H);
    g_opt = g_opt.'; % g_opt: M * K

    D = zeros(num_user, num_user);
    F = zeros(num_user, num_user);
    for k = 1:num_user
        h_k = H(:, k); % h_k: M*1
        g_k = g_opt(:, k); % g_k: M*1
        D(k, k) = 1 / (abs(h_k' * g_k)^2 / num_user);
        for ii = 1:num_user
            if ii ~= k
                g_i = g_opt(:, ii); % g_i: M*1
                F(k, ii) = abs(h_k' * g_i)^2 / num_user;
            end
        end
    end
    tmp = eye(num_user) - tau_opt * (D * F);
    tmp_inv = inv(tmp);
    p_opt = tmp_inv * (D * ones(num_user, 1)) * tau_opt;
end
