function [ W ] = downlink_OLP(H, Pt)
    % Using OLP to find optimal DL beamformer W under the fixed RIS reflection coefficients
    
    % input H: K * M

    % intermediate g_opt: K * M
    % intermediate tau_opt: 1*1 R++ number
    % intermediate p_opt: K * 1

    % output W: M * K

    H = H.'; % H: M * K
    [num_antenna_bs, num_user] = size(H);

    % ------------compute beamforming vector at BS-------------------------
    [g_opt, tau_opt] = compute_beamforming_vec(H, Pt);
    % ------------compute power allocation vector at BS--------------------
    [ p_opt ] = compute_power_allocation(H, g_opt, tau_opt);
    
    % Compute the Beamformer W based on g_opt, p_opt
    W = zeros(num_antenna_bs, num_user);
    for kk = 1:num_user
        W(:, kk) = sqrt(p_opt(kk)/num_user) * g_opt(kk, :);
    end
end