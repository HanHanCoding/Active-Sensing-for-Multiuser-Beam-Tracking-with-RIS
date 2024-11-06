function [ m ] = compute_m(H, q)
    % this function is used for compute the common part in denominator for all the equations!
    % we have normalized the noise power to 1!

    % input H: M*K
    % input q: K * 1
    
    % output m: (num_user, num_antenna_bs, num_antenna_bs)

    [num_antenna_bs, num_user] = size(H);

    m = zeros(num_user, num_antenna_bs, num_antenna_bs);
    for k = 1:num_user
        tmp = eye(num_antenna_bs);
        for ii = 1:num_user
            if ii ~= k
                h_i = H(:, ii); % h_i: M*1
                hh = h_i * h_i';
                tmp = (q(ii) / num_user) * hh + tmp;
            end
        end
        m(k, :, :) = tmp;
    end
end
