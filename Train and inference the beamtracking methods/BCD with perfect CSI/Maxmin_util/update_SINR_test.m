function [ min_rate, rate_all ] = update_SINR_test( H,W )
    % this function is used for computing the min rate under Hermitian
    
    % input H: (K * M) -------> H=Hd+Hr*Theta*G, already Hermitianed!
    % input W: (M, K)

    % output min_rate: real number


    [K, ~] = size(H);
    
    signal_power_all = zeros(1,K);
    rate_all = zeros(1,K);
    for k1 = 1:K
        channel_gain_k1 = H(k1, :); % 1,M
        for k2 = 1:K
            signal_power_all(k2) = abs(channel_gain_k1 * W(:,k2))^2;
        end
        SINR_k1 = signal_power_all(k1)/(sum(signal_power_all) - signal_power_all(k1) + 1); % we have normalized the noise power to 1!
        rate_all(k1) = log2(1+SINR_k1);
    end

    min_rate = min(rate_all);
end

