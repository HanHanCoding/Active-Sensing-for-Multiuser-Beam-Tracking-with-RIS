function [ min_rate, rate_all ] = MinRate_Transpose_version( Hd, Hr, G, W, theta)
    % this function is used for computing the min rate under transpose not Hermitian
    
    % input Hd: (1, M, K) -------> channel_bs_user
    % input Hr: (1, N, K) -------> channel_irs_user
    % input G: (1, M, N) -------> channel_bs_irs
    % input W: (M, K)
    % input theta: (N, 1)

    % output min_rate: real number

    Hd = squeeze(Hd); % M, K
    Hr = squeeze(Hr); % N, K
    G = squeeze(G); % M, N

    [~, K] = size(Hd);
    
    signal_power_all = zeros(1,K);
    rate_all = zeros(1,K);
    for k1 = 1:K
        channel_gain_k1 = Hd(:,k1) + G * diag(theta) * Hr(:,k1); % M,1
        for k2 = 1:K
            % here, we use regular transpose rather than Hermitian to compute the signal power
            signal_power_all(k2) = abs(channel_gain_k1.' * W(:,k2))^2;
        end
        SINR_k1 = signal_power_all(k1)/(sum(signal_power_all) - signal_power_all(k1) + 1); % we have normalized the noise power to 1!
        rate_all(k1) = log2(1+SINR_k1);
    end

    min_rate = min(rate_all);

end

