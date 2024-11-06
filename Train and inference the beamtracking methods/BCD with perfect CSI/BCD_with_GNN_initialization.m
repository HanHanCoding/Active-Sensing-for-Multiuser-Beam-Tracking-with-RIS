%% References:
% One need to download the Manopt toolbox from https://www.manopt.org/
% https://github.com/guohuayan/WSR_maximization_for_RIS_system/blob/master/README.md
% https://github.com/taojiang-github/GNN-IRS-Beamforming-Reflection
close all
clear all

figure('Renderer', 'painters', 'Position', [360 150 620 485]);
set(0,'defaulttextInterpreter','latex');
color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0, 0.5, 0];
color4=[1, 0, 0];
color5=[0.4940, 0.1840, 0.5560];
color6=[0.9290, 0.6940, 0.1250]; 
color7=[0.2, 0.3, 0.7]; 
color8=[0, 0, 0]; 
color9=[0.5, 0.5, 0.5];
fs2=14;

%% Loading channel data & Basic parameter setting
addpath('./Maxmin_util/')
load('./channel_data/channel(8, 100, 3)_10.mat');
load('./Rician_channel/Estimated_channel_gain(8, 100, 3)_10.mat');
% Input: channel_bs_user:(num_sample,M,K,total_blocks); %%%%%%%%%%%%%%%%%%%% here, the total_blocks is the moving frames!
% Input: channel_irs_user:(num_sample,N,K,total_blocks); 
% Input: channel_bs_irs(num_sample,M,N,total_blocks)

% Input: H_lmmse_combine_designed_theta = [batch_size_test_1block, num_antenna_bs, 1, num_user, frames]

% Input: theta_noDirect_all = [batch_size_test_1block, num_elements_irs, 1, frames]
theta_noDirect_all = squeeze(theta_noDirect_all); % [batch_size_test_1block, num_elements_irs, frames]
% Input: W_GNN_only = [batch_size_test_1block, num_antenna_bs, num_user, frames]
% Input: raw_SNR_DL: float

% get the parameter settings for the system
[num_sample,M,N,total_blocks] = size(channel_bs_irs);
[~,~,K,~] = size(channel_bs_user); 

% global parameters
etai=1;
path_d = ones(1,K);
weight=1./((path_d)); % assume all the UEs have the equal weights
weight=weight./sum(weight);
% remember, we assume omega = 1 rather than 1/K for all the users.
omega= K * weight; 
snr=raw_SNR_DL; %raw SNR in the downlink! 
Pt=10.^(snr/10);

%% running Proposed Alternating Algorithm for each testing block (1st block at each frame)
% here, w is the Beamformer, theta is the DL RIS reflection coefficients
GNNTheta_LMMSEW_AO_ALLBLOCK = zeros(1,total_blocks);
randTheta_OLPW_ALLBLOCK = zeros(1,total_blocks);

ts = tic();
for block = 1:total_blocks
    % we run Algorithm 2 w.r.t. the channels at each different blocks!
    GNNTheta_LMMSEW_AO = zeros(1,num_sample);
    randTheta_OLPW = zeros(1,num_sample);
    
    it=100;
    parfor t0=1:num_sample
        %--------------------------channel--------------------------------
        % =================== the True channels
        % Notice, in the optimizations, we assume the Hermitian for Channel Reciprocity
        Hd=squeeze(channel_bs_user(t0,:,:,block))'; 
        % set the DL RIS reflection coefficients to the designed optimal v from the trained GNN
        theta_rand=exp(1j.*rand(N,1).*2.*pi); 
        Theta_rand=diag(theta_rand');

        theta_gnn = theta_noDirect_all(t0, :, block); % 1*N
        Theta_gnn=diag(theta_gnn');
        % channel
        G=squeeze(channel_bs_irs(t0,:,:,block))'; 
        Hr=squeeze(channel_irs_user(t0,:,:,block))'; 

        %  =================== the estimated channels (combined with the designed RIS reflection coefficients)
        H_lmmse = squeeze(H_lmmse_combine_designed_theta(t0,:,:,:,block))' % size: (K,M)
        
        % -----------------------------------------------------------------------------
        % -------------Initialize W with LMMSE estimated low-dim channel; theta using GNN designed results, then apply OLP+RCG---------------------
        Theta_AO=diag(theta_gnn');
        H_AO = Hd+Hr*Theta_AO*G; % K * M;
        % running the OLP Algorithm for each testing block (1st block at each frame) to get the designed beamformer
        [ W_AO ] = downlink_OLP(conj(H_lmmse), Pt); % M * K
        for con0=1:it
            % update the DL RIS reflection coefficients using RCG, initilized by theta_gnn
            [theta_AO, ~, ~] = man_opt_theta_direct( W_AO,Hd,Hr,Theta_AO,G,N,K,omega );

            % Update the combined 'H' and compute the new downlink rate
            Theta_AO=diag(theta_AO');
            H_AO=Hd+Hr*Theta_AO*G;

            % update the DL beamformer W using OLP
            [ W_AO ] = downlink_OLP(conj(H_AO), Pt); % M * K
        end
        [ f2, ~ ] = update_SINR_test( H_AO, W_AO );

        % -------------Initialize W,theta randomly, then apply OLP+RCG---------------------
        H_rand = Hd+Hr*Theta_rand*G; % K * M;
        % init for the above loop ----> if we do not init, 'W_rand' will be cleared at the beginning of each iteration of the parfor-loop
        % update the DL beamformer W using OLP, initilized randomly
        [ W_rand ] = downlink_OLP(conj(H_rand), Pt); % M * K
        % record the result that only optimize Beamformer, but apply random RIS reflection coefficients ---> to see how much gain can the RIS bring to the system!
        [ f5, ~ ] = update_SINR_test( H_rand, W_rand );
        
        % --------------------------Record the Results------------------------------------
        GNNTheta_LMMSEW_AO(t0)=f2;
        randTheta_OLPW(t0)=f5;
    end
    % record the result that computed by assuming Hermitian!
    GNNTheta_LMMSEW_AO_ALLBLOCK(block) = mean(GNNTheta_LMMSEW_AO); % this is the performance upper bound
    randTheta_OLPW_ALLBLOCK(block) = mean(randTheta_OLPW); % this is the performance lower bound

    % monitor the program
    fprintf('finished the computing for the %dth block\n', block)
    % record the running time
    running_time = toc(ts)
end
%% plot
N_w_using = 3;
frames = 30;

figure(1)
tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact'); 
block = 1:1:frames;
plot(block,GNNTheta_LMMSEW_AO_ALLBLOCK(1:length(block)),'-','color',color4,'lineWidth',1.7,'markersize',4);
hold on
plot(block,randTheta_OLPW_ALLBLOCK(1:length(block)),'-','color',color5,'lineWidth',1.7,'markersize',4);
grid on

lg=legend('Alternating optimization w/ full CSI (initialized by the GNN solution)',...
          'Random $\mathbf{w}^{(t)}$ w/ optimal linear precoding under full CSI',...
          'Location','southeast');
set(lg,'Fontsize',fs2-3);
set(lg,'Interpreter','latex');
xlabel('Transmission frame','Interpreter','latex','FontSize',fs2);
% xlabel('Block \#','Interpreter','latex','FontSize',fs2);
ylabel('Minimum user rate (bps/Hz)','Interpreter','latex','FontSize',fs2);
% xlim([1,frames])
% ylim([2.5,5.5])
% title('$M = 8$, $N_r = 100$, $K = 3$, Perfect CSI')
 

str = sprintf('./results_data/results.mat');
save(str,'GNNTheta_LMMSEW_AO_ALLBLOCK','randTheta_OLPW_ALLBLOCK');
