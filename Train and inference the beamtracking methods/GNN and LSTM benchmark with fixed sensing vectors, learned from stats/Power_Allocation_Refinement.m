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
load('./Rician_channel/channel_Test(8, 100, 3, 10, 3000, 10, 100, 90).mat');
load('./Rician_channel/Estimated_channel_gain(8, 100, 3)_10.mat');
% Input: channel_bs_user:(num_sample,M,K,total_blocks); 
% Input: channel_irs_user:(num_sample,N,K,total_blocks); 
% Input: channel_bs_irs(num_sample,M,N,total_blocks)

% Input: H_lmmse_combine_designed_theta = [batch_size_test_1block, num_antenna_bs, 1, num_user, frames]
% Input: theta_noDirect_all = [batch_size_test_1block, num_elements_irs, 1, frames]
% Input: W_GNN_only = [batch_size_test_1block, num_antenna_bs, num_user, frames]

% Input: raw_SNR_DL: float

% get the parameter settings for the system
[num_sample,M,N,total_blocks] = size(channel_bs_irs);
[~,~,K,~] = size(channel_bs_user); 
[~, ~, ~, frames] = size(theta_noDirect_all);

N_w_using = total_blocks/frames;
theta_noDirect_all = squeeze(theta_noDirect_all); % [batch_size_test_1block, num_elements_irs, frames]

% global parameters
etai=1;
path_d = ones(1,K);
weight=1./((path_d)); % assume all the UEs have the equal weights
weight=weight./sum(weight);
% remember, we assume omega = 1 rather than 1/K for all the users.
omega= K * weight; 
snr=raw_SNR_DL; %raw SNR in the downlink! 
Pt=10.^(snr/10);

%% running the Linear Precoding to refine the designed Beamformer W! for each testing block (1st block at each frame) to get the designed beamformer
%-----------apply OLP to compute the designed beamformer W---------------------
W_refined_all = zeros(num_sample, M, K, frames);
parfor t0=1:num_sample
    for frame = 1:frames
        % channel
        H_lmmse = squeeze(H_lmmse_combine_designed_theta(t0,:,:,:,frame)).' % size: (K,M) % this is normal Transpose

        % compute beamformer W for this transmission frame
        [ W_refined ] = downlink_OLP(H_lmmse, Pt); % M * K
        W_refined_all(t0,:,:,frame) = W_refined;
    end
end


%----------------------- Compute the downlink rate ---------------------------
ts = tic();
% here, w is the Beamformer, theta is the DL RIS reflection coefficients
GNNTheta_GNNW_RATE_ALLBLOCK = zeros(1,total_blocks);
GNNTheta_refinedW_RATE_ALLBLOCK = zeros(1,total_blocks);


for block = 1:total_blocks
    GNNTheta_refinedW_RATE = zeros(1,num_sample);
    GNNTheta_GNNW_RATE = zeros(1,num_sample);
    
    for t0=1:num_sample
        % the True downlink channels (here since we assume Hermitian as the UL-DL reciprocity)
        Hd=squeeze(channel_bs_user(t0,:,:,block))'; 
        G=squeeze(channel_bs_irs(t0,:,:,block))'; 
        Hr=squeeze(channel_irs_user(t0,:,:,block))'; 

        % DL RIS reflection coefficients
        theta_GNN_only = theta_noDirect_all(t0, :, floor((block-1)/N_w_using)+1); % 1*N
        Theta_GNN_only=diag(theta_GNN_only');

        % DL beamformer
        W_refined_block = squeeze(W_refined_all(t0,:,:,floor((block-1)/N_w_using)+1)); % M * K
        W_GNN_only_block = squeeze(W_GNN_only(t0,:,:,floor((block-1)/N_w_using)+1)); % M * K


        % Compute the downlink min rate 
        H_true_combined = Hd+Hr*Theta_GNN_only*G;
        [ min_rate_with_refining, ~ ] = update_SINR_test( H_true_combined, W_refined_block );
        [ min_rate_GNN_only, ~ ] = update_SINR_test( H_true_combined, W_GNN_only_block );
        
        % --------------------------Record the Results------------------------------------
        GNNTheta_refinedW_RATE(t0)=min_rate_with_refining;
        GNNTheta_GNNW_RATE(t0)=min_rate_GNN_only;
    end
    % record the result that computed by assuming Hermitian!
    GNNTheta_refinedW_RATE_ALLBLOCK(block) = mean(GNNTheta_refinedW_RATE);
    GNNTheta_GNNW_RATE_ALLBLOCK(block) = mean(GNNTheta_GNNW_RATE);
    
end
% record the running time
fprintf('finished the computing for the %dth block\n', block)
running_time = toc(ts)

%% plot
N_w_using = 3;
frames = 30;
blocks_total = N_w_using*frames;

figure(1)
tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact'); 
block = 1:1:blocks_total;
p1 = plot(block,GNNTheta_refinedW_RATE_ALLBLOCK(1:length(block)),'-o','color',color1,'lineWidth',1.7,'markersize',5);
p1.MarkerIndices = 1:3:length(block);
hold on
p2 = plot(block,GNNTheta_GNNW_RATE_ALLBLOCK(1:length(block)),'-square','color',color3,'lineWidth',1.7,'markersize',5);
p2.MarkerIndices = 1:3:length(block);
grid on

lg=legend('GNN designs $\mathbf{w}^{(t)}$, $\mathbf{p}^{(t)}$, and $\mathbf{\lambda}^{(t)}$; w/ refining on beamformer',...
    'GNN designs $\mathbf{w}^{(t)}$ and $\mathbf{B}^{(t)}$, where $\mathbf{B}^{(t)}$ = TMMSE($\mathbf{p}^{(t)}$)',...
    'Location','southeast');
set(lg,'Fontsize',fs2-3);
set(lg,'Interpreter','latex');
xlabel('Transmission frame','Interpreter','latex','FontSize',fs2);
% xlabel('Block \#','Interpreter','latex','FontSize',fs2);
ylabel('Minimum user rate (bps/Hz)','Interpreter','latex','FontSize',fs2);
xlim([1,90])
ylim([2.5,6])

xticks([1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,73,76,79,82,85,88])
xticklabels({'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23', '24', '25', '26', '27', '28', '29', '30'}); set(lg,'Interpreter','latex')
 

str = sprintf('./results_data/results.mat');
save(str,'GNNTheta_refinedW_RATE_ALLBLOCK','GNNTheta_GNNW_RATE_ALLBLOCK');

str = sprintf('./results_data/LSTMbasedBenchmark_learnedV_LS_L3_ULnoise70_tauw1000_LSTM21_NoIncrease30_block3_moving02.mat');
save(str,'GNNTheta_refinedW_RATE_ALLBLOCK');
 