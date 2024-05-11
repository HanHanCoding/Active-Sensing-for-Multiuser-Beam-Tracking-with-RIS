clear
fs2=10;

sample_x = 25;
sample_y = 25;

user_coor_x = importdata('./Generate_MatFile_forVisualization/user_coor_x.mat');
user_coor_y = importdata('./Generate_MatFile_forVisualization/user_coor_y.mat');
range_coordinate_all_x = importdata('./Generate_MatFile_forVisualization/range_coordinate_x.mat');
range_coordinate_all_y = importdata('./Generate_MatFile_forVisualization/range_coordinate_y.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check the user path
N_w_using = 1;
total_w_number = 40;
total_frames = N_w_using * total_w_number;
z = 1:1:total_frames;

figure(1)
% plot3(user_coor_x,user_coor_y,z,'-o','Color','b','MarkerSize',10,'MarkerFaceColor','#D9FFFF')
% xlabel('User coordinate - x coordinate (m)','Interpreter','latex','FontSize',fs2+2)
% ylabel('User coordinate - y coordinate (m)','Interpreter','latex','FontSize',fs2+2)
% zlabel('time frames')
% grid on
plot(user_coor_x(1:30,:,1),user_coor_y(1:30,:,1),'-o','Color','r','MarkerSize',10,'MarkerFaceColor','r')
hold on
plot(user_coor_x(1:30,:,2),user_coor_y(1:30,:,2),'-square','Color','g','MarkerSize',10,'MarkerFaceColor','g')
plot(user_coor_x(1:30,:,3),user_coor_y(1:30,:,3),'-pentagram','Color','b','MarkerSize',10,'MarkerFaceColor','b')
xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2+2)
ylabel('$y$-coordinate (m)','Interpreter','latex','FontSize',fs2+2)
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% change the dimension and prepare for the rate plot
Array_response_F = importdata('./Generate_MatFile_forVisualization/Array_response_F.mat'); % [N_angles, num_user, frames]
Array_response_w_range_all = importdata('./Generate_MatFile_forVisualization/Array_response_w_range_all.mat');

% % Array Response for the DL RIS reflection coefficients
% Array_response_w = zeros(sample_x, sample_y, total_frames);
% for kk = 1:total_frames
%     % for x
%     for ii = 1:sample_x
%         % for y
%         for jj = 1:sample_y
%             Array_response_w(ii, jj, kk) = Array_response_w_range_all(sample_y*(ii-1)+jj, 1, kk);
%         end
%     end
% end

% focusing to the users, not the entire plane!
Array_response_w = zeros(13, 12, total_frames);
for kk = 1:total_frames
    % for x
    for ii = 6:18
        % for y
        for jj = 8:19
            Array_response_w(ii-5, jj-7, kk) = Array_response_w_range_all(sample_y*(ii-1)+jj, 1, kk);
        end
    end
end
range_coordinate_all_x = range_coordinate_all_x(6:18,:);
range_coordinate_all_y = range_coordinate_all_y(8:19,:);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Visualize the Array Response for the DL RIS reflection coefficients
%plot the rate: frame[1, 2, 4, 8, 12, 24, 36]
figure(4)
tiledlayout(1,7, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_w(:,:,1).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
colorbar
shading interp
hold on
plot(user_coor_x(1,1,1), user_coor_y(1,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
plot(user_coor_x(1,1,2), user_coor_y(1,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
plot(user_coor_x(1,1,3), user_coor_y(1,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')

% xlim([5,45])
% ylim([-35,35])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
ylabel('$y$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$1$st transmission frame','Interpreter','latex','FontSize',fs2)
% 

% frame = 2
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_w(:,:,2).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
colorbar
shading interp
hold on
plot(user_coor_x(2,1,1), user_coor_y(2,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
plot(user_coor_x(2,1,2), user_coor_y(2,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
plot(user_coor_x(2,1,3), user_coor_y(2,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')

% xlim([5,45])
% ylim([-35,35])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$2$nd transmission frame','Interpreter','latex','FontSize',fs2)

% 

% frame = 4
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_w(:,:,4).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
colorbar
shading interp
hold on
plot(user_coor_x(4,1,1), user_coor_y(4,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
plot(user_coor_x(4,1,2), user_coor_y(4,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
plot(user_coor_x(4,1,3), user_coor_y(4,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')

% xlim([5,45])
% ylim([-35,35])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$4$th transmission frame','Interpreter','latex','FontSize',fs2)

% 
% frame = 8
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_w(:,:,8).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
colorbar
shading interp
hold on
plot(user_coor_x(8,1,1), user_coor_y(8,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
plot(user_coor_x(8,1,2), user_coor_y(8,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
plot(user_coor_x(8,1,3), user_coor_y(8,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')

% xlim([5,45])
% ylim([-35,35])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$8$th transmission frame','Interpreter','latex','FontSize',fs2)

% frame = 12
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_w(:,:,12).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
colorbar
shading interp
hold on
plot(user_coor_x(12,1,1), user_coor_y(12,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
plot(user_coor_x(12,1,2), user_coor_y(12,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
plot(user_coor_x(12,1,3), user_coor_y(12,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')

% xlim([5,45])
% ylim([-35,35])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$12$th transmission frame','Interpreter','latex','FontSize',fs2)

% frame = 24
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_w(:,:,24).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
colorbar
shading interp
hold on
plot(user_coor_x(24,1,1), user_coor_y(24,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
plot(user_coor_x(24,1,2), user_coor_y(24,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
plot(user_coor_x(24,1,3), user_coor_y(24,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')

% xlim([5,45])
% ylim([-35,35])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$24$th transmission frame','Interpreter','latex','FontSize',fs2)

% frame = 36
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_w(:,:,36).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
colorbar
shading interp
hold on
plot(user_coor_x(36,1,1), user_coor_y(36,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
plot(user_coor_x(36,1,2), user_coor_y(36,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
plot(user_coor_x(36,1,3), user_coor_y(36,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')

% xlim([5,45])
% ylim([-35,35])

xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
title('$36$th transmission frame','Interpreter','latex','FontSize',fs2)




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Visualize the Array Response for the DL beamformer F
%plot the rate: frame[1, 2, 4, 8, 12, 24, 36]
figure(5)
tiledlayout(1,7, 'Padding', 'none', 'TileSpacing', 'compact');  
N_angles = 1000;
effective_AOA = linspace(0, pi, N_angles);

% Array_response_F = importdata('./Generate_MatFile_forVisualization/Array_response_F.mat'); % [N_angles, num_user, frames]

% frame = 1
nexttile
plot(effective_AOA, Array_response_F(:,1,1),'-','color','r','lineWidth',1.7,'markersize',4)
hold on
plot(effective_AOA, Array_response_F(:,2,1),'-','color','g','lineWidth',1.7,'markersize',4)
plot(effective_AOA, Array_response_F(:,3,1),'-','color','b','lineWidth',1.7,'markersize',4)
grid on
legend('User 1', 'User 2', 'User 3');
xlabel('$\phi_1$ (rad)','Interpreter','latex','FontSize',fs2)
ylabel('Array response','Interpreter','latex','FontSize',fs2)
title('$1$st transmission frame','Interpreter','latex','FontSize',fs2)
% ylim([0, 2.1])

% 
% frame = 2
nexttile
plot(effective_AOA, Array_response_F(:,1,2),'-','color','r','lineWidth',1.7,'markersize',4)
hold on
plot(effective_AOA, Array_response_F(:,2,2),'-','color','g','lineWidth',1.7,'markersize',4)
plot(effective_AOA, Array_response_F(:,3,2),'-','color','b','lineWidth',1.7,'markersize',4)
grid on
% legend('User 1', 'User 2', 'User 3')
xlabel('$\phi_1$ (rad)','Interpreter','latex','FontSize',fs2)
title('$2$nd transmission frame','Interpreter','latex','FontSize',fs2)
% ylim([0, 2.1])

% 
% frame = 4
nexttile
plot(effective_AOA, Array_response_F(:,1,4),'-','color','r','lineWidth',1.7,'markersize',4)
hold on
plot(effective_AOA, Array_response_F(:,2,4),'-','color','g','lineWidth',1.7,'markersize',4)
plot(effective_AOA, Array_response_F(:,3,4),'-','color','b','lineWidth',1.7,'markersize',4)
grid on
% legend('User 1', 'User 2', 'User 3')
xlabel('$\phi_1$ (rad)','Interpreter','latex','FontSize',fs2)
title('$4$th transmission frame','Interpreter','latex','FontSize',fs2)
% ylim([0, 2.1])


% frame = 8
nexttile
plot(effective_AOA, Array_response_F(:,1,8),'-','color','r','lineWidth',1.7,'markersize',4)
hold on
plot(effective_AOA, Array_response_F(:,2,8),'-','color','g','lineWidth',1.7,'markersize',4)
plot(effective_AOA, Array_response_F(:,3,8),'-','color','b','lineWidth',1.7,'markersize',4)
grid on
% legend('User 1', 'User 2', 'User 3')
xlabel('$\phi_1$ (rad)','Interpreter','latex','FontSize',fs2)
title('$8$th transmission frame','Interpreter','latex','FontSize',fs2)
% ylim([0, 2.1])

% frame = 12
nexttile
plot(effective_AOA, Array_response_F(:,1,12),'-','color','r','lineWidth',1.7,'markersize',4)
hold on
plot(effective_AOA, Array_response_F(:,2,12),'-','color','g','lineWidth',1.7,'markersize',4)
plot(effective_AOA, Array_response_F(:,3,12),'-','color','b','lineWidth',1.7,'markersize',4)
grid on
% legend('User 1', 'User 2', 'User 3')
xlabel('$\phi_1$ (rad)','Interpreter','latex','FontSize',fs2)
title('$12$th transmission frame','Interpreter','latex','FontSize',fs2)
% ylim([0, 2.1])

% frame = 24
nexttile
plot(effective_AOA, Array_response_F(:,1,24),'-','color','r','lineWidth',1.7,'markersize',4)
hold on
plot(effective_AOA, Array_response_F(:,2,24),'-','color','g','lineWidth',1.7,'markersize',4)
plot(effective_AOA, Array_response_F(:,3,24),'-','color','b','lineWidth',1.7,'markersize',4)
grid on
% legend('User 1', 'User 2', 'User 3')
xlabel('$\phi_1$ (rad)','Interpreter','latex','FontSize',fs2)
title('$24$th transmission frame','Interpreter','latex','FontSize',fs2)
% ylim([0, 2.1])

% frame = 36
nexttile
plot(effective_AOA, Array_response_F(:,1,36),'-','color','r','lineWidth',1.7,'markersize',4)
hold on
plot(effective_AOA, Array_response_F(:,2,36),'-','color','g','lineWidth',1.7,'markersize',4)
plot(effective_AOA, Array_response_F(:,3,36),'-','color','b','lineWidth',1.7,'markersize',4)
grid on
% legend('User 1', 'User 2', 'User 3')
xlabel('$\phi_1$ (rad)','Interpreter','latex','FontSize',fs2)
title('$36$th transmission frame','Interpreter','latex','FontSize',fs2)
% ylim([0, 2.1])


%% SAVE
% [left bottom width height]

f3 = figure(4);  
f3.Position = [30 30 1500 225]; 
set(f3, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f3,'./Generate_MatFile_forVisualization/AR_RIS_DLdata_MU.pdf')

f4 = figure(5);  
f4.Position = [30 30 1500 225]; 
set(f4, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f4,'./Generate_MatFile_forVisualization/AR_Beamformer.pdf')