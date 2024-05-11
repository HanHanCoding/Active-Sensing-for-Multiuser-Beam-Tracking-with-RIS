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
SNR_SINR_table_allFrame = importdata('./Generate_MatFile_forVisualization/SNR_SINR_table_allFrame.mat');


% Since we have 3 UEs in total
SINR_perLocation_allFrame_UE_all = importdata('./Generate_MatFile_forVisualization/SINR_perLocation_allFrame_UE_all.mat');

SINR_perLocation_allFrame_UE_1 = reshape(SINR_perLocation_allFrame_UE_all(:,:,1,:), [size(SINR_perLocation_allFrame_UE_all, 1), size(SINR_perLocation_allFrame_UE_all, 2), size(SINR_perLocation_allFrame_UE_all, 4)]);
SINR_perLocation_allFrame_UE_2 = reshape(SINR_perLocation_allFrame_UE_all(:,:,2,:), [size(SINR_perLocation_allFrame_UE_all, 1), size(SINR_perLocation_allFrame_UE_all, 2), size(SINR_perLocation_allFrame_UE_all, 4)]);
SINR_perLocation_allFrame_UE_3 = reshape(SINR_perLocation_allFrame_UE_all(:,:,3,:), [size(SINR_perLocation_allFrame_UE_all, 1), size(SINR_perLocation_allFrame_UE_all, 2), size(SINR_perLocation_allFrame_UE_all, 4)]);

% Array_response_V_range_all_1 = importdata('./Generate_MatFile_forVisualization/Array_response_V_range_all_1.mat');
% Array_response_V_range_all_2 = importdata('./Generate_MatFile_forVisualization/Array_response_V_range_all_2.mat');
% Array_response_V_range_all_3 = importdata('./Generate_MatFile_forVisualization/Array_response_V_range_all_3.mat');


% focusing to the users, not the entire plane!
SINR_UE_1 = zeros(13, 12, total_frames);
SINR_UE_2 = zeros(13, 12, total_frames);
SINR_UE_3 = zeros(13, 12, total_frames);
for kk = 1:total_frames
    % for x
    for ii = 6:18
        % for y
        for jj = 8:19
            SINR_UE_1(ii-5, jj-7, kk) = SINR_perLocation_allFrame_UE_1(sample_y*(ii-1)+jj, 1, kk);
            SINR_UE_2(ii-5, jj-7, kk) = SINR_perLocation_allFrame_UE_2(sample_y*(ii-1)+jj, 1, kk);
            SINR_UE_3(ii-5, jj-7, kk) = SINR_perLocation_allFrame_UE_3(sample_y*(ii-1)+jj, 1, kk);
        end
    end
end
range_coordinate_all_x = range_coordinate_all_x(6:18,:);
range_coordinate_all_y = range_coordinate_all_y(8:19,:);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Visualize the array response for each sensing vectors!
%% plot the rate: frame[1, 2, 4, 8, 12, 24, 36]
figure(2)
tiledlayout(1,7, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_1(:,:,1).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_1(:,:,2).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_1(:,:,4).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


% frame = 8
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_1(:,:,8).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_1(:,:,12).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_1(:,:,24).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_1(:,:,36).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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

%% plot the rate: frame[1, 2, 4, 8, 12, 24, 36]
figure(3)
tiledlayout(1,7, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_2(:,:,1).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_2(:,:,2).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_2(:,:,4).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


% frame = 8
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_2(:,:,8).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_2(:,:,12).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_2(:,:,24).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_2(:,:,36).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


%% plot the rate: frame[1, 2, 4, 8, 12, 24, 36]
figure(4)
tiledlayout(1,7, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_3(:,:,1).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_3(:,:,2).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_3(:,:,4).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


% frame = 8
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_3(:,:,8).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_3(:,:,12).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_3(:,:,24).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
pcolor(range_coordinate_all_x, range_coordinate_all_y, SINR_UE_3(:,:,36).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


%% SAVE
f2 = figure(2);  
f2.Position = [30 30 1500 225]; 
set(f2, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f2,'./Generate_MatFile_forVisualization/SINR_UE_1.pdf')

f3 = figure(3);  
f3.Position = [30 30 1500 225]; 
set(f3, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f3,'./Generate_MatFile_forVisualization/SINR_UE_2.pdf')

f4 = figure(4);  
f4.Position = [30 30 1500 225]; 
set(f4, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f4,'./Generate_MatFile_forVisualization/SINR_UE_3.pdf')