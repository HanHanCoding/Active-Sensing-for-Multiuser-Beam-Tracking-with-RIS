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
% Since we have 3 Sensing vectors in total
Array_response_V_range_all_1 = importdata('./Generate_MatFile_forVisualization/Array_response_V_range_all_1.mat');
Array_response_V_range_all_2 = importdata('./Generate_MatFile_forVisualization/Array_response_V_range_all_2.mat');
Array_response_V_range_all_3 = importdata('./Generate_MatFile_forVisualization/Array_response_V_range_all_3.mat');

% % Array Response for the UL RIS sensing vector
% Array_response_V_1 = zeros(sample_x, sample_y, total_frames);
% Array_response_V_2 = zeros(sample_x, sample_y, total_frames);
% Array_response_V_3 = zeros(sample_x, sample_y, total_frames);
% for kk = 1:total_frames
%     % for x
%     for ii = 1:sample_x
%         % for y
%         for jj = 1:sample_y
%             Array_response_V_1(ii, jj, kk) = Array_response_V_range_all_1(sample_y*(ii-1)+jj, 1, kk);
%             Array_response_V_2(ii, jj, kk) = Array_response_V_range_all_2(sample_y*(ii-1)+jj, 1, kk);
%             Array_response_V_3(ii, jj, kk) = Array_response_V_range_all_3(sample_y*(ii-1)+jj, 1, kk);
%         end
%     end
% end

% % focusing to the users, not the entire plane!
% Array_response_V_1 = zeros(13, 12, total_frames);
% Array_response_V_2 = zeros(13, 12, total_frames);
% Array_response_V_3 = zeros(13, 12, total_frames);
% for kk = 1:total_frames
%     % for x
%     for ii = 6:18
%         % for y
%         for jj = 8:19
%             Array_response_V_1(ii-5, jj-7, kk) = Array_response_V_range_all_1(sample_y*(ii-1)+jj, 1, kk);
%             Array_response_V_2(ii-5, jj-7, kk) = Array_response_V_range_all_2(sample_y*(ii-1)+jj, 1, kk);
%             Array_response_V_3(ii-5, jj-7, kk) = Array_response_V_range_all_3(sample_y*(ii-1)+jj, 1, kk);
%         end
%     end
% end
% range_coordinate_all_x = range_coordinate_all_x(6:18,:);
% range_coordinate_all_y = range_coordinate_all_y(8:19,:);

Array_response_V_1 = zeros(19, 14, total_frames);
Array_response_V_2 = zeros(19, 14, total_frames);
Array_response_V_3 = zeros(19, 14, total_frames);
for kk = 1:total_frames
    % for x
    for ii = 3:21
        % for y
        for jj = 7:20
            Array_response_V_1(ii-2, jj-6, kk) = Array_response_V_range_all_1(sample_y*(ii-1)+jj, 1, kk);
            Array_response_V_2(ii-2, jj-6, kk) = Array_response_V_range_all_2(sample_y*(ii-1)+jj, 1, kk);
            Array_response_V_3(ii-2, jj-6, kk) = Array_response_V_range_all_3(sample_y*(ii-1)+jj, 1, kk);
        end
    end
end
range_coordinate_all_x = range_coordinate_all_x(3:21,:);
range_coordinate_all_y = range_coordinate_all_y(7:20,:);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Visualize the array response for each sensing vectors!
%% plot the rate: frame[1, 18, 36]
figure(2)
tiledlayout(1,3, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_1(:,:,1).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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

% frame = 8
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_1(:,:,8).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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

% frame = 36
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_1(:,:,36).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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

%% plot the rate: frame[1, 18, 36]
figure(3)
tiledlayout(1,3, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_2(:,:,1).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


% frame = 8
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_2(:,:,8).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


% frame = 36
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_2(:,:,36).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


%% plot the rate: frame[1, 18, 36]
figure(4)
tiledlayout(1,3, 'Padding', 'none', 'TileSpacing', 'compact');  

% frame = 1
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_3(:,:,1).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


% frame = 8
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_3(:,:,8).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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


% frame = 36
nexttile
pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_3(:,:,36).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
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
% 
% % frame = 35
% nexttile
% pcolor(range_coordinate_all_x, range_coordinate_all_y, Array_response_V_3(:,:,35).') %%%%%%%%%%%%%%%%% notice! transpose the rate matrix here!
% colorbar
% shading interp
% hold on
% plot(user_coor_x(35,1,1), user_coor_y(35,1,1), 'o', 'MarkerSize', 5, 'MarkerFaceColor','r')
% plot(user_coor_x(35,1,2), user_coor_y(35,1,2), 'square', 'MarkerSize', 7, 'MarkerFaceColor','g')
% plot(user_coor_x(35,1,3), user_coor_y(35,1,3), 'pentagram', 'MarkerSize', 7, 'MarkerFaceColor','b')
% 
% % xlim([5,45])
% % ylim([-35,35])
% 
% xlabel('$x$-coordinate (m)','Interpreter','latex','FontSize',fs2)
% title('$35$th transmission frame','Interpreter','latex','FontSize',fs2)


%% SAVE [left bottom width height]
f2 = figure(2);  
f2.Position = [30 30 1500*3/7 225]; 
set(f2, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f2,'./Generate_MatFile_forVisualization/AR_RIS_SensingVector_1.pdf')

f3 = figure(3);  
f3.Position = [30 30 1500*3/7 225]; 
set(f3, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f3,'./Generate_MatFile_forVisualization/AR_RIS_SensingVector_2.pdf')

f4 = figure(4);  
f4.Position = [30 30 1500*3/7 225]; 
set(f4, 'PaperSize', [30 30])    % Same, but for PDF output
saveas(f4,'./Generate_MatFile_forVisualization/AR_RIS_SensingVector_3.pdf')