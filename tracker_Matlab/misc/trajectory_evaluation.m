function [nt, sp, np] = trajectory_evaluation(groundtruth)

for gg = 1:size(groundtruth, 2)
  groundtruth(gg).pos = groundtruth(gg).pos / 4.8;
  groundtruth(gg).pos(2, :) = 120 - groundtruth(gg).pos(2, :);
end

fprintf(1, 'Loading files...\n');

files = dir(['./*.mat']);
files = sort({files.name});

% Now read in every file

% Number of training files
N = size(files, 2);

for nn = 1:20
  fprintf(1, 'Reading file %d: %s\n', nn, files{nn});
  load(files{nn});

  tr = extract_trajectories(object_hist);
[labels, errpos, new_assoc_pos, switch_percent, nupdated_percent, avg_distance] = tracker_confusion(tr, groundtruth);

  nt(nn) = size(switch_percent, 1);
  sp(nn).sp = switch_percent;
  np(nn).np = nupdated_percent;

%whos

% Some random error function
error = switch_percent + nupdated_percent + avg_distance;

[switch_percent, (1 - nupdated_percent), avg_distance]

figure(1);
hold on;
%scatter(ones(size(switch_percent, 1), 1) * nn, switch_percent + randn(size(switch_percent, 1), 1) * 0.2, '+');
scatter(ones(size(error, 1), 1) * nn, error, '+');
hold off;
figure(2);
hold on;
scatter(ones(size(nupdated_percent, 1), 1) * nn, nupdated_percent + randn(size(nupdated_percent, 1), 1) * 0.2, '+');
hold off;

ntimes_propagated = cat(1, tr.frame_destroyed) - cat(1,tr.frame_created) - cat(1, tr.ntimes_updated);

figure(3);
hold on;
scatter(ones(size(nupdated_percent, 1), 1) * nn, switch_percent + ntimes_propagated + randn(size(nupdated_percent, 1), 1) * 0.2, '+');
hold off;
drawnow;

%figure(4);
%switch_percent
%plot_trajectories(zeros(120, 160, 3), tr(find(switch_percent <= 0.008)), 0);
%figure(5);
%plot_trajectories(zeros(120, 160, 3), tr(find(switch_percent > 0.008)), 0);




%  sp_sorted = sort(switch_percent, 'ascend');
%  np_sorted = sort(nupdated_percent, 'descend');

%  if(size(switch_percent, 1) >= 15)
%    nt(nn) = size(switch_percent, 1);
%    sp(:, nn) = sp_sorted(1:15);
%    np(:, nn) = np_sorted(1:15);
%  end
end

%function [labels, switch_percent, nupdated_percent, errpos] = ...
%      trajectory_evaluation(object_hist, groundtruth);
%[labels, switch_percent, nupdated_percent, errpos] = ...
%      trajectory_evaluation(object_hist, groundtruth);
% Wrapper script to evaluate the trajectories using the other helper functions. 

% Scale groundtruth

% Use the background image from the first object as the background
% image. 

%% bg = object_hist(1).background;

%% subplot(2, 1, 1);
%% plot_trajectories(bg, trajectories, 0);
%% % Add error positions
%% hold on
%% plot(errpos(1, :), errpos(2, :), '+y');
%% plot(new_assoc_pos(1, :), new_assoc_pos(2, :), '+r');
%% plot([errpos(1, :); new_assoc_pos(1, :)], [errpos(2, :); new_assoc_pos(2, :)])
%% title('Tracker results');
%% hold off

%% subplot(2, 1, 2);
%% plot_trajectories(bg, groundtruth, 0);
%% title('Groundtruth');
