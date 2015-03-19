function [] = plot_trajectories(image, trajectories, percent, last, scale)
% [] =  plot_trajectories(image, trajectories, percent, last, scale)
% Given a background image and a structure array with fields pos, plot
% those trajectories that had been updated at least percent times of their
% length. Plot each trajectory only until the frame indicated in last. 
% Scale the data: if we have ground truth it needs to be
% transformed. 

% Count how many "good" trajectories we have so we can compute an index into
% the colour map. Each trajectory will receive a maximally different colour.

ntrajectories = size(trajectories, 2);
ngoodtrajectories = 0;
for traj = 1:ntrajectories
  % Plot only those trajectories which were update more than half the time

  if((trajectories(traj).ntimes_updated / ...
      (size(trajectories(traj).pos, 2) - trajectories(traj).frame_created + 1)) >= percent)
    ngoodtrajectories = ngoodtrajectories + 1;
  end
end

imagesc(image);
hold on;
nth_goodtrajectory = 0;

C = colormap; % Give each line a different colour
for traj = 1:ntrajectories
  % Plot only those trajectories which were update more than percent of the time
  if((trajectories(traj).ntimes_updated / ...
      (size(trajectories(traj).pos, 2) - trajectories(traj).frame_created + 1)) >= percent)
    nth_goodtrajectory = nth_goodtrajectory + 1; % This is the nth good trajectory. 

    % Compute a pointer into the colour map. 
    colourindex =  ceil((nth_goodtrajectory / ngoodtrajectories) * size(C, 1));

    % Now plot
    frame_created = trajectories(traj).frame_created;
%    last_updated = trajectories(traj).last_updated;
last_updated = min(trajectories(traj).last_updated, last(traj))

    trajectories(traj).pos(:, frame_created);

    if(scale)
      trajectories(traj).pos = trajectories(traj).pos ./ 4.8;
      trajectories(traj).pos(2, :) = 120 - trajectories(traj).pos(2, :);
    end

    plot(trajectories(traj).pos(1, frame_created:last_updated), ...
	 trajectories(traj).pos(2, frame_created:last_updated), 'Color', ...
	 C(colourindex, :), 'LineWidth', 2);
  end
end
drawnow;
hold off; axis off; axis ij;
