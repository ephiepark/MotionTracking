function [trajectories] = extract_trajectories(object_hist)
% [] = extract_trajectories(object_hist)
% Given a structure array of object arrays, extract the trajectories as a
% structure array with fields pos and label indicating the trajectory data. 

% First accumulate the trajectory data of corresponding objects. Use the
% labels information from the kalman tracker for this purpose and build
% an array of structs with fields pos and label. Also keep a count of
% how many times the object was updated (i.e. how many times the
% ntimes_propagated field was 0). 

% Start with no trajectories
trajectories = struct([]);
next_trajectory = 1;

% For each frame
for tt = 1:size(object_hist, 2)
  fprintf(1, 'Processing objects at frame %d\n', tt);

  objects = object_hist(tt).objects;

  % Make a list of the trajectories we are maintaining. 
  if(length(trajectories) > 0)
    % So we can search for labels
    labels = cat(2, trajectories.label);
  else
    labels = [];
  end

  % For each object we recognised at that frame
  for obj = 1:size(objects, 2)
    % Is this a new trajectory?
    if(~any(labels == objects(obj).label))
      % Make new trajectory
      fprintf('Adding new trajectory %d\n', objects(obj).label);
      trajectories(next_trajectory).label =  objects(obj).label;
      trajectories(next_trajectory).frame_created = tt;
      trajectories(next_trajectory).pos(:, tt) = objects(obj).expected_y_t_plus_1_given_t(1:2);
      trajectories(next_trajectory).ntimes_updated = 1; % First time updated
      next_trajectory = next_trajectory + 1;
    else
      % Look up trajectory index
      index = find(labels == objects(obj).label);

      % Ensure it has a field pos. 
%      if(~isfield(trajectories(index), 'pos'))
%	 trajectories(index).pos = zeros(2, 0);
%      end

      trajectories(index).pos(:, tt) = objects(obj).expected_y_t_plus_1_given_t(1:2);

      % It was updated
      if(objects(obj).ntimes_propagated == 0)
	trajectories(index).ntimes_updated = trajectories(index).ntimes_updated + 1;
	trajectories(index).last_updated = tt;
      end
    end
  end
end

% Finally set each frame_destroyed to indicate the length of the pos
% vector, and set a field traj_length to indicate the length of the
% active trajectory. 

for traj = 1:size(trajectories, 2)
  trajectories(traj).frame_destroyed = size(trajectories(traj).pos, 2);
  trajectories(traj).traj_length = trajectories(traj).frame_destroyed - ...
      trajectories(traj).frame_created + 1;
end
