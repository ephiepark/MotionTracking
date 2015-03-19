function [labels, err_pos, new_assoc_pos, switch_percent, nupdated_percent, average_distance_error] = tracker_confusion(trajectories, groundtruth)
% [labels, err_pos, new_assoc_pos, switch_percent, nupdated_percent] =
%  tracker_confusion(trajectories, groundtruth)

THRESH = 9; % Allow 9 pixels deviation at no cost. 

% For each timestep, copute how many filters have switched from tracking
% one object to tracking another. This is achieved by looking for the
% closest groundtruth measurement at time tt and comparing the id of the
% corresponding track to the id of the previously associated track at
% time tt - 1. 

labels = cat(2, trajectories.label);

% How many times did the object switch from tracking one object to
% tracking another object. This is determined using a LAP formulation on
% closest ground truth trajectories. 

nswitch = ones(size(trajectories, 2), 1);
err_pos = [];
new_assoc_pos = [];

% Mark as unassigned
prev_assigned = ones(size(trajectories, 2), 1) * -1;

% Look for the longest pos field in the trajectory and ground-truth 
% data. This is how long we run the algorithm. 

T = 0;

for traj = 1:size(trajectories, 2)
  if(size(trajectories(traj).pos, 2) > T)
    T = size(trajectories(traj).pos, 2);
  end
end

for traj = 1:size(groundtruth, 2)
  if(size(groundtruth(traj).pos, 2) > T)
    T = size(groundtruth(traj).pos, 2);
  end
end

for tt = 1:T
  % Create a matrix of distance costs between positions we
  % have in "trajectories" and "groundtruth" at time tt.

  % Mark the active kalman trajectories in a vector to save
  % computation later. 
  
  costs_ii = 0;
  costs_ii_map = [];
  
  for ii = 1:size(trajectories, 2)
    % If the two trajectories are active during this time tt, we will 
    % add a new entry to the costs matrix.
    
    if(trajectories(ii).frame_created <= tt && ...
       tt <= trajectories(ii).frame_destroyed)
      
      % Add new row to costs matrix
      costs_ii = costs_ii + 1;
      
      % Also remember the index in the trajectory structure. By
      % indexing into this vector we know which row corresponds to
      % which trajectory. 
      costs_ii_map(costs_ii) = ii;
    end
  end

  % Mark the active groundtruth trajectories in a vector to save
  % computation later. 
  
  costs_jj = 0;
  costs_jj_map = [];
  
  for jj = 1:size(groundtruth, 2)
    % If the two trajectories are active during this time tt, we will 
    % add a new entry to the costs matrix.
    
    if(groundtruth(jj).frame_created <= tt && ...
       tt <= groundtruth(jj).frame_destroyed)
      
      % We will add a new entry in cost column
      costs_jj = costs_jj + 1;
      
      % Also remember the groundtruth index. By indexing into this 
      % vector we know which column corresponds to which 
      % groundtruth trajectory.  
      costs_jj_map(costs_jj) = jj;
    end
  end

  costs = zeros(costs_ii, costs_jj);
 
  for costs_ii = 1:length(costs_ii_map)
    for costs_jj = 1:length(costs_jj_map)	
      % Cost is the norm of position difference at time tt
      costs(costs_ii, costs_jj) = ...
	  norm(trajectories(costs_ii_map(costs_ii)).pos(:, tt) - ...
	       groundtruth(costs_jj_map(costs_jj)).pos(:, tt));
    end
  end

  if(any(size(costs) == 0))
    costs = [];
  end

  [mincost, minindex] = min(costs, [], 2);

  for row = 1:size(costs, 1)
    % The corresponding Kalman trajectory and groundtruth trajectory
    % indeces for this match are:
    traj = costs_ii_map(row);
    ground = costs_jj_map(minindex(row));
      
    % See if this object has been matched before. If not, then match it
    % now at no cost

    if(prev_assigned(traj) == -1)
      fprintf(1, 'Time: %d: associating %d with %d\n', tt, traj, ground)
      
      % Associate at no cost
      prev_assigned(traj) = ground; % Associate 
      prev_dist(traj) = mincost(row);
    end

    distance_accumulator(traj) = mincost(row);

    % If they differ. 
    if(prev_assigned(traj) ~= ground)
      % Don't flag an error if both distances were below a certain
      % limit. If two persons walk closeby, then we suppress errors dure
      % to frequent switching. FIXME is this ok?
      if(~(prev_dist(traj) < THRESH && mincost(row) < THRESH))
	fprintf(1, 'Time: %d: %d was associated with %d, now %d\n', tt, traj, prev_assigned(traj), ground);
	
	% Increment error and record the position where the error occurred
	nswitch(traj) = nswitch(traj) + 1;
	
	err_pos(:, end + 1) = trajectories(traj).pos(:, tt);
        new_assoc_pos(:, end + 1) = groundtruth(ground).pos(:, tt); 
    
        prev_assigned(traj) = ground; % Associate 
	prev_dist(traj) = mincost(row);
      end

    end
  end
end

% Finally, we scale each error entry by the length of the observed
% track. 

lengths = cat(1, trajectories.traj_length);

%switch_percent = nswitch;
%nupdated_percent = cat(1, trajectories.ntimes_updated);

switch_percent = nswitch ./ lengths;
nupdated_percent = cat(1, trajectories.ntimes_updated) ./ lengths;

average_distance_error = distance_accumulator' ./ lengths;

%plot(distance_accumulator ./ lengths');
