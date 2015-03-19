function [trajectory_id] = closest_trajectory(pos, tt, groundtruth)
% [trajectory_id] = closest_trajectory(pos, tt, groundtruth)
% Helper function to determine the closest trajectory (i.e. the closest
% trajectory observation at a given time tt to position pos. 

dist = [];
						       
for traj = 1:size(groundtruth, 2) 
  if((groundtruth(traj).frame_created <= tt) && ...
     (tt <= groundtruth(traj).frame_destroyed));
     dist(end + 1) = norm(pos - groundtruth(traj).pos(:, tt));
  end
end

[junk, index] = min(dist);
trajectory_id = groundtruth(index).label;
