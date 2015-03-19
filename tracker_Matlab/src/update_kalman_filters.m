function [objects] = update_kalman_filters(objects, kalmanparam, assignments, ...
					   objects_map, object_sizes, pos_observed)
%[objects] = update_kalman_filters(objects, kalmanparam, assignments, ...
%					   objects_map, object_sizes, pos_observed)
% Update a structure array of tracked objects with new observations
% (blobs). Parameters for the updates of Kalman filters are specified in
% the parameter structure kalmanparam. Blob associations come from the
% assignment matrix and measurements from objects_map, object_sizes and
% pos_observed. The function returns an update structure array of
% objects. 
 
global HEIGHT WIDTH KALMAN_STATE_SIZE

% Since some assignments were gated, some rows in the assignments
% matrix may be all zero. Compute a list of those objects that
% actually received a match. Those that didn't will have their filter
% artificially propagated by feeding their own prediction back as an 
% observation.

assigned_objects = sum(assignments, 2);

% For each Kalman Filter, update it with its assignment. Those filters
% which were not assigned an actual observations will instead be
% fed their prediction back into their measurement update. 

for obj = 1:size(objects, 2)
  % Each object is either updated, or propagated. No matter what, the
  % length of the trajectory increases. 
  objects(obj).traj_length = objects(obj).traj_length + 1;

  % It has matched 
  if(assigned_objects(obj)) % This filter was assigned an observation
    obj_observation = find(assignments(obj, :)); % This is the observation
    % fprintf(1, 'Matching filter %d with point %d\n', obj, obj_observation);

    if(objects(obj).new_kalman)
      % Newly initialised filters don't have a previous
      % state. create_kalman_observation() sets any state fields that may
      % depend on a previous state (velocity etc.) to zero if no
      % previous state is given. 
      objects(obj).y_t_plus_1 = create_kalman_observation(pos_observed(:, obj_observation), ...
 	                        object_sizes(obj_observation));

      % Also pretend that the last prediction of the new filter was
      % the current state. This is easy as long as B is the identity matrix. 
      objects(obj).x_t_plus_1_given_t = inv(kalmanparam.B) * objects(obj).y_t_plus_1;

      % The expected mean observation for this state is the current observation
      objects(obj).expected_y_t_plus_1_given_t = objects(obj).y_t_plus_1;

      % Furthermore, give the filter a new covariance matrix P. This
      % should ideally be the same kind of matrix we would get from
      % one time prediction at the very beginning. For current
      % settings, this is the identity matrix, if the P at the onset
      % was the zero or identity matrix. 

      objects(obj).P_t_plus_1_given_t = eye(KALMAN_STATE_SIZE);
    else % The filter was not newly added, so we have previous
         % state information available
      	
      % Construct new observation from pos_observed and size. Spatial
      % velocity is approximated by the coordinate difference to the 
      % previously matched observation. Similarly for the rate of change
      % of the size component. 

      objects(obj).y_t_plus_1 = create_kalman_observation(pos_observed(:, obj_observation), ...
                                object_sizes(obj_observation), ...
			        objects(obj).y_t_plus_1);
    end

    % We will consider the filter finished when it has been propagated
    % too many times. Reset the count here, since it was updated with
    % an observation.
    objects(obj).ntimes_propagated = 0;

    % Each time the object is matched, increment the updated counter. 
    % When the object is destroyed we can use this information to
    % determine whether the object trajectory was "good" or "bad".
    objects(obj).ntimes_updated = objects(obj).ntimes_updated + 1;

    % Compute the bounding box
    obs_indicator = (objects_map == obj_observation); 

    % Get the coordinates for a rectangle containing the observation
    [X, Y] = meshgrid(1:WIDTH, 1:HEIGHT);
    X = X .* obs_indicator;
    Y = Y .* obs_indicator;

    % Sprite rectangle coordinates
    objects(obj).bbox_x(1) = min(nonzeros(X(:)));
    objects(obj).bbox_x(2) = max(X(:));
    
    objects(obj).bbox_y(1) = min(nonzeros(Y(:)));
    objects(obj).bbox_y(2) = max(Y(:));
  else % No observation was matched with this filter
%      fprintf(1, 'Propagating filter %d\n', obj);
  
    % Use the prediction of that filter as the next observation
    objects(obj).y_t_plus_1 = objects(obj).x_t_plus_1_given_t;

    % We will consider the filter finished when it has propagated
    % too many times. Record this information here. 
    objects(obj).ntimes_propagated = objects(obj).ntimes_propagated + 1;
  end
 
  % Update each filter with its observation. 
  [objects(obj).x_t_given_t, objects(obj).P_t_given_t] = ...
      kalman_measurement_update(objects(obj).y_t_plus_1, ...
      objects(obj).x_t_plus_1_given_t, objects(obj).P_t_plus_1_given_t, kalmanparam);
end 
