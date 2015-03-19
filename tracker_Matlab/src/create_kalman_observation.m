function y = create_kalman_observation(positions, object_sizes, varargin)
% y = create_kalman_observation(positions, object_sizes), or
% y = create_kalman_observation(positions, object_sizes, previous_states)
% Generate new state observations y (with columns corresponding to state
% vectors) using the positions and object_sizes matrices. If given, and useful
% for the state representation chosen, the previous states will be used
% to set additional state fields (such as velocity, or change of size
% over time), based on the previous state. All matrices have relevant
% entries arranged in column form, where each column corresponds to a
% different state vector.  

global KALMAN_STATE_TYPE KALMAN_STATE_SIZE;

if(length(varargin) > 0)
  HAVE_PREVIOUS_STATES = 1;
  previous_states = varargin{1};
else
  HAVE_PREVIOUS_STATES = 0;
end

nobservations = size(positions, 2);

% Create output matrix, some entries may remain set at zero
y = zeros(KALMAN_STATE_SIZE, nobservations);

% The first two state entries are the positions, regardless of what
% representation was chosen. 
y(1:2, :) = positions;

% If required by the state and available, fill in velocity information
if(strcmp(KALMAN_STATE_TYPE, 'pos+vel') || ...
   strcmp(KALMAN_STATE_TYPE, 'pos+vel+size') || ...
   strcmp(KALMAN_STATE_TYPE, 'pos+vel+size+size_vel'))
  if(HAVE_PREVIOUS_STATES == 1)
    y(3:4, :) = positions - repmat(previous_states(1:2), 1, size(positions, 2)); 
  end
end

% If required by the state, fill in size information
if(strcmp(KALMAN_STATE_TYPE, 'pos+vel+size') || ...
   strcmp(KALMAN_STATE_TYPE, 'pos+vel+size+size_vel'))
  y(5, :) = object_sizes;
end

% If required by the state, and available, fill in change of size
% information. 
if(strcmp(KALMAN_STATE_TYPE, 'pos+vel+size+size_vel'))
  if(HAVE_PREVIOUS_STATES == 1)
    y(6, :) = object_sizes - repmat(previous_states(5), 1, size(positions, 2));
  end
end
