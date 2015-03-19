function [costs] = kalman_costs(objects, pos_observed, object_sizes)
% [costs] = kalman_costs(objects, pos_observed, object_sizes)
% Computes costs for matching Kalman filters to observations using the
% predictive distribution of the expected next observation of Kalman
% filters. All Kalman filters are stored as part of the structure array
% objects. All observations (blobs) are reported as the blob centroids
% in pos_observed and the blob sizes in object_sizes. From these
% features new observations are created and these are evaluated under the
% negative log likelihood function to yield matching costs. 

nobjects = size(objects, 2);
nobservations = size(pos_observed, 2);

% Compute a cost matrix
costs = zeros(nobjects, nobservations);
for obj = 1:nobjects
  proposed_states = create_kalman_observation(pos_observed, ...
		    object_sizes, objects(obj).y_t_plus_1);
  
  % Compute the negative log probability density of every observation 
  % we made to the data the filter expects, given its history of 
  % previous observations. 
  costs(obj, :) = -loggauss(proposed_states, objects(obj).expected_y_t_plus_1_given_t, objects(obj).expected_sigma_y_t_plus_1_given_t);
end
