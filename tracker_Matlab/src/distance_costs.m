function [costs] = distance_costs(objects, pos_observed)
%[costs] = distance_costs(objects, pos_observed)
% Computes the squared distance between the centroid predictions of a list of
% objects and a list of observed blob centroids. objects are kept in a
% structure array of that records all Kalman filtering
% information. pos_observed are the blob centroids in column-vector
% form. Centroid predictions for each objects obj are taken from the field
% objects(obj).expected_y_t_plus_1_given_t(1:2)

nobjects = size(objects, 2);
nobservations = size(pos_observed, 2);

% Compute a cost matrix
costs = zeros(nobjects, nobservations);
for obj = 1:nobjects
  % Compute the square distance between Kalman filter predictions and observations
  costs(obj, :) = ...
  sum((repmat(objects(obj).expected_y_t_plus_1_given_t(1:2), 1, nobservations) - pos_observed) .^ 2, 1);
end
