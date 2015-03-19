function [objects] = kalman_prediction(objects, kalparam)
% [objects] = kalman_prediction(objects, kalparam)
% Predict the next observation for each Kalman filter. Takes a structure
% array of tracked objects and returns an updated structure array of
% objects. kalparam is a structure of Kalman parameters. See
% kalman_parameters.m for details. 

nobjects = size(objects, 2);

for obj = 1:nobjects
  % Update mean
  objects(obj).x_t_plus_1_given_t = kalparam.A * objects(obj).x_t_given_t;
  
  % And covariance
  objects(obj).P_t_plus_1_given_t = kalparam.A * objects(obj).P_t_given_t * kalparam.A' + kalparam.G * kalparam.Q * kalparam.G';
  
  % Expected mean of next observation given previous observations
  objects(obj).expected_y_t_plus_1_given_t = kalparam.B * objects(obj).x_t_plus_1_given_t;

  % Expected variance of next observation given previous observations
  objects(obj).expected_sigma_y_t_plus_1_given_t = kalparam.B * objects(obj).P_t_plus_1_given_t * kalparam.B' + kalparam.R;

  % This Kalman Filter was not newly created, so nothing needs to be initialised
  objects(obj).new_kalman = 0;
end
