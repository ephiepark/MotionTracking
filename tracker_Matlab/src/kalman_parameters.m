function [kalparam] = kalman_parameters()
% [kalparam] = kalman_parameters()
% Helper function to create a structure of Kalman filtering
% parameters. Among others, the structure defines all relevant matrices,
% the kind of state space used, as well as types of cost functions and
% association algorithms. Use type kalman_parameters.m to see relevant
% options.

global KALMAN_STATE_TYPE KALMAN_STATE_SIZE; % They will be needed in functions later

% Negative log probability dummy
kalparam.ASSOC_DUMMY_COST = 2010;

% Distance dummy for PETS 2001 sequence
%kalparam.ASSOC_DUMMY_COST = 100

kalparam.NEW_HYP_DUMMY_COST = 2;

% What type of state representstion does the Kalman Filter use,
% i.e. what aspects of an object silhouette are used to describe the
% object. 
% 'pos' = only x and y position
% 'pos+vel' = x and y position and also their velocities
% 'pos+vel+size' = positions, velocities, size
% 'pos+vel+size+size_vel' = positions, velocities, size and change of size over time

KALMAN_STATE_TYPE = 'pos+vel+size';
%KALMAN_STATE_TYPE = 'pos+vel';

% Depending on the state representation, set the state transition matrix
% A accordingly. 

if(strcmp(KALMAN_STATE_TYPE, 'pos'))
  KALMAN_STATE_SIZE = 2;

  % Update using position only
  % This assumes that state vectors have the form:
  % [pos_x, pos_y]'
  
  kalparam.A = [1 0;
		0 1];
end

if(strcmp(KALMAN_STATE_TYPE, 'pos+vel'))
  KALMAN_STATE_SIZE = 4;

  % Update using position and velocity
  % This assumes that state vectors have the form:
  % [pos_x, pos_y, vel_x, vel_y]'
  kalparam.A = [1 0 1 0;
		0 1 0 1;
		0 0 1 0;
		0 0 0 1];
end

if(strcmp(KALMAN_STATE_TYPE, 'pos+vel+size'))
  KALMAN_STATE_SIZE = 5;
  
  % Update using position and velocity
  % This assumes that state vectors have the form:
  % [pos_x, pos_y, vel_x, vel_y, size]'
  kalparam.A = [1 0 1 0 0;
		0 1 0 1 0;
		0 0 1 0 0;
		0 0 0 1 0;
		0 0 0 0 1];
end

if(strcmp(KALMAN_STATE_TYPE, 'pos+vel+size+size_vel'))
  KALMAN_STATE_SIZE = 6;
  
  % Update using position and velocity
  % This assumes that state vectors have the form:
  % [pos_x, pos_y, vel_x, vel_y, size, d_size_d_t]'
  kalparam.A = [1 0 1 0 0 0;
		0 1 0 1 0 0;
		0 0 1 0 0 0;
		0 0 0 1 0 0;
		0 0 0 0 1 1;
		0 0 0 0 0 1];
end

G_SCALEFACT = 1;
B_SCALEFACT = 1;
Q_SCALEFACT = 1;
R_SCALEFACT = 1;

kalparam.G = G_SCALEFACT * eye(KALMAN_STATE_SIZE);
kalparam.B = B_SCALEFACT * eye(KALMAN_STATE_SIZE); % Called C in Jordan's book
kalparam.Q = Q_SCALEFACT * eye(KALMAN_STATE_SIZE); 
kalparam.R = R_SCALEFACT * eye(KALMAN_STATE_SIZE);

% A filter is allowed to propate at most MAXPROPAGATE times
kalparam.MAXPROPAGATE = 100;

% What kind of cost values are used to populate the cost matrix
% Can be either 'distance' for squared distance between candidate pairs,
% or 'kalman_expectation for the negative log probability under the
% statistics for the expected observation. 

%kalparam.ASSOC_COST_TYPE = 'distance';
kalparam.ASSOC_COST_TYPE = 'kalman_expectation';

% What kind of association algorithm is used for the data association. Two
% options include 'LAP', the Linear Assignment Problem algorithm provided by
% Jonker and Volgenant, 'SMP', the Stable Marriage Problem proposed by Gale
% and Shapley and 'MUNK', a Matlab implementation of the Munkres
% algorithm. If you are having trouble compiling the file lap.cpp, switch
% this variable to 'MUNK' or 'SMP' instead.

kalparam.ASSOC_ALG_TYPE = 'MUNK';
