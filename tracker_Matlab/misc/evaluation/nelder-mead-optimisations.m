%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script computes 15 Nelder-Mead optimisation runs on a specific image
% sequence (video7) with its associated ground truth data. As a result, is
% produces parameters that near-minimise the segmentation error. The
% file can be used as a skeleton for parameter searches on other
% sequences. The parameters are also useful starting points for other
% sequences. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

global HEIGHT WIDTH C D T

% Set some search paths
cd src;
addpath(pwd);
cd lap;
addpath(pwd);
cd ..;
cd misc;
addpath(pwd);
cd ../..;

% Load the raw data
data = load_data('data1/datasets/video7_long-scaled/', 'png');

% Load the ground truth data. The ground truth is still in RGB vector
% form. 
groundtruth_unprocessed = load_data('data1/datasets/video7_groundTruth_long-scaled', 'png');

% To be useful for us, we need to convert each ground truth frame to
% a binary foreground mask. Allocate some memory to hold the converted
% ground truth data.  

groundtruth = zeros(HEIGHT, WIDTH, T, 'uint8');

% Convert each frame from an RGB image to a binary mask
for tt = 1:T
  fprintf(1, 'Converting frame %04d to foreground map\r', tt);
  groundtruth(:, :, tt) = im2bw(reshape(groundtruth_unprocessed(:, :, tt), ...
					     HEIGHT, WIDTH, C));
end

clear 'groundtruth_unprocessed';

fprintf(1, '\n');

% We run a number of Nelder-Mead optimisations from quasi-random
% starting points. The next few vectors indicate reasonable starting
% points for each of the parameters that will be tuned. At the start of
% each run, a specific combination of these will be selected at random
% as a starting point for the optimisations. 

alpha_vec = 0.001:0.001:0.01;
rho_vec = 0.001:0.001:0.01;
dev_sq_thresh_vec = 20:2:40;
init_var_vec = 2:1:6;
init_prior_vec = 0.01:0.01:0.05;
background_thresh_vec = 0.3:0.05:0.7;

% We select random combinations. Initialise the random number generator state

rand('state', 0);

% Set some options for the optimiser function
options = optimset;
options.MaxFunEvals = 100;
options.MaxIter = 100;

% Before we start, clear persistent variables in parameter_eval.m
clear parameter_eval

% Do 15 optimisations from quasi-random starting points. 

for tt = 1:15;
  % Select a random initial starting position. 
  select = ceil(rand(6, 1) .* [length(alpha_vec); 
			       length(rho_vec);
			       length(dev_sq_thresh_vec);
			       length(init_var_vec);
			       length(init_prior_vec);
			       length(background_thresh_vec)]);

  % Put all parameters in the initial parameter vector. 
  X0 = [alpha_vec(select(1));
        rho_vec(select(2));
        dev_sq_thresh_vec(select(3));
        init_var_vec(select(4));
        init_prior_vec(select(5));
        background_thresh_vec(select(6))];
  
  % Run the optimisation!
  fprintf('Starting optimisation number %d\n', tt)
  [X, finalval, exitflag, output]  = ...
      fminsearch(@(X0) parameter_eval(X0, data, groundtruth, [250, 750], ...
				      sprintf('./data1/tmp/Nelder-Mead_logresp_%d', tt)), X0, options);
  
  % Store results
  search(tt).X0 = X0;
  search(tt).X = X;
  search(tt).finalval = finalval;
  search(tt).exitflag = exitflag;
  search(tt).output = output;
end
