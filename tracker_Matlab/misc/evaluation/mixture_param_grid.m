clear;

global HEIGHT WIDTH D C T

cd src;
addpath(pwd);
cd lap;
addpath(pwd);
cd ..;
cd misc;
addpath(pwd);
cd ../..;

data = load_data('data1/datasets/video7_long-scaled/', 'png');
groundtruth_unprocessed = load_data('data1/datasets/video7_groundTruth_long-scaled', 'png');

groundtruth = zeros(HEIGHT, WIDTH, T, 'uint8');

for tt = 1:T
  fprintf(1, 'Converting frame %04d to foreground map\r', tt);
  groundtruth(:, :, tt) = im2bw(reshape(groundtruth_unprocessed(:, :, tt), ...
					     HEIGHT, WIDTH, C));
end

fprintf(1, '\n');

% We evaluate the whole alpha rho grid
alpha_vec = 0.00001:0.05:1;
rho_vec = 0.00001:0.05:1;

% And select random settings for these
dev_sq_thresh_vec = 20:3:60;
init_var_vec = 3:1:6;
init_prior_vec = 0.01:0.01:0.05;
background_thresh_vec = 0.3:0.05:0.7;

% Before we start, make sure persistent variables in parameter_eval.m
% are cleared (precaution, even though we set the name argument to [] 
% during call time.)
clear parameter_eval

for tt = 1:15;
  % Select a random initial starting position. 
  select = ceil(rand(6, 1) .* [0; 
			       0;
			       length(dev_sq_thresh_vec);
			       length(init_var_vec);
			       length(init_prior_vec);
			       length(background_thresh_vec)]);

  X0 = [0;
        0;
        dev_sq_thresh_vec(select(3));
        init_var_vec(select(4));
        init_prior_vec(select(5));
        background_thresh_vec(select(6))];

  
  % Evaluate on each alpha-rho combination

  for ii = 1:length(alpha_vec)
    for jj = 1:length(rho_vec)
      ii
      jj

      X0(1) = alpha_vec(ii);
      X0(2) = rho_vec(jj);

      X0
	
      % Save no trajectory information
      errors(ii, jj) = parameter_eval(X0, data, groundtruth, [250, 750], '');
    end
  end

  % Just in case, mark the first entries as invalid. The alpha and rho
  % vectors are stored below and X0 just records the remaining
  % parameters. 
  X0(1) = -1;
  X0(2) = -1;

  save(sprintf('./data1/tmp/Parameter_Grid_coarse_run_%d.mat', tt), 'errors', ...
       'X0', 'alpha_vec', 'rho_vec');
end
