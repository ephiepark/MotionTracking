function [error, errors] = parameter_eval(X0, data, groundtruth, window, name)
% error = parameter_eval(X0, data, groundtruth, window, name)
% Evaluate data on parameters X0. Final error is averaged over the range
% of frames specified in window. If filename given, saves 
% trajectory of subsequent calls in the files name_1.mat and name_2.mat 
% in alternating fashion, in case Matlab crashes. If name == '',
% accumulation stops and nothing more is saved. 

  persistent trajectory prev_name next_index;

  % Start with empty name
  if(isempty(prev_name))
    prev_name = [];
  end

  % Set up a new trajectory if the previous name and current
  % name differ. 
  if(~strcmp(name, prev_name))
    prev_name = name;
    trajectory = struct([]);
    next_index = 1;
  end

  % But only save if the name is not empty
  save_trajectory = 0;
  if(~strcmp(name, ''))
     save_trajectory = 1;
  end

  % Store the parameter we are evaluating. We are being called by
  % fminsearch and would like to keep a trajectory of the parameters we
  % evaluate. 

  X0
  
  mixparam.K = 3;
  mixparam.COMPONENT_THRESH = 10;
    
  % Start parameters
  mixparam.ALPHA = X0(1);
  mixparam.RHO = X0(2);
  mixparam.DEVIATION_SQ_THRESH = X0(3);
  mixparam.INIT_VARIANCE = X0(4);
  mixparam.INIT_MIXPROP = X0(5);
  mixparam.BACKGROUND_THRESH = X0(6);

  kalparam = kalman_parameters(); 

  errors = track(data, mixparam, kalparam, groundtruth);
  
  % Return the average error over a window of frames. 
  error = sum(errors(window(1):window(2))) / (window(2) - window(1) + 1)

  save_trajectory

  % Save it to alternating files in case Matlab crashes
  if(save_trajectory)
    next_index
    name
    sprintf('%s_%d.mat', name, ...
		 mod(next_index, 2))
     trajectory(next_index).X0 = X0;
     trajectory(next_index).error = error;
     next_index = next_index + 1;
     save(sprintf('%s_%d.mat', name, ...
		 mod(next_index, 2)), 'trajectory');
  end