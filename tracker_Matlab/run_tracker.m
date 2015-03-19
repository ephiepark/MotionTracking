% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Motion Tracker main file. 
% This file is responsible for loading one of a set of available data
% sets, appropriate parameters, and run the tracking algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add search paths
cd src;
addpath(pwd);
cd lap;
addpath(pwd);
cd ..;
cd suptitle;
addpath(pwd);
cd ../..;

% A variety of different data sets can be loaded if the matrix "data"
% doesn't already exist. Commend one of the following lines to load the
% appropriate files

if(exist('data'))
  fprintf(1, 'Already read data from %s. Continuing with normal processing.\n', sequence_path);
else
  fprintf(1, ['Enter the path to the directory containing the image ' ...
               'sequence. Make sure \n']);
  fprintf(1, ['that you enclose the path with single apostropes, ' ...
               'as in ''path/to/sequence/''.\n']);

  sequence_path = input('Path: ');  
  fprintf(1, ['Enter the image format used (i.e. ''jpg'', ''png'', etc.). Make ' ...
               'sure to use enclosing apostropes again.\n']);

  image_format = input('Format: ');

  % Load the data
  data = load_data(sequence_path, image_format);
  
  % Synthetic data can be generated using this script. 
  %generate_synthetic_sequences;

  %data = load_data('./data1/datasets/pets_2000_test_images_scaled', 'png');
  %data = load_data('./data1/datasets/pets_2001_testing_cam1_scaled/', 'png');
  %data = load_data('./data1/datasets/pets_2001_testing_cam2_scaled/', 'png');
  %data = load_data('./data1/datasets/pets_vs_2003_training_cam3_scaled_120by160/', 'png'); 
  %data = load_data('./data1/datasets/stgeorge_8fps/', 'png');
  %data = load_data('./data1/datasets/video7_long-scaled', 'png');
  %data = load_data('./data1/datasets/water-scaled_25_fps/', 'png');
  %load ~/tracker/data1/datasets/sprites-6-ready-made-labelled/data_cropped.mat
  
  % Data can also be collected from the web using this special
  % notation. The web addess is preceeded with the keyword "web:"
  % followed by the address to the image location. 
  %data = 'web:http://magellan.homelinux.org/cam/cam.php?thumb=1';
end

% Load parameters for the Mixture of Gaussian background adaptation
% algorithm.
mixparam = mixture_parameters();

% Load parameters for Kalman tracking
kalparam = kalman_parameters(); 

% Some functions rely on persistent variables. We clear them before
% proceeding. 
clear pixel_mixture_stat.m
clear add_new_hypotheses.m 

% Finally, run the algorithm with the given data and parameters. 
[frames, object_hist] = track(data, mixparam, kalparam);
