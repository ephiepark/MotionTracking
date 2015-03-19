function [varargout] = track(data, mixparam, kalparam, varargin) 
% track(data, mixparam, kalparam) 
% [frames, object_history] = track(data, mixparam, kalparam) 
% [errors] = track(data, mixparam, kalparam, groundtruth) 
%
% Track() accepts a data matrix with pixel indices as first dimension, the
% pixel colour component as second dimension and the time-slice as third
% dimension. Mixture parameters are passed through the struct mixparam which
% has the following fields: 
%
% K: how many components does the mixture have
% ALPHA: How fast to adapt the component weights
% RHO: How fast to adapt the component means and covariances
% DEVIATION_SQ_THRESH: Threshold used to find matching versus
%   unmatching components. 
% INIT_VARIANCE: Initial variance for newly placed components
% INIT_MIXPROP: Initial prior for newly placed components
% COMPONENT_THRESH: Filter out connected components of smaller size
% BACKGROUND_THRESH: Percentage of weight that must be accounted for by
%   background models. 
%
% Kalman tracking parameters are given through the structure kalparam. 
%
% Track() optionally accepts a binary ground truth segmentation
% matrix as a second argument, which has the I and J image coordinates
% as the first two dimensions, and the time slice as third dimension. If
% this matrix is provided, track() returns a vector of frame-wise sum
% squared errors.
% Optionally, by setting switch variables inside the code, the
% function will return recorded image frames, as well as a structure array 
% of objects at each frame. This is used to plot object trajectories 
% and the evolution of the appearance and occupancy maps. If switch
% variables are off, then these will be empty. 

global HEIGHT WIDTH C D T
global Mus Sigmas Weights debug_coord

% The return values will be overwritten if required. By default they are
% empty. 
varargout{1} = [];
varargout{2} = [];

% Should we ignore input data parameter and read images from live camera?
STREAMING_DATA = 0;

% If "data" is a string, it indicates the source that we should read
% images from.

if(isstr(data))
  STREAMING_DATA = 1;
  T = inf; % We set this, because we don't really have any data yet

  % Store for polling later
  stream_file_locator = data;

  % Returned value is a D-by-C matrix containing the first image, or []
  % upon error
  data = get_streaming_image(stream_file_locator);

  % Check if the image is an empty matrix (indicates error)
  if(isempty(data))
    error('FATAL: cannot read first image');
  end
end

% Print informative messages?
PRINTMESSAGES = 1;

% Display images and graphs?
DISPLAY = 1;

% Save image frames of the data window for making movies?
SAVE_IMAGE_FRAMES = 0;

% Save the the objects struct at each frame so we can produce sprite
% visualisations, as well as trajectory plots later. 
SAVE_OBJECT_HISTORY = 1;

% This switch was used for developing and experimenting with sprites. It
% is not normally turned on. 
DO_CHEATING_SPRITES = 0;

% Enable sprite processing (experimental). 
DO_SPRITES = 0;

% The data can be optionally converted into chromaticity coordinates
% before processing. Set to one to activate chromaticity coordinates. 
DO_CHROMATICITY = 0;

% Are we evaluating the performance of parameters (i.e. do we have
% groundtruth labellings?). This will be automatically set to 1 if
% corresponding arguments are supplied. So the default initial
% value is 0. 

EVALUATE_MIXPARAM = 0; 

% Disable messages, and graphics when we have groundtruth for parameter evaluations. 
if(length(varargin) > 0)
%  PRINTMESSAGES = 0;
  DISPLAY = 0;
  SAVE_IMAGE_FRAMES = 0;
  SAVE_OBJECT_HISTORY = 0;
  DO_CHEATING_SPRITES = 0;
  EVALUATE_MIXPARAM = 1;
  groundtruth = varargin{1};

  % If we are evaluating mixparam, then sanity check the input. This
  % is useful when using automatic optimisation methods, such as the
  % Nelder-Mead search technique. 
  if(invalid_mixparam(mixparam))
    varargout{1}(1:T) = inf; % Return infinite error. 
    return;
  end
end

% Initialise some global variables
K = mixparam.K;
ALPHA = mixparam.ALPHA;
RHO = mixparam.RHO;
DEVIATION_SQ_THRESH = mixparam.DEVIATION_SQ_THRESH;
INIT_VARIANCE = mixparam.INIT_VARIANCE;
INIT_MIXPROP = mixparam.INIT_MIXPROP;
COMPONENT_THRESH = mixparam.COMPONENT_THRESH;
BACKGROUND_THRESH = mixparam.BACKGROUND_THRESH;

debug_coord = [-1, -1];

fprintf(1, 'Computing mean vectors... ');
global Mus Sigmas;

% Place the first Gaussian on the first observed pixel. Place all other
% Gaussians randomly. By placing the Gaussians randomly we somewhat reduce the
% chance that they might match data close to the first pixel. 
% (Note: ideally we want some other method that they can never be matched.)


Mus = zeros(D, K, C);
Mus(:, 1, :) = double(data(:, :, 1));
Mus(:, 2:K, :) = rand([D, K-1, C]) * 255;
fprintf(1, 'done.\n');

if(DO_CHROMATICITY)
  % Replace third chromaticity one by the luminance
  luminance = double(sum(data(:, :, 1), 2) ./ 3);

  Mus(:, 1, :) = (double(data(:, :, 1)) ./ ...
	double(repmat(sum(data(:, :, 1), 2), 1, C))) * 255;
  Mus(:, 1, 3) = luminance;
end

fprintf(1, 'Setting covariance matrices... ');
Sigmas = ones(D, K, C) * INIT_VARIANCE; 
fprintf(1, 'done.\n');

% Give each Gaussian except the first one zero
% weight. This ensures that uninitialised Gaussians (i.e. everything but
% the first) will be replaced first time we have the chance. 
Weights = [ones(D, 1), zeros(D, K-1)];

deviations_squared = zeros(D, K);
matched_gaussian = zeros(D, K); % All gaussians that matched a pixel

log_responsibilities_unnorm = zeros(D, K); % For responsibility replacement strategy

% Kalman Filters are kept in a structure array "objects". Each structure contains 
% the the following fields describing the filter. 
% new_kalman: Is set to 1, if some fields require further initialisation. 0 otherwise. 
% ntimes_propagated: How many times has this filter been propagated without receiving an observation.
% label: The filter's integer label
% y_t_plus_1: A measurement vector with which the filter was most recently updated
% x_t_plus_1_given_t: The mean of the hidden state at time t+1 given observations y up to time t
% expected_y_t_plus_1_given_t: The expected mean of the next measurement
% P _t_plus_1_given_t: The covariance matrix of the hidden state at time t+1 given observations y up to t
% x_t_given_t: The mean of the hidden state given observations y up to time t
% P_t_given_t: The covariance matrix of the hidden state given observations y up to time t

% Currently we have no Kalman filters. 
nobjects = 0;
objects = struct([]);

% For each sprite we have a occupancy map. By default these are empty,
% but they are filled in if sprites are enabled. 
sprite_occupancies = zeros(D, 0);

% The background is viewed as a special kind of sprite with its own
% occupancy map
bg_occupancy = ones(D, 1) * 0.8;

if(DISPLAY && ~EVALUATE_MIXPARAM)
  fig_data = figure(1);

  % Old pos [23 518 560 403]
  set(gcf, 'Name', 'Data display', 'NumberTitle', 'off', 'Position', [23 444 562 477]);

  set(fig_data, 'WindowButtonDownFcn', 'store_pixel_coord(get(gca, ''CurrentPoint''))');

  if(DO_SPRITES)
    fig_sprites_tiled = figure(4);
%    campos([-1000, 0, 0]);
  end
end

for tt = 1:T
  % If we want chromaticity coordinates, then scale the image now
  if(DO_CHROMATICITY)
    % We want to have normal graphics
    chromaticity_orig_image = data(:, :, tt);

    % Replace third chromaticity one by the luminance
    luminance = double(sum(data(:, :, tt), 2) ./ 3);
 
    data(:, :, tt) = (double(data(:, :, tt)) ./ ...
		      double(repmat(sum(data(:, :, tt), 2), 1, C))) * 255;
    data(:, 3, tt) = luminance;
  end

  % If we are reading streaming images the overwrite data
  if(STREAMING_DATA)
    tt = 1;

    % Returned value is a D-by-C matrix containing the image, or [] upon error
    im = get_streaming_image(stream_file_locator);

    % Check if the image is an empty matrix (indicates error)
    if(isempty(im))
      continue; % Continue with next iteration.
    end

    data = im;
  end

  if(PRINTMESSAGES)
    fprintf(1, 'Evaluating frame %d... ', tt);
  end

  for kk = 1:K
    if(PRINTMESSAGES)
      fprintf(1, ' %d ', kk);
    end

    if(any(Sigmas < 0))
      error('FATAL: Sigmas is negative at start of main loop');
    end

    % Center the image around the means of the kk-th Gaussian
    data_centered = double(data(:, :, tt)) - reshape(Mus(:, kk, :), D, C);

    % Compute the squared Mahalanobis distance. This only works if
    % Sigmas represents a diagonal covariance matrix in the form
    % of a vector of diagonal entries.

    deviations_squared(:, kk) = sum((data_centered .^ 2) ./ reshape(Sigmas(:, kk, :), D, C), 2); 

    % We compute unnormalised responsibilities for each
    % component. To be efficient, we make heavy use of the fact that the 
    % covariance is diagonal and isotropic. This is written so that the 
    % covariance may be changed to be non-isotropic, but still diagonal 
    % later (i.e. needing more than one Sigmas entry per component). 
    % The prod() simulates the determinant of a diagonal matrix to
    % compute the normaliser. 

    log_normalisers(:, kk) = -(1/2) * log(prod(2 * pi * squeeze(Sigmas(:, kk, :)), 2));
  end

  % Compute unnormalised log responsibilities for an alternative replacement strategy
  % This will in generall give a complaint about log of zero, but for
  % our purposes this is fine, since we look for the minimal log
  % responsibility later and negative infinity is perfect for this
  warning('off', 'MATLAB:log:logofZero')
  log_responsibilities_unnorm = log(Weights) + log_normalisers - ((1/2) .* deviations_squared); 
  warning('on', 'MATLAB:log:logofZero')


  if(PRINTMESSAGES)
    fprintf(1, 'done.\n');
  end

  if(any(isnan(deviations_squared)))
    error('FATAL: deviations_squared contains NaN');
  end

  % Now see if any of the Gaussians matches to within DEVIATION_SQ_THRESH
  % deviations.

  % The best match for a pixel has min deviation. 
  [junk, index] = min(deviations_squared, [], 2); 

  % From index, whose elements are indeces of the minimum element of each
  % row, compute an indicator matrix that has a 1 in the corresponding
  % position of the min element and a 0 otherwise. 

  matched_gaussian = zeros(size(deviations_squared));
  matched_gaussian(sub2ind(size(deviations_squared), 1:length(index), index')) = ones(D, 1);

  % Weed out those indeces that correspond to rows that have no element
  % below a certain threshold. I.e. only matching Gaussians should be
  % marked. 

  matched_gaussian = matched_gaussian & (deviations_squared < DEVIATION_SQ_THRESH);

  if(any(sum(matched_gaussian, 2) > 1))
    error('FATAL: indicator matrix is has multiple matching Gaussians per pixel');
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Update parameters                                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  [Weights, Mus, Sigmas, replaced_gaussian] = ...
    update_mixture_params(data(:, :, tt), Weights, Mus, Sigmas, matched_gaussian, ...
    K, ALPHA, RHO, deviations_squared, log_responsibilities_unnorm, INIT_MIXPROP, INIT_VARIANCE);

  active_gaussian = matched_gaussian + replaced_gaussian;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Segment the image                                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  [foreground_map, background_gaussians, best_background_gaussian] = ...
     segment_image(Weights, Sigmas, active_gaussian, BACKGROUND_THRESH, K);

  % Since we know the best background component at each pixel, we can
  % produce an image of the means of those components we are most
  % confident in. 

  best_background = uint8(best_background_image(Mus, best_background_gaussian));

  % Smoothing of the foreground mask could be done here, but is kind of
  % cheating. It was not done for the project. 
  % foreground_map = conv2(double(foreground_map), ones(3), 'same') > SMOOTHING_THRESHOLD;

  % If we are evaluating mixparam then we record the summed pixel-wise
  % segmentation error of this frame, according to some groundtruth. 

  if(EVALUATE_MIXPARAM)
    varargout{1}(tt) = sum(sum(abs(double(groundtruth(:, :, tt)) - ...
				   double(foreground_map))));
    continue;  
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Connected components; filter out some noise.              %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Connected components of 8 and thresholded to a count of COMPONENT_THRESH pixels
  [objects_map, nobservations, object_sizes, pos_observed] = ...
    foreground_objects(foreground_map, 8, COMPONENT_THRESH);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Kalman prediction (time update and estimation of new observation) %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if(nobjects > 0)
    % Let each filter do its time update, and compute statistics for the
    % expected new observation. 
    objects = kalman_prediction(objects, kalparam);

    % At each frame, we calculate which Kalman Filters are "finished",
    % i.e. which can be deleted. If the Kalman Filter prediction is not
    % within the central area of the window, mark it as finished. Also, if
    % a filter hasn't been updated for a certain time, it is considered
    % finished. 

    ntimes_filter_propagated = cat(2, objects.ntimes_propagated);
    expected_y_t_plus_1_given_t = cat(2, objects.expected_y_t_plus_1_given_t);

    finished_objects = ...
	(~positions_within_window(expected_y_t_plus_1_given_t(1:2, :), [1, WIDTH], [1, HEIGHT])) | ... 
	(ntimes_filter_propagated > kalparam.MAXPROPAGATE);

    % Now adjust the pool of available objects by deleting all those 
    % whose filters we consider finished. 

    % How many objects are remaining
    nobjects = sum(finished_objects == 0);

    % Discard all objects that have finished
    objects = objects(find(~finished_objects));
  end

  % Based on the filtered displacement predictions of the Kalman
  % filters, update the sprite model. Note, that we do not use observed
  % displacement information to update the sprite models, as then we would
  % be ignoring the context of previous observations the Kalman filter may
  % have made (i.e. no filtering).

  if(DO_SPRITES)
    [bg_occupancy, sprite_occupancies, objects] = update_sprites(data(:, :, tt), background_gaussians, K, bg_occupancy, objects, nobjects:-1:1);

    if(~exist('fig_sprites_tiled') || ~ishandle(fig_sprites_tiled))
      fig_sprites_tiled = figure(3);
    end

    display_sprites_tiled(fig_sprites_tiled, data(:, :, tt), best_background, bg_occupancy, objects);
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Data association                                           %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Intially, and by default nothing is associated. So if no
  % observations are made at the current frame, every unfinished filter
  % will still be propagated. 
  assignments = zeros(nobjects, nobservations);

  if(nobservations > 0)
    [assignments, gate] = data_association(objects, pos_observed, object_sizes, kalparam);
    [assignments, objects, nobjects] = add_new_hypotheses(assignments, objects, ...
	objects_map, pos_observed, reshape(data(:, :, tt), HEIGHT, WIDTH, C), kalparam);
  end

% The next section only executes if we are computing the sprites from
% the preprocessed and labelled data
%% Horrible initialisation, but works
     if(DO_CHEATING_SPRITES)
      if(tt == 52)
        load ./data1/datasets/sprites-6-ready-made-labelled/points_cropped.mat
        size(objects)
        cheating_objects = objects([1, 4]);
        size(cheating_objects)
        cheating_order = [2, 1];

        cheating_bg_occupancy = bg_occupancy;
      end
  
      if(tt >= 52)
        cheating_objects(1).expected_y_t_plus_1_given_t = points(2, :, tt);
        cheating_objects(2).expected_y_t_plus_1_given_t = points(1, :, tt);

        display_sprites_tiled(fig_sprites_tiled, data(:, :, tt), best_background, bg_occupancy, cheating_objects);
    
        [cheating_bg_occupancy, sprite_occupancies, cheating_objects] = ...
          update_sprites(data(:, :, tt), background_gaussians, K, cheating_bg_occupancy, cheating_objects, cheating_order);
    
        cheating_hist(tt - 51).objects = cheating_objects;
        cheating_hist(tt - 51).background = best_background;

        if(mod(tt, 20) == 0)
          save('./data1/tmp/cheating_hist.mat', 'cheating_hist');
        end
      end
    end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Update Kalman Filters (measurement update)                 %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  objects = update_kalman_filters(objects, kalparam, assignments, ...
				  objects_map, object_sizes, pos_observed);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Display some graphs                                        %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if(DISPLAY)
    % Sometimes Matlab crashes using OpenGL graphics. For this reason we
    % disable it. Unfortunately, this means that the stacked sprite
    % visualisation will not be effective without transparency maps. If 
    % OpenGL works, this line may be commented and the stacked graphics 
    % should work. 

    opengl neverselect;

    if(DO_CHROMATICITY)
      display_result(fig_data, tt, reshape(chromaticity_orig_image, HEIGHT, WIDTH, C), ...
  		   foreground_map, objects_map, sprite_occupancies, ...
		   best_background, pos_observed, objects);
    else      				     
      display_result(fig_data, tt, reshape(data(:, :, tt), HEIGHT, WIDTH, C), ...
  		   foreground_map, objects_map, sprite_occupancies, ...
		   best_background, pos_observed, objects);
    end

    % If the debug_coords have changed from the ones before, then
    % check if the debugging window handles exist and are still valid. 
    % If not, then create new handles. The user can click on a pixel to
    % receive detailed statistics, and can then close the window again
    % when she is done. 

    % On first iteration we merely initialise. 
    if(~exist('prev_debug_coord'))
      prev_debug_coord = debug_coord;
    end
    
    if(any(debug_coord ~= prev_debug_coord))
      if(~exist('fig_mix_stat') || ~ishandle(fig_mix_stat))
	fig_mix_stat = figure(2);
	set(gcf, 'Name', 'Pixel statistics', 'NumberTitle','off', 'Position', [641 31 580 890]);
      end

      if(~exist('fig_mix_graph') || ~ishandle(fig_mix_graph))
	fig_mix_graph = figure(3);
	set(gcf, 'Name', 'Pixel scatter plot', 'NumberTitle','off', 'Position', [23 16 561 403]);
      end
    end
    
    % Remember if anything was updated for next time.
    prev_debug_coord = debug_coord;

    % If the debugging window handles exist and are valid, then display
    % a figure containing the pixel statistics. 

    if(exist('fig_mix_stat') && ishandle(fig_mix_stat))
      pixel_mixture_stat(fig_mix_stat, debug_coord, data, tt, ...
			 matched_gaussian, replaced_gaussian, ...
			 background_gaussians, mixparam);

    end

    % Also show a graph of how the mean and variances adapt if the
    % handle exists and is valid. 
    if(exist('fig_mix_graph') && ishandle(fig_mix_graph))
      pixel_mixture_graph(fig_mix_graph, debug_coord, data, tt, ...
			  matched_gaussian, replaced_gaussian, ...
			  background_gaussians, mixparam);
    end

    if(SAVE_IMAGE_FRAMES)
       frames(tt) = getframe(fig_data);
    end


%    if(DO_CHROMATICITY)
%      save_result(tt, reshape(chromaticity_orig_image, HEIGHT, WIDTH, C), ...
%  		   foreground_map, objects_map, sprite_occupancies, ...
%		   best_background, pos_observed, objects);
%    else      				     
%      save_result(tt, reshape(data(:, :, tt), HEIGHT, WIDTH, C), ...
%  		   foreground_map, objects_map, sprite_occupancies, ...
%		   best_background, pos_observed, objects);
%    end
  end

  if(SAVE_OBJECT_HISTORY)
    object_hist(tt).objects = objects;
%    object_hist(tt).background = best_background;
  end
end

if(SAVE_IMAGE_FRAMES && ~EVALUATE_MIXPARAM)
  varargout{1} = frames;
end

if(SAVE_OBJECT_HISTORY && ~EVALUATE_MIXPARAM)
  varargout{2} = object_hist;
end
