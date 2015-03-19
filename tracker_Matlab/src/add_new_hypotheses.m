function [assignments, objects, nobjects] = add_new_hypotheses(assignments, objects, objects_map, pos_observed, image, kalparam)
%[assignments, objects, nobjects] = add_new_hypothese(assignments, objects, objects_map, pos_observed, image, kalparam)
% This function is responsible for adding new track hypotheses. It uses
% a given binary indicator matrix of assignments, structure array of
% objects, objects blobs in objects_map, blob centroid positions in
% pos_observed, and a dummy value which will be used as the
% maximum allowable cost for matching consecutive blobs. 

% Original code by Fabian Wauthier, University of Edinburgh, June
% 2007. Modified to add munkres algorithm by Ken Ho, Sydney University on
% 25/9/08

global HEIGHT WIDTH C

persistent next_kalman_label;
persistent prev_unassigned;

% Is this is the first call to this function initialise the label to 1
if(isempty(next_kalman_label))
  next_kalman_label = 1;
end

% If this is the first run, then create an empty list of previously
% unassigned observations. 

if(isempty(prev_unassigned))
  prev_unassigned = zeros(2, 0);
end

nobjects = size(objects, 2);
nobservations = size(pos_observed, 2);
nprev_unassigned = size(prev_unassigned, 2);

% If there are observations that were not matched on this iteration,
% then we will add a new track for them, if there exists an unmatched observation 
% from the last iteration to which the current observation has a small
% squared distance, and if the previous observation was close to the image
% border. 

% Unassigned observations
unassigned = find(sum(assignments, 1) == 0);
nunassigned = length(unassigned);

new_hyp_cost = zeros(nunassigned, nprev_unassigned);
for obs = 1:nunassigned
  % How "good" are tracks that started at one of the previously
  % unassigned observations? The cost amplification is high in 
  % the middle, but low on the sides. 
  amp = beta_likelihood(prev_unassigned, 1.5, 1.5);

  % Compare this observation to all previously unassigned observations. 
  distance = ...
  sqrt(sum((prev_unassigned - repmat(pos_observed(:, unassigned(obs)), 1, ...
				nprev_unassigned)) .^ 2, 1));

  % The cost of matching is the product.
  new_hyp_cost(obs, :) = amp .* distance;
end

% Make the matrix square by adding dummies
nrows = size(new_hyp_cost, 1);
ncols = size(new_hyp_cost, 2);

if(nrows == ncols)
  lap_new_hyp_cost = new_hyp_cost;
end

if(nrows < ncols)
  lap_new_hyp_cost = [new_hyp_cost; ones(ncols - nrows, ncols) * kalparam.NEW_HYP_DUMMY_COST];
end

if(ncols < nrows)
  lap_new_hyp_cost = [new_hyp_cost, ones(nrows, nrows - ncols) * kalparam.NEW_HYP_DUMMY_COST];
end

% Add further dummies to avoid matching unlikely pairs. 
lap_new_hyp_cost = [lap_new_hyp_cost, ones(size(lap_new_hyp_cost)) * kalparam.NEW_HYP_DUMMY_COST;
	          ones(size(lap_new_hyp_cost)) * kalparam.NEW_HYP_DUMMY_COST, ones(size(lap_new_hyp_cost)) * kalparam.NEW_HYP_DUMMY_COST];

% Run the LAP assignment algorithm to minimise the cost. There's two
% flavours to solving this. One relies on a compiled version of the
% lap.cpp code, the other uses a matlab implementation of the Munkres
% algorithm. T

if(strcmp(kalparam.ASSOC_ALG_TYPE, 'LAP')) % Use lap.cpp
  obs_assignments = lap(lap_new_hyp_cost);
else % Use munkres.m
  obs_assignments = munkres(lap_new_hyp_cost);
%  obs_assignments = obs_assignments';
end

%keyboard
obs_assignments = obs_assignments(1:nrows, 1:ncols); % Crop out dummy matches

% Have to do this in a two-step process, since matlab find(sum())
% doesn't work consistently on zero-by-N matrices. 

newly_assigned = [];
if(any(sum(obs_assignments, 2)))
   newly_assigned = find(sum(obs_assignments, 2));
end

% Every observation that was now assigned a match. 
for obs = newly_assigned' % Transpose, since newly_assigned is a column vector
  nobjects = nobjects + 1;

  % Remaining fields such as state will be initialised later. 
  objects(nobjects).new_kalman = 1; 

  objects(nobjects).ntimes_propagated = 0;

  % After the object has been initialised with the Kalman statistics,
  % this will be set to one.
  objects(nobjects).ntimes_updated = 0;

  % Similarly, the initialisation gives the first length count. 
  objects(nobjects).traj_length = 0;

  objects(nobjects).label = next_kalman_label;

  next_kalman_label = next_kalman_label + 1;
    
  % Mark the match (we have to convert the obs index into
  % unassigned(obs) so we can index into the original
  % assignments matrix. 
  assignments(nobjects, unassigned(obs)) = 1;

  % Create new sprite layers for new observations. First compute the spatial 
  % extent of each observation, then copy out a rectangle containing the 
  % appearance and create a corresponding rectangular occupancy map. The 
  % Occupancy map has a value of 0.8 for foreground pixels. 

  % A HEIGHT-by-WIDTH indicator matrix of where the observation "obs" is located

  obs_indicator = (objects_map == unassigned(obs)); 

  % Get the coordinates for a rectangle containing the observation
  [X, Y] = meshgrid(1:WIDTH, 1:HEIGHT);
  X = X .* obs_indicator;
  Y = Y .* obs_indicator;

  % Sprite rectangle coordinates
  sprite_x(1) = min(nonzeros(X(:)));
  sprite_x(2) = max(X(:));
    
  sprite_y(1) = min(nonzeros(Y(:)));
  sprite_y(2) = max(Y(:));

  % Create an image of the current frame so we can copy the appearance sprite
  obs_image = image .* uint8(repmat(obs_indicator, [1, 1, C]));
%  obs_image(find(obs_indicator == 0)) = 255;		    

%  canvas = zeros(10 + (sprite_y(2) - sprite_y(1) + 1), 10 + (sprite_x(2) - sprite_x(1) + 1), C);
%  canvas(6:6 + sprite_y(2) - sprite_y(1), 6:6 + sprite_x(2) - sprite_x(1), :) = double(image(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2), :));
%  objects(nobjects).appearance = canvas; 
%  size(canvas)

  objects(nobjects).appearance = double(obs_image(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2), :));

%  canvas = zeros(10 + (sprite_y(2) - sprite_y(1) + 1), 10 + (sprite_x(2) - sprite_x(1) + 1));
%  canvas(6:6 + sprite_y(2) - sprite_y(1), 6:6 + sprite_x(2) - sprite_x(1)) = obs_indicator(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2), :); 
%  objects(nobjects).occupancy = 0.8 * canvas; 

  objects(nobjects).occupancy = 0.8 * obs_indicator(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2));

  % Also store the image coordinates of the sprite rectangle 
  objects(nobjects).sprite_x = sprite_x;
  objects(nobjects).sprite_y = sprite_y;

  % Sprite coordinates are different from the bounding box coordinates
  % (ideally they shouldn't be, but for now they are). They will be
  % initialised identically, but then develop
  % independently. 

  objects(nobjects).bbox_x = sprite_x;
  objects(nobjects).bbox_y = sprite_y;
end

% Anything that was not assigned this round will be available for
% matching with new unmatched observations in the next round. 
prev_unassigned = pos_observed(:, find(sum(assignments, 1) == 0));
