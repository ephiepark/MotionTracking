function [bg_occupancy_t, sprite_occupancies_t, objects] = update_sprites(image, background_gaussians, K, bg_occupancy, objects, ordering)
% [bg_occupancy_t, sprite_occupancies_t, objects] = update_sprites(image, background_gaussians, K, bg_occupancy, objects, ordering)
% Ordering is a vector that indicates at which depth a given sprite
% is. Thus ordering(k) is the sprite at depth k. The sprite closest to
% the background is ordering(1), and the sprite most distant is
% ordering(end).

global HEIGHT WIDTH D C;
global Weights Mus Sigmas;

%sigma_I_bg = 15;
%sigma_I = 30;
%sigma_A = 100;

sigma_I = 20;
sigma_A = 100;

nobjects = length(objects);

sprite_distances = zeros(D, 0);

% update_sprites() may be run with an empty structure array "objects". 
% In that case the ML sprite_occupancies_t we return should be empty. 
% (This way, the code can compute the posterior of the background 
% with no further modifications when no sprites are available, even 
% though we already know the posterior will be one). 
sprite_occupancies_t = zeros(D, 1);

% We smooth each occupancy and appearance map with a kernel. This gives
% sprites a chance to occupy pixels that currently have an occupancy of
% zero. A simple guess is that neighboring pixels will be similar to
% each other. 
occupancy_smoothing_kernel = fspecial('gaussian', 85, 1);
appearance_smoothing_kernel = fspecial('gaussian', 11, 1);

% Align all Appearance and Occupancy maps with the image frame by using
% the filtered displacement prediction 
% (in objects(obj).expected_y_t_plus_1_given_t(1:2)). This is achieved 
% by positioning the sprites in an oversized matrix and cropping the 
% matrix to HEIGHT-by-WIDTH  size later. Normalised displacement 
% information is stored for later use.

for obj = 1:nobjects
  displacement = objects(obj).expected_y_t_plus_1_given_t(1:2);

  % Align object occupancz map
  [alignment, sprite_x, sprite_y] = ...
      align_sprite([HEIGHT, WIDTH], objects(obj).occupancy, displacement);

  sprite_occupancies_t_minus_1(:, obj) = reshape(alignment, D, 1);

  % Smooth the occupancy map
  sprite_occupancies_t(:, obj) = reshape(conv2(alignment, occupancy_smoothing_kernel, 'same'), D, 1); 

  % Align object appearance map
  [alignment, sprite_x, sprite_y] = ...
      align_sprite([HEIGHT, WIDTH, C], objects(obj).appearance, displacement);

  sprite_appearances_t_minus_1(:, :, obj) = reshape(alignment, D, C);
  
  % Smooth the appearance map
  for cc = 1:C
    alignment(:, :, cc) = conv2(alignment(:, :, cc), appearance_smoothing_kernel, 'same'); 
%    [I, J] = ind2sub([HEIGHT, WIDTH], find(sprite_occupancies_t_minus_1(:, obj) == 0));
%    alignment(I, J, cc) = 255;
  end

  sprite_appearances_t(:, :, obj) = reshape(alignment, D, C);

  objects(obj).sprite_x = sprite_x;
  objects(obj).sprite_y = sprite_y;
end

sprite_occupancies_t(find(sprite_occupancies_t == 0)) = 1e-200;

% Keep a copy of the background occupancy and sprite occupancy and 
% appearance maps at time t-1 for use in the M step later.

bg_occupancy_t_minus_1 = bg_occupancy;
bg_occupancy_t = conv2(bg_occupancy, occupancy_smoothing_kernel, 'same');

%occupancy_t = occupancy_t_minus_1;
%sprite_appearances_t = sprite_appearances_t_minus_1;

% Iterate the EM algorithm a few times. 
for ii = 1:5
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % E-step: compute the posterior visibility given the current occupancy %
  % parameters, those from the last step, as well as the current         %
  % appearance parameters.                                               %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % For the Mixture of Gaussians, we calculate the posterior by only including 
  % those mixture components that belong to the background. This is because through 
  % updating the mixture, it has seen all data at least once and could
  % otherwise easily steal some of the foreground sprite posterior mass. The weights of
  % background mixing proportions will be normalised beforehand so that the distribution 
  % still integrates to unity. 

  mog_weights = Weights .* double(background_gaussians);
  mog_weights = mog_weights ./ repmat(sum(mog_weights, 2), 1, K);
  
  for kk = 1:K
    image_centered = double(image) - squeeze(Mus(:, kk, :));
    mog_normalisers(:, kk) = prod(2 * pi * squeeze(Sigmas(:, kk, :)), 2) .^ -(1/2);

    mog_distances(:, kk) = -(1/2) * sum((image_centered .^ 2) ./ squeeze(Sigmas(:, kk, :)), 2);
  end

%  image_centered = double(image) - double(reshape(best_background, D, C));
%  likelihood = (2 * pi * sigma_I) ^ -(C/2) * exp(-(1/2) * sum((image_centered .^ 2) ./ sigma_I, 2));

%  l(:, 1) = likelihood;
  

  % Posteriors for other sprites. Compute them according to the sprite ordering. 
  for obj = 1:nobjects
    image_centered = double(image) - sprite_appearances_t(:, :, obj);
    sprite_distances(:, obj) = -(1/2) * sum((image_centered .^ 2) ./ sigma_I, 2);

%    likelihood = (2 * pi * sigma_I) ^ -(C/2) * exp(-(1/2) * sum((image_centered .^ 2) ./ sigma_I, 2));

%    l(:, 1 + obj) = likelihood;
%    % Current sprite is visible and no closer sprite is visible
%    prior = sprite_occupancies_t(:, obj) .* ...
%	prod(1 - sprite_occupancies_t(:, ordering((find(ordering == obj)+1):end)), 2);

%    p(:, 1 + obj) = prior;
%    unnorm_posterior(:, 1 + obj) = likelihood .* prior;
  end


  % Make all those that belong to the foreground have minus infinity
  % distance. 

  mog_distances(find(abs(1 - background_gaussians))) = -inf;
  distances = [mog_distances, sprite_distances];

%  if(any(distances(:) > 0))
%    warning('Distances has positive');
%  end
  
  % Pick the median distance we have in each row. 
%  median_distance = median(distances, 2)
  median_distance = max(distances, [], 2);

  mog_dist_old = mog_distances;
  sprite_dist_old = sprite_distances;

  % Subtract this distance from all 
  mog_distances = mog_distances - repmat(median_distance, 1, size(mog_distances, 2));
  sprite_distances = sprite_distances - repmat(median_distance, 1, size(sprite_distances, 2));

%  if(any(sum([mog_distances, sprite_distances] == 0, 2) == 0))
%    warning('DOES NOT CONTAIN ZEROS');
%  end
 
  % Now compute the likelihoods and posteriors
  likelihood = zeros(D, 1);

  for kk = 1:K
    likelihood = likelihood + mog_weights(:, kk) .* mog_normalisers(:, kk) .* exp(mog_distances(:, kk));
  end

  l(:, 1) = likelihood;

  % Background is occupied but none of the other sprites that lie in
  % front of the background. 
  prior = bg_occupancy_t .* prod(1 - sprite_occupancies_t, 2);
  p(:, 1) = prior;

  % Fill in posterior for background model
  unnorm_posterior(:, 1) = likelihood .* prior;

  for obj = 1:nobjects
    likelihood = (2 * pi * sigma_I) ^ -(C/2) * exp(sprite_distances(:, obj));
    l(:, 1 + obj) = likelihood;
    prior = sprite_occupancies_t(:, obj) .* ...
	prod(1 - sprite_occupancies_t(:, ordering((find(ordering == obj)+1):end)), 2);
    p(:, 1 + obj) = prior;
    unnorm_posterior(:, 1 + obj) = likelihood .* prior;
  end

  % Normalise the posteriors: Rows must sum to one. Posterior matrix has one extra column 
  % for the background model.

  posterior = unnorm_posterior ./ repmat(sum(unnorm_posterior, 2), 1, nobjects + 1);

%  index = sub2ind([HEIGHT, WIDTH], 34, 41);
%  fprintf('Posteriors: %d\n',  posterior(index, :));
%  fprintf('\n');

  if(any(isnan(posterior(:))))
    warning('FATAL: We have NaNs in posterior distribution'); 
    keyboard;
    % Fix up NaN entries by setting them to 1/2. Both observations are
    % so unlikely that it may not hurt so much if we treat them the same. 
    posterior(find(isnan(posterior))) = 1/(nobjects + 1);
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % M-step: compute the ML parameters, given the posterior visibilities  %
  % we calculated. The ML formulation relies on some of the parameters   %
  % we had in the last time-step.                                        %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Now compute new Occupancy and Appearance parameters for
  % non-background layers. The Background layer is modelled through a 
  % mixture of Gaussians, so we will not compute Appearance parameters 
  % for it, but only Occupancy. Since the expected complete log
  % likelihood is a sum over layers, we can assume the appearance map
  % for the background layer to be a fixed input parameter. 

  % Compute Occupancy according to equation 20. 

  alpha = bg_occupancy_t_minus_1;
    
  % Since we only have one view, we let a = posterior(:, 1);
  a = posterior(:, 1);
  
  % Compute ML background occupancy. b = 0 since the background is the
  % bottom-most layer. 
  bg_occupancy_t = (alpha + a) ./ (1 + a);

  % Compute Occupancy and Appearance for all other sprites
  for obj = 1:nobjects
    % Compute occupancy according to equation 20. 
    
    alpha = sprite_occupancies_t_minus_1(:, obj);

    % Since we only have one view, we let a be the posterior of the "obj" sprite, 
    % and b the sum of posteriors of sprites that are further away than
    % obj in the ordering. Note that  posteriors is indexed such that
    % the first column vector corresponds to the background posterior.
    % Sprite posterior vectors follow after the background. 

    a = posterior(:, 1 + obj);
    b = sum(posterior(:, [1, 1 + ordering(1:(find(ordering == obj)-1))]), 2);
    
    sprite_occupancies_t(:, obj) = (alpha + a) ./ (1 + a + b);
    
    % Compute appearance using a multivariate version of equation 23
    % We have only one view j. 
    sprite_appearances_t(:, :, obj) = (repmat(posterior(:, 1 + obj) ./ sigma_I, 1, C) .* double(image) + ...
				 (sprite_appearances_t_minus_1(:, :, obj) ./ sigma_A)) ./ ...
 	              repmat(((posterior(:, 1 + obj) ./ sigma_I) + (1 / sigma_A)), 1, C);
  end
end

% Cut out the appearance and occupancy maps of each Kalman filter. We
% want to be able to grow and shrink the map.

for obj = 1:nobjects
  % Determine new extent of cut-out such that all pixels with an
  % occupancy of greater that 1/2 are included. 

  % Start with the initial cut-out size for this sprite, which we stored
  % from the alignment before. 

  sprite_x = objects(obj).sprite_x;
  sprite_y = objects(obj).sprite_y;

  % Is there an occupied pixel outwith the current cutout boundary. If so,
  % include it by adjusting the boundary accordingly. 

  occupancy = reshape(sprite_occupancies_t(:, obj), HEIGHT, WIDTH);
 
  objects(obj).occupancy = occupancy(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2));
  
  appearance = reshape(sprite_appearances_t(:, :, obj), HEIGHT, WIDTH, C);
  objects(obj).appearance = appearance(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2), :);
end
