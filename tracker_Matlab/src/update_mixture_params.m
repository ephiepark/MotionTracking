function [Weights, Mus, Sigmas, replaced_gaussian] = ...
  update_mixture_params(image, Weights, Mus, Sigmas, matched_gaussian, ...
  K, ALPHA, RHO, deviations_squared, log_responsibilities_unnorm, INIT_MIXPROP, INIT_VARIANCE)
%[Weights, Mus, Sigmas, replaced_gaussian] = ...
%  update_mixture_params(image, Weights, Mus, Sigmas, matched_gaussian, ...
%  K, ALPHA, RHO, deviations_squared, log_responsibilities_unnorm, INIT_MIXPROP, INIT_VARIANCE)
% Update each mixture of each pixel by reading in new pixel values and
% updating matching mixture componetns. This function implements the
% algorithm specified by Stauffer and Grimson 2000 to update mixtures
% of Gaussians to streaming input pixels.   

global HEIGHT WIDTH D C;

% Now update the weights. Increment weight for the selected Gaussian (if any),
% and decrement weights for all other Gaussians.

Weights = (1 - ALPHA) .* Weights + ALPHA .* matched_gaussian;

% At this stage some rows may not sum to unity, since some pixels may have 
% not been matched by a mixture component at all. These mixtures will have 
% one component replaced later and weights subsequently renormalised at the end. 

% Adjust Mus and Sigmas for matching distributions. This is
% tricky. Extend the indicator matrix to cover the C dimensions of
% each pixel and then play with the Mus.

for kk = 1:K
  pixel_matched = repmat(matched_gaussian(:, kk), 1, C);
  pixel_unmatched = abs(pixel_matched - 1); % Inverted and mutually exclusive

  Mu_kk = reshape(Mus(:, kk, :), D, C);
  Sigma_kk = reshape(Sigmas(:, kk, :), D, C);

  Mus(:, kk, :) = pixel_unmatched .* Mu_kk + ...
      pixel_matched .* (((1 - RHO) .* Mu_kk) + ...
  			(RHO .* double(image)));

  % Get updated Mus; Sigmas is still unchanged
  Mu_kk = reshape(Mus(:, kk, :), D, C); 

  if(any(Mu_kk < 0))
    error('FATAL: Mus is negative after update of component');
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % For only isotropic covariance matrices use the following update. It  %
  % fills the last dimension (size C) with the same value sigma instead  %
  % of maintaining a single value to allow for an easy switch between    %
  % isotropic and non-isotropic.                                         %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Only have one sigma per pixel, but copy it over the last dimension.
  Sigmas(:, kk, :) = pixel_unmatched .* Sigma_kk + ...
                     pixel_matched .* (((1 - RHO) .* Sigma_kk) + ...
		     repmat((RHO .* sum((double(image) - Mu_kk) .^ 2, 2)), 1, C));

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % If we decide to use a diagonal but not isotropic covariance          %
  % matrix, then use the following update for the Sigmas.                %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %  Sigmas(:, kk, :) = pixel_unmatched .* Sigma_kk + ...
  %                     pixel_matched .* (((1 - RHO) .* Sigma_kk) + ...
  %      	            (RHO .* ((double(image) - Mu_kk) .^ 2)));

  if(any(Sigmas(:, kk, :) < 0))
    error('FATAL: Sigmas is negative after update of component');
  end
end

% Maintain an indicator matrix of those components that were replaced because no component matched. 
replaced_gaussian = zeros(D, K); 

% Find those pixels which have no Gaussian that matches
mismatched = find(sum(matched_gaussian, 2) == 0);

% Select only those entries where 

for ii = 1:length(mismatched)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % A method that works well: Replace the component we            %
  % are least confident in. This includes weight in the choice of %
  % component.                                                    %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Select the component we are least "confident" in. 
  [junk, index] = min(Weights(mismatched(ii), :) ./ sqrt(Sigmas(mismatched(ii), :, 1)));

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Another method to try: Replace the component we               %
  % which has the least responsibility for this point. This also  %
  % includes weight in the choice of component.                   %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %[junk, index] = min(log_responsibilities_unnorm(mismatched(ii), :));

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % As a control case, use this method to select the component    %
  % with maximum standard deviation.                              %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % [junk, index] = max(deviations_squared(mismatched(ii), :));

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % For diagonal but not isotropic covariance, use this           %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %  Analogue to the confidence method, but for diagonal covariances
  % [junk, index] = min(Weights(mismatched(ii), :) ./ ...
  % 	       prod(Sigmas(mismatched(ii), :, :), 3) .^ (1/6));

  % Mark that this Gaussian will be replaced
  replaced_gaussian(mismatched(ii), index) = 1;

  % With a distribution that has the current pixel as mean
  Mus(mismatched(ii), index, :) = image(mismatched(ii), :);
  % And a relatively wide variance
  Sigmas(mismatched(ii), index, :) = ones(1, C) * INIT_VARIANCE;

  % Also set the weight to be relatively small
  Weights(mismatched(ii), index) = INIT_MIXPROP;
end

% Now renormalise the weights so they still sum to 1
Weights = Weights ./ repmat(sum(Weights, 2), 1, K);

% Every pixel has exactly one replaced or matched Gaussian
if(any(sum(matched_gaussian + replaced_gaussian, 2) ~= 1))
  error('FATAL: Pixel has more than one replaced or matched Gaussian');
end
