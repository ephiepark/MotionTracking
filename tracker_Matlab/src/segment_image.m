function [foreground_map, background_gaussians, best_background_gaussian] = segment_image(Weights, Sigmas, active_gaussian, BACKGROUND_THRESH, K)
%[foreground_map, background_gaussians, best_background_gaussian] = segment_image(Weights, Sigmas, active_gaussian, BACKGROUND_THRESH, K)
% Returns a foreground mask of all pixels which are considered
% foreground by the outlier detection method defined by Stauffer and
% Grimson 2000. Weight is a matrix of mixing proportions, Sigmas the
% covariances, active Gaussians is those components which were either
% matched or replaced. The BACKGROUND_TRESH is the threshold standard
% deviation for outlier detection on each gaussian. K is the number of
% components used in the mixtures. 

global HEIGHT WIDTH D

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Find maximum weight/sigma per row. Sigmas in general records the diagonal entries 
 % of each covariance matrix, but since for now the matrix is isotropic, we 
 % just divide by the square root of the first diagonal entry and sort. 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 [junk, index] = sort(Weights ./ sqrt(Sigmas(:, :, 1)), 2, 'descend');
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 % For diagonal, but not isotropic covariance use this version
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fprintf(1, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n');
% [junk, index] = sort(Weights ./ (prod(Sigmas, 3) .^ (1/6)), 2, 'descend');
 
% Record indeces of those each pixel's component we are most confident
% in, so that we can display a single background estimate later. While
% our model allows for a multi-modal background, this is a useful
% visualisation when something goes wrong. 
best_background_gaussian = index(:, 1);

% We want to reorder the actual weight according to the row-wise
% ordering given in 'index'. So first we convert each row-index into a
% fully linearised index for a D-by-K matrix. 
  
linear_index = (index - 1) * D + repmat([1:D]', 1, K);
  
% Now we can reorder the weights according to ordering in index using linear_index
weights_ordered = Weights(linear_index);
  
% Find the minimum amount of weights so that BACKGROUND_THRESH is exceeded. First 
% accumulate weights in a matrix. 
for kk = 1:K
  accumulated_weights(:, kk) = sum(weights_ordered(:, 1:kk), 2);
end
  
% If the accumulated weight up to and including component k does not exceed the 
% threshold, then we already know that component k+1 must also belong to the 
% background.
background_gaussians(:, 2:K) = accumulated_weights(:, 1:(K-1)) < BACKGROUND_THRESH;
background_gaussians(:, 1) = 1; % The first will always be selected

% Now reorder it into an indicator matrix of background_gaussians
background_gaussians(linear_index) = background_gaussians;

% Is the active Gaussian (matched, or replaced) one of the several background 
% Gaussians we have? Convert background_gaussians into an indicator matrix, with 
% a one indicating those active components that belong to the background model.  
active_background_gaussian = active_gaussian & background_gaussians;

% Those pixels that have no active background Gaussian are considered forground. 
foreground_pixels = abs(sum(active_background_gaussian, 2) - 1);
foreground_map = reshape(sum(foreground_pixels, 2), HEIGHT, WIDTH);
