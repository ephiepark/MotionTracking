function [likelihood] = beta_likelihood(pos, alpha, beta);
% [likelihood] = beta_likelihood(pos, alpha, beta);
% Computes a beta probility density on image coordinates pos. pos is a matrix of column-vectors that each 
% must specify a point within the image of HEIGHT-by-WIDTH size 
% X component first, Y second. The corresponding points are mapped 
% into the interval [0, 1] and the joint likelihood (assuming independence) 
% of the set of points is returned as a vector. 

% The "goodness" of a potential new track depends on where the first
% observation was made. This likelihood is mapped onto the dimensions of the
% image frame of size [HEIGHT, WIDTH], and has high values around the 
% corners, but low values in the middle. We prefer to create tracks near
% the corners than in the middle. 

global HEIGHT WIDTH

% Map into interval [0, 1]. 
x = pos(1, :) / WIDTH;
y = pos(2, :) / HEIGHT;

norm = gamma(alpha + beta) / (gamma(alpha) * gamma(beta));
likelihood_x = norm * (x .^ (alpha - 1)) .* ((1 - x) .^ (beta - 1));
likelihood_y = norm * (y .^ (alpha - 1)) .* ((1 - y) .^ (beta - 1));

% Joint likelihood. We assume independence. 
likelihood =  likelihood_x .* likelihood_y;
