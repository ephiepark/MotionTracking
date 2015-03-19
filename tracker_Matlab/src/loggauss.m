function l = loggauss(x, Mu, Sigma)
% l = loggauss(x, Mu, Sigma)
% Compute the log of the Gaussian distribution for each point in x 
% using Mu and Sigma. x is a matrix of column-wise data-points, Mu 
% is a column vector, and Sigma the covariance matrix of respecive size. 

nobservations = size(x, 2);

Mu = repmat(Mu, 1, nobservations);
x_centered = x - Mu;

% Compute the log probability for each column in x_centered. The 
% element-wise product and sum over the second dimension is used 
% to allow us to compute the distance for the matrix x in one step. 

l = -(1/2) * (log(det(2 * pi * Sigma)) + sum((x_centered' * inv(Sigma)) .* x_centered', 2));
