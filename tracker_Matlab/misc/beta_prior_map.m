function [prior] = beta_prior_map(alpha, beta);

global HEIGHT WIDTH

prior = zeros(HEIGHT, WIDTH);

for xx = 1:WIDTH
  for yy = 1:HEIGHT
    prior(yy, xx) = beta_prior([xx; yy], alpha, beta);
  end
end
