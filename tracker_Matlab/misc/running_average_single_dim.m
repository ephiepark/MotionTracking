function [frames] = running_average(mustart, sigmastart, final, alpha)

mulist = [];
sigmalist = [];
datalist = [];

mu = mustart;
sigma = sigmastart;

epsilon = 0.1

% Predict the time it takes for the mu to reach within a certain error
t = log(epsilon / (final - mustart)) / log(1 - alpha);

while(~((final - mu) < epsilon) || ~(sigma < 10))
  mu = (1 - alpha) * mu + alpha * final;
  mulist(end+1) = mu;
  sigma = (1 - alpha) * sigma + alpha * (final - mu)' * (final - mu);
  sigmalist(end+1) = sigma;
end

figure(1);
plot(mulist(1, :), 'b');
hold on;
plot(sigmalist(1, :), 'r');
hold off;

fprintf(1, 'Iterations required %d, predicted %d\n', length(mulist), t);
