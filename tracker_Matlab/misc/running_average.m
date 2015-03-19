function [frames] = running_average(start, final, alpha)

mulist = [];
sigmalist = [];
datalist = [];

mu = start;
sigma = [1; 1];

BUFFER = 20;
NPREVPOINTS = 20;

figure(1);

while(~(abs(mu(1) - final(1)) / final(1) < 0.01) || ~(sigma(1) < 20))
  data = randn(2, 1) + final;
  mu = (1 - alpha) * mu + alpha * data;
  sigma = (1 - alpha) * sigma + alpha * (data - mu)' * (data - mu);

  datalist(:, end+1) = data;
  mulist(:, end+1) = mu;
  sigmalist(:, end+1) = sigma;

  index = max(1, length(mulist) - NPREVPOINTS);

%  hold off;
%  scatter(datalist(1, index:end-1), datalist(2, index:end-1), ...
%	25, repmat(fliplr([1:length(mulist)-index])', 1, 3) ./ (length(mulist) - index), ...
%	'Marker', '.');

%  hold on;

% Mark the start point
%  scatter(start(1), start(2), 50, 'k+');

%  axis equal;

  % Determine axis limits
  axisminx = min(start(1), final(1)) - BUFFER;
  axismaxx = max(start(1), final(1)) + BUFFER;

  axisminy = min(start(2), final(2)) - BUFFER;
  axismaxy = max(start(2), final(2)) + BUFFER;

  set(gca, 'XLim', [axisminx, axismaxx]);
  set(gca, 'YLim', [axisminy, axismaxy]);

  % Plot the last pixel blue
%  scatter(datalist(1, end), datalist(2, end), 500, 'b.');
%  plotellipse(mulist(1, end), mulist(2, end), ...
%      sqrt(sigmalist(1, end)), sqrt(sigmalist(2, end)), 'r');
%  hold off;
%  drawnow;
%  if(~exist('frame'))
%     frames(1) = getframe();
%   end
%  frames(end+1) = getframe();
end

figure(2);
plot(mulist(1, :), 'b');
hold on;
plot(sigmalist(1, :), 'r');
hold off;
