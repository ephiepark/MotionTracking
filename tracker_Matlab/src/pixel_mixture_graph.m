function [] = pixel_mixture_graph(fig_mix_graph, debug_coord, data, tt, ...
			    matched_gaussian, replaced_gaussian, background_gaussians, mixparam);
% [] = pixel_mixture_graph(fig_mix_graph, debug_coord, data, tt, ...
%		     matched_gaussian, replaced_gaussian, ...
%		     background_gaussian, mixparam);
% Draw 2d scatterplots of the fist two dimensions of the data and
% superimpose ellipses indicating the Gaussians. 
% Arguments are a figure handle, the pixel-coordinate of the debugging
% pixel the data-matrix of all images, the time index, a binary
% indicator matrix of which Gaussian matched at each pixel, an indicator
% matrix of replaced Gaussians, a binary
% indicator matrix of all background Gaussians at each pixel, and the
% parameter structure used to update the mixtures with. 

global HEIGHT WIDTH;
global Mus Sigmas;

NPREVPOINTS = 200;

set(0, 'CurrentFigure', fig_mix_graph);

pixel_index = sub2ind([HEIGHT, WIDTH], debug_coord(1, 2), debug_coord(1, 1));

start_index = max(1, tt - NPREVPOINTS);

hold off;

% Plot recent points with a little bit of noise added, so that density
% becomes visible despite discreteness of data. Make noise matrix three 
% dimensional like the sliced data matrix, so we can simply add in
% noise.  
noise = randn(1, 2, tt - start_index) * 0.1; 
scatter(double(data(pixel_index, 1, start_index:tt-1)) + noise(1, 1, :), ...
	double(data(pixel_index, 2, start_index:tt-1)) + noise(1, 2, :), ...
	25, repmat(fliplr([1:tt-start_index])', 1, 3) ./ (tt - start_index), ...
	'Marker', '.');

hold on;

% Plot the last pixel blue
scatter(data(pixel_index, 1, tt), data(pixel_index, 2, tt), 500, 'b.');

xlabel('Red');
ylabel('Green');

axis equal;

% For each component, plot an ellipse (or circle if covariance matrix is
% diagonal) of its unit standard deviation. Also plot a dashed 
% representation of the matching boundary imposed by standard deviation 
% threshold DEVIATION_SQ_THRESH. 

for kk = 1:mixparam.K
  hold on;
  
  % A Gaussian is drawn black by default
  colour = 'k';

  % Line width
  width = 2;

  % If this Gaussian has replaced another Gaussian, draw it red
  if(replaced_gaussian(pixel_index, kk) == 1)
    colour = 'r';
  else % Draw it green if it was matched
    if(find(matched_gaussian(pixel_index, :)) == kk)
      colour = 'g';
    end
  end

  plotellipse(Mus(pixel_index, kk, 1), Mus(pixel_index, kk, 2), ...
      sqrt(Sigmas(pixel_index, kk, 1)), sqrt(Sigmas(pixel_index, kk, 2)), colour, width);

  % Now plot the matching boundary. The unit standard deviation is
  % the square root of the variance. The matching boundary is given in
  % terms of standard deviations, so we multiply by each components own
  % standard deviation to plot proper ellipse. We use the fact that the
  % covariance matrix is diagonal here. Use blue to indicate background
  % boundaries. 

  % All regions drawn in broken black by default
  marker = 'k-.';

  % Background mixtures in broken blue
  if(background_gaussians(pixel_index, kk) == 1)
    marker = 'b-.';
  end

  plotellipse(Mus(pixel_index, kk, 1), Mus(pixel_index, kk, 2), ...
	      sqrt(Sigmas(pixel_index, kk, 1) * mixparam.DEVIATION_SQ_THRESH), ...
	      sqrt(Sigmas(pixel_index, kk, 2) * mixparam.DEVIATION_SQ_THRESH), marker, width);
end

%axis([0, 255, 0, 255]);

% Calculate the position for a text label
% Extract the axis scales
XLim = get(gca, 'XLim');
YLim = get(gca, 'YLim');

% Compute positions
xpos = XLim(1) + (XLim(2) - XLim(1)) / 20;
ypos = YLim(2) - (YLim(2) - YLim(1)) / 20;

% Place frame number 
text(xpos, ypos, sprintf('%04d', tt), 'Color', 'r');

%text(10, 245, sprintf('%04d', tt), 'Color', 'r');

drawnow;
hold off;
