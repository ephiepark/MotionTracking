function [] = display_sprites_stacked(fig_sprites_stacked, background, ...
				     objects, ordering, varargin)
% display_sprites_stacked(fig_sprites_stacked, background, objects, ordering)
% display_sprites_stacked(fig_sprites_stacked, background, objects, ordering, viewpoint)
% This function is used to display a stacked visualisation of different
% sprites. fig_sprites_stacked is a figure handle for
% displaying. background is a background estimate, objects is the
% structure array of all tracked objects, and ordering is a vector which
% indicates the order of objects in the depth, so that ordering(i) is
% the depth order of the i-th object. Low values indicate closeness to
% the background. Optionally a viewpoint can be given to change the
% angle from which the stack is viewed. 

global HEIGHT WIDTH;

nobjects = size(objects, 2);

% Flip order of Z so that images appear upright
[X, Z] = meshgrid(1:WIDTH, HEIGHT:-1:1);

set(0, 'CurrentFigure', fig_sprites_stacked);

% Get view (for rotating) and CameraViewAngle (for zooming) parameters 
% so that we can reset them when the new figure has been drawn. Then 
% the user can manipulate the view interactively and we maintain the 
% view after future updates.

[az, el] = view;

% If viewpoint parameters were given through the optional argument, 
% then those override the currently set ones. This is useful for
% generating rotating graphs. 

if(length(varargin) > 0)
  viewpoint = varargin{1}

  az = viewpoint(1);
  el = viewpoint(2);
end

viewangle = get(gca, 'CameraViewAngle');

hold off;
s = surf(X, zeros(HEIGHT, WIDTH), Z, 'facecolor', 'texturemap', 'cdata', uint8(background));
set(s, 'edgecolor', 'none');

hold on;

% Now add a border to highlight each sprite's extent in 3d space
plot3([1, WIDTH, WIDTH, 1, 1], ...
      zeros(1, 5), [1, 1, HEIGHT, HEIGHT, 1], 'linewidth', 2);


% Flip order of Z so that images appear upright
for obj = 1:nobjects
  [X, Z] = meshgrid(objects(obj).sprite_x(1):objects(obj).sprite_x(2), objects(obj).sprite_y(1):objects(obj).sprite_y(2));

  Y = zeros(size(X));

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % OpenGL seems to be broken on Dice machines and frequently crashes X.    %
  % If OpenGL works and is enabled in track.m, the following code will      %
  % make sprite pixels with low occupancy transparent. If opengl is         %
  % disabled, sprites will appear as fully visible rectangles that may       %
  % occlude each other.                                                     %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Only a subset of the appearance map is visible. Pick those that have
  % an occupancy of > 0.5. Uncomment if OpenGL works. 

  % Smooth each object occupancy. 

  visible = uint8(conv2(objects(obj).occupancy, fspecial('gaussian', 11, 1), 'same') > 0.5);
  s = surf(X, Y - ordering(obj), HEIGHT - Z, 'facealpha', 'texturemap', ...
	   'alphadata', visible, 'facecolor', 'texturemap', 'cdata', ...
	   uint8(objects(obj).appearance));

  set(s, 'edgecolor', 'none');

  % Now add a border to highlight each sprite's extent in 3d space
  plot3([1, WIDTH, WIDTH, 1, 1], ...
	zeros(1, 5) - ordering(obj), ...
	[1, 1, HEIGHT, HEIGHT, 1], 'linewidth', 2);
end

axis([1, WIDTH, -3, 0, 1, HEIGHT]);
axis off vis3d;
grid off;
view(az, el);
set(gca, 'CameraViewAngle', viewangle);

hold off;
drawnow;
