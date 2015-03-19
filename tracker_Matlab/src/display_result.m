function [] = display_result(fig_data, tt, image, foreground_map, objects_map, sprite_occupancies, best_background, positions, objects);
%display_result(fig_data, tt, data, foreground_map, objects_map,
%  	        sprite_occupancies, best_background, positions, objects);
% Display data and implement a point-and-click callback that creates
% additional windows with information concerning the particular
% pixel-mixture the user clicked on. fig_data is a figure handle,
% foreground_map is a binary map indicating foreground pixels, objects_map
% indicates connected components which are fed to the tracking
% subsystem. sprite_occupancies can be used to indicate appearance
% outlines even when an object is occluded (requires the sprite
% subsystem to be activated). best_background is a unimodal background
% estimate, positions indicates the centroids of blobs and objects is a
% structure array of all objects which are currently tracked. 

global C D HEIGHT WIDTH;
global Mus debug_coord;

RED_HUE = 128;

foreground_hue_map = zeros(HEIGHT, WIDTH, C, 'uint8');
foreground_hue_map(:, :, 1) = RED_HUE * foreground_map;

% Add a red hue to foreground pixels
foreground_hued_image = uint8(image + foreground_hue_map);

objects_hue_map = zeros(HEIGHT, WIDTH, C, 'uint8');
objects_hue_map(:, :, 1) = RED_HUE * (objects_map > 0);

% Add a red hue to objects pixels
objects_hued_image = uint8(image + objects_hue_map);

set(0, 'CurrentFigure', fig_data);
set(gcf, 'doublebuffer', 'on');

subplot(2, 2, 1);
set(gca, 'Units', 'normalized', 'Position', [0 0.505 0.495 0.5])
imagesc(image);
hold on; axis ij; axis off;
if(~any(debug_coord < 0))
  plot(debug_coord(1), debug_coord(2), 'yo');
end
text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 15);
hold off;

subplot(2, 2, 2);
set(gca, 'Units', 'normalized', 'Position', [0.505 0.505 0.495 0.495])
imagesc(foreground_hued_image);
hold on; axis ij; axis off;
%contour(foreground_map, 1, 'r');
if(~any(debug_coord < 0))
  plot(debug_coord(1), debug_coord(2), 'yo');
end
text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 15);
hold off;

subplot(2, 2, 3);
set(gca, 'Units', 'normalized', 'Position', [0 0 0.495 0.495])
imagesc(best_background);
hold on; axis ij; axis off;
if(~any(debug_coord < 0))
  plot(debug_coord(1), debug_coord(2), 'yo');
end
text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 15);
hold off;

subplot(2, 2, 4);
set(gca, 'Units', 'normalized', 'Position', [0.505 0 0.495 0.495])
imagesc(objects_hued_image);
hold on; axis ij; axis off;
%contour(objects_map > 0, 1, 'r');

% For each sprite, display a contour. 
for obj = 1:size(sprite_occupancies, 2)
  % Smooth each object occupancy
  occupancy = conv2(reshape(sprite_occupancies(:, obj), HEIGHT, WIDTH), fspecial('gaussian', 11, 1), 'same');
  
  % Only plot those parts of the occupancy map that are not 
  % occluded by other objects. 

  contour(occupancy > 0.5, 1, 'y');
end

if(~any(debug_coord < 0))
  plot(debug_coord(1), debug_coord(2), 'yo');
end
% If there are any objects, plot the center of mass of each 
if(size(positions, 2) > 0)
  scatter(positions(1, :), positions(2, :), 150, 'y+');
end
text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 15);

for obj = 1:size(objects, 2)
  xpos = objects(obj).expected_y_t_plus_1_given_t(1);
  ypos = objects(obj).expected_y_t_plus_1_given_t(2);
  
  % By default, filter predictions are coloured black if they were propagated
  obj_colour = 'k';
  obj_marker = 'ok';
    
  % If the object was matched (i.e. never propagated) plot it red or green
  if(objects(obj).ntimes_propagated == 0)
    if(objects(obj).new_kalman)
      obj_colour = 'r';
      obj_marker = 'or';
    else
      obj_colour = 'g';
      obj_marker = 'og';
    end

    % Draw a bounding box
    plot([objects(obj).bbox_x(1), objects(obj).bbox_x(2), ...
	  objects(obj).bbox_x(2), objects(obj).bbox_x(1), ...
	  objects(obj).bbox_x(1)], ...
	 [objects(obj).bbox_y(1), objects(obj).bbox_y(1), ...
	  objects(obj).bbox_y(2), objects(obj).bbox_y(2), ...
	  objects(obj).bbox_y(1)], obj_colour);
  end
      
  scatter(xpos, ypos, obj_marker, 'fill');
 %text(5 + xpos, 3 + ypos, sprintf('%d', objects(obj).label), 'Color', obj_colour, 'FontSize', 15, 'BackgroundColor', [1, 1, 1]);
text(objects(obj).bbox_x(2) + 5, objects(obj).bbox_y(1) - 5, ...
     sprintf('%d', objects(obj).label), 'Color', obj_colour, ...
     'BackgroundColor', [0.5, 0.5, 0.5], 'Fontsize', 15);
  end

hold off;

drawnow;
