function [] = save_result(tt, image, foreground_map, objects_map, sprite_occupancies, best_background, positions, objects);
%save_result(tt, data, foreground_map, objects_map,
%  	        sprite_occupancies, best_background, positions, objects);
% This function is in principle identical to display_result() with the
% difference that it uses different font and marker sizes suitable for
% presenting videos on smaller projectors. It saves various figures that
% were displayed to image sequences.

% It creates three new
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

if(~ishandle(10))
  figure(10);
  set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
end

if(~ishandle(11))
  figure(11);
  set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
end

if(~ishandle(12))
  figure(12);
  set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
end

set(0, 'CurrentFigure', 10);
set(gcf, 'doublebuffer', 'on');

subplot(2, 2, 1);
set(gca, 'Units', 'normalized', 'Position', [0 0.505 0.495 0.5])

imagesc(image);
hold on; axis ij; axis off;
text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 15);
hold off;

subplot(2, 2, 2);
set(gca, 'Units', 'normalized', 'Position', [0.505 0.505 0.495 0.495])
imagesc(foreground_hued_image);
hold on; axis ij; axis off;
%contour(foreground_map, 1, 'r');
text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 15);
hold off;

subplot(2, 2, 3);
set(gca, 'Units', 'normalized', 'Position', [0 0 0.495 0.495])
imagesc(best_background);
hold on; axis ij; axis off;
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
	  objects(obj).bbox_y(1)], obj_colour, 'Linewidth', 2);
  end
      
text(objects(obj).bbox_x(2) + 5, objects(obj).bbox_y(1) - 5, ...
     sprintf('%d', objects(obj).label), 'Color', obj_colour, ...
     'BackgroundColor', [0.5, 0.5, 0.5], 'Fontsize', 20);
  end

hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw the input frame on its own figure. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set(0, 'CurrentFigure', 11);

imagesc(image);
hold on; axis ij; axis off;
text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 30);
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw the output frame on its own figure. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set(0, 'CurrentFigure', 12);

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

text(10, 10, sprintf('%04d', tt), 'Color', 'r', 'Fontsize', 30);

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
	  objects(obj).bbox_y(1)], obj_colour, 'Linewidth', 3);
  end
      
text(objects(obj).bbox_x(2) + 5, objects(obj).bbox_y(1) - 5, ...
     sprintf('%d', objects(obj).label), 'Color', obj_colour, ...
     'BackgroundColor', [0.5, 0.5, 0.5], 'Fontsize', 30, 'Fontweight', 'bold');
  end

hold off;
drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now save the three frames
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

exportfig(10, sprintf('../summer/stgeorge/stages/stages_frame_%05d.jpg', tt), 'Color', 'cmyk', 'Format', 'jpeg');
exportfig(11, sprintf('../summer/stgeorge/input/input_frame_%05d.jpg', tt), 'Color', 'cmyk', 'Format', 'jpeg');
exportfig(12, sprintf('../summer/stgeorge/output/output_frame_%05d.jpg', tt), 'Color', 'cmyk', 'Format', 'jpeg');
