function [] = display_sprites_tiled(fig_sprites_tiled, image, background, BgOccupancy, objects)
% [] = display_sprites_tiled(fig_sprites_tiled, background, BgOccupancy, objects)
% Given a figure handle, an image representation of the background
% layer the background occupancy map and an array of objects, display a
% tiled representation of occupancy maps and appearance maps for each
% layer/sprite. The background occupancy is BgOccupancy and objects is a
% structure array of tracked objects. 

global HEIGHT WIDTH C

nobjects = size(objects, 2);

set(0, 'CurrentFigure', fig_sprites_tiled);

subplot(3, 1 + nobjects, 1);
imagesc(reshape(image, HEIGHT, WIDTH, C));
title('Background');
axis off;

subplot(3, 1 + nobjects, 1 + nobjects + 1);
imagesc(reshape(BgOccupancy, HEIGHT, WIDTH));
caxis([0, 1]);
axis off;

subplot(3, 1 + nobjects, 2 * (1 + nobjects) + 1);
imagesc(uint8(background));
axis off;

for obj = 1:nobjects
  % subplot(3, 1 + nobjects, 1 + obj);
  % imagesc(reshape(posterior(:, 1 + obj), HEIGHT, WIDTH));
  % % caxis([0, 1]);
  % axis off;

  % Display an image of the true object appearance
  %  [alignment, sprite_x, sprite_y] = align_sprite([HEIGHT, WIDTH], objects(obj).occupancy, ...
  %		     objects(obj).expected_y_t_plus_1_given_t(1:2));

  sprite_y = objects(obj).sprite_y;
  sprite_x = objects(obj).sprite_x;

  input_appearance = reshape(image, HEIGHT, WIDTH, C);
  input_appearance = input_appearance(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2), :);

  subplot(3, 1 + nobjects, 1 + obj);
  imagesc(input_appearance);
  title(sprintf('Sprite %d', objects(obj).label));
  axis off;

  % Display the occupancy of our sprite
  subplot(3, 1 + nobjects, 2 + nobjects + obj);
  imagesc(objects(obj).occupancy);
  hold on;
  contour(objects(obj).occupancy, 0.5, 'r');
  hold off;
  caxis([0, 1]);
  axis off;

  subplot(3, 1 + nobjects, 2 * (1 + nobjects) + 1 + obj);
  imagesc(uint8(objects(obj).appearance));
  axis off;
end

drawnow;
