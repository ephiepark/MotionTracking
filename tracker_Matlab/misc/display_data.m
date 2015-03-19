function [] = display_data(data, HEIGHT, WIDTH, C);
% [] = display_data(data, HEIGHT, WIDTH, C);
% Display a frame sequence on screen, C is the number of colour components

T = size(data, 3);

for tt = 1:T
  imagesc(reshape(data(:, :, tt), HEIGHT, WIDTH, C));
  text(10, 10, sprintf('%04d', tt), 'Color', 'r');
  drawnow;
end
