function [] = display_data_and_track(data, HEIGHT, WIDTH, C,  trajectories, scale, start);
% [] = display_data_and_track(data, HEIGHT, WIDTH, C,  trajectories, scale, start);
% Display a frame sequence on screen, C is the number of colour components

T = size(data, 3);

for tt = start:T
  imagesc(reshape(data(:, :, tt), HEIGHT, WIDTH, C));
  text(10, 10, sprintf('%04d', tt), 'Color', 'r');
  axis off
  hold on
  for tr = 1:length(trajectories)
    if(size(trajectories(tr).pos, 2) > tt)
      if(scale)
	pos = trajectories(tr).pos(:, tt);
	pos = pos ./ 4.8;
	pos(2) = 120 - pos(2);

	scatter(pos(1), pos(2), 'ro');
      else
	scatter(trajectories(tr).pos(1, tt), trajectories(tr).pos(2, tt), 'ro');
      end
    end
  end
  hold off;
  drawnow;
end
