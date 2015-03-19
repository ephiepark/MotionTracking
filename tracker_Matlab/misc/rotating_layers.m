function [frames] = rotating_layers(object_hist)

global HEIGHT WIDTH

if(isempty(HEIGHT) || isempty(WIDTH))
  warning('HEIGHT and WIDTH not globally defined');
end

fig_layers = figure(1);
nframes = size(object_hist, 2)

imagenum = 1;

% First run no rotation
for tt = 1:nframes
  objects = object_hist(tt).objects;
  background = object_hist(tt).background;

  display_sprites_stacked(fig_layers, background, objects, ...
			 size(objects, 2):-1:1, [0, 0]);

  imagenum = imagenum + 1;
  saveas(fig_layers, sprintf('synthetic_rotating_%05d.jpg', imagenum), 'jpg');
end

az = 0;
el = 0;

for repeat = 1:4
  for tt = 1:nframes
    objects = object_hist(tt).objects;
    background = object_hist(tt).background;

    if(el < 15)
      el = el + 0.2;
    end

    az = az + 2;

    display_sprites_stacked(fig_layers, background, objects, ...
			   size(objects, 2):-1:1, [az, el]);
     
    imagenum = imagenum + 1;
    saveas(fig_layers, sprintf('/tmp/synthetic_rotating_%05d.jpg', imagenum), 'jpg');
  end
end

