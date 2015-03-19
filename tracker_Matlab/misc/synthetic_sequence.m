function [data] = synthetic_sequence(HEIGHT, WIDTH, T, objects)
% [data] = synthetic_data(HEIGHT, WIDTH, T, obj1, obj2)
% Generate a sequence of T frames of HEIGHT and WIDTH, with two moving 
% objects, where objects is an array of structs with fields
% sidelength: object is a square of given sidelength
% position: a 2-vector with inital x and y components
% velocity: a 2-vector with x and y velocity components
% meancolour: a 3-vector of the mean colour used for each object pixel
% Objects are only added after the first blankframes frames. 

D = HEIGHT * WIDTH;
C = 3; % Assume colour output
NOBJECTS = size(objects, 2);

% Background offsets for the bi-modal background. Each pixel in the
% background is sampled from a Gaussian centered at either 
% [80 80 80] or [120 120 120]. 
BGOFFSETS = [80, 120];

% Preallocate memory
data = zeros(HEIGHT * WIDTH, C, T, 'uint8');
groundtruth = zeros(HEIGHT, WIDTH, T, 'uint8');

dyn_ind = ones(length(objects), 1);

for tt = 1:T
  fprintf(1, 'Generating frame %d\n', tt);

  % For each pixel, randomly pick one of the two backgrounds
  select = rand(HEIGHT, WIDTH) > 0.5;

  % Create background on which we paste objects
  background = repmat(select, [1, 1, C]) * BGOFFSETS(1) + ...
      repmat(abs(select - 1), [1, 1, C]) * BGOFFSETS(2);

  frame = reshape(background, HEIGHT, WIDTH, C) + randn(HEIGHT, WIDTH, C) * 2;
  foreground = zeros(HEIGHT, WIDTH, 'uint8');

  % Only begin adding objects after blankframes frames
  for obj = 1:NOBJECTS
    % See if we should be switching to the next dynamic
    if(dyn_ind(obj) < length(objects(obj).dynamics_start)) 
      % Switch to next dynamic if we reached the frame. 
      if(objects(obj).dynamics_start(dyn_ind(obj) + 1) == tt)
	dyn_ind(obj) = dyn_ind(obj) + 1;
      end
    end

    % FIXME: the <= sign might have broken simulations on the 
    % two first synthetic sequences. It was < before
    if(objects(obj).dynamics_start(dyn_ind(obj)) <= tt)
      objects(obj).position = objects(obj).position + ...
	  objects(obj).velocities(:, dyn_ind(obj));
      
      objects(obj).velocities(:, dyn_ind(obj)) = ...
	  objects(obj).velocities(:, dyn_ind(obj)) + ...
	  objects(obj).accellerations(:, dyn_ind(obj));

      objects(obj).sidelength = ...
	  objects(obj).sidelength + ...
	  objects(obj).looming(:, dyn_ind(obj));
    end
    
    pos_int = int32(objects(obj).position);
    
    if(((pos_int(1) + objects(obj).sidelength) < 1) || (pos_int(1) > WIDTH))
      continue;
    end
    
    xstart = max(1, pos_int(1));
    xend = min(WIDTH, pos_int(1) + objects(obj).sidelength - 1);
    ystart = max(1, pos_int(2));
    yend = min(HEIGHT, pos_int(2) + objects(obj).sidelength - 1);

    % Does object have a hole? If both values of the holetimes vector
    % are identical, then this if block is not entered. 
    % If the object has a hole during this time, paste some background
    % over the middle region of the object. 
    if((objects(obj).holetimes(1) <= tt) && (tt < objects(obj).holetimes(2)))
      hole_border_width = ceil((objects(obj).sidelength - ...
				objects(obj).holesidelength) / 2);
      
      hole_xstart = max(1, pos_int(1) + hole_border_width);
      hole_xend = min(WIDTH, pos_int(1) + objects(obj).sidelength - hole_border_width - 1);
      hole_width = hole_xend - hole_xstart + 1;
      
      hole_ystart = max(1, pos_int(2) + hole_border_width);
      hole_yend = min(HEIGHT, pos_int(2) + objects(obj).sidelength - hole_border_width - 1);
      hole_height = hole_yend - hole_ystart + 1;

      for cc = 1:C
	height = yend - ystart + 1;
	width = hole_xstart - xstart;

	frame(ystart:yend, xstart:(hole_xstart-1), cc) = ...
  	  objects(obj).meancolour(cc) + randn(height, width) * objects(obj).noiseamp;
	
	height = yend - ystart + 1;
	width = xend - hole_xend;

	frame(ystart:yend, (hole_xend+1):xend, cc) = ...
          objects(obj).meancolour(cc) + randn(height, width) * objects(obj).noiseamp;

	height = hole_ystart - ystart;
	width = xend - xstart + 1;

	frame(ystart:(hole_ystart-1), xstart:xend, cc) = ...
          objects(obj).meancolour(cc) + randn(height, width) * objects(obj).noiseamp;

	height = yend - hole_yend;
	width = xend - xstart + 1;

	frame((hole_yend+1):yend, xstart:xend, cc) = ...
          objects(obj).meancolour(cc) + randn(height, width) * objects(obj).noiseamp;
            
    end
    else
    for cc = 1:C
      frame(ystart:yend, xstart:xend, cc) = objects(obj).meancolour(cc);
    end

    height = yend - ystart + 1;
    width = xend - xstart + 1;
    
    % Add some Gaussian noise
    frame(ystart:yend, xstart:xend, :) = ...
	frame(ystart:yend, xstart:xend, :) + randn(height, width, 3) * objects(obj).noiseamp;
    
    foreground(ystart:yend, xstart:xend, :) = ...
        ones(height, width);
    end
  end
  frame = frame(1:HEIGHT, 1:WIDTH, :);
  foreground = foreground(1:HEIGHT, 1:WIDTH);

  data(:, :, tt) = uint8(reshape(frame, HEIGHT * WIDTH, C));
  groundtruth(:, :, tt) = foreground;
end
