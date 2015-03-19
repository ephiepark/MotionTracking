global HEIGHT WIDTH D C T

T = 400;
HEIGHT = 120;
WIDTH = 160;
D = HEIGHT * WIDTH;
C = 3;

SIDELENGTH = 10;
NOBJECTS = 5;

OFFSETS = [ones(1, floor(T/2)) * 80, ones(1, T - ceil(T/2)) * 110];
OFFSETS = OFFSETS(randperm(T));

data = zeros(HEIGHT * WIDTH, C, T, 'uint8');
groundtruth = zeros(HEIGHT, WIDTH, T, 'uint8');

% Initialise object positions to illegal values
pos(1:2, 1:NOBJECTS) = [ones(1, NOBJECTS) * (WIDTH + 1);
			zeros(1, NOBJECTS)];

object_vel = zeros(2, NOBJECTS);

for tt = 1:T
  fprintf(1, 'Generating frame %d\n', tt);
  
  % Get a random background noise
  frame = OFFSETS(tt) + randn(HEIGHT, WIDTH, C) * 2;
  foreground = zeros(HEIGHT, WIDTH, 'uint8');

  for obj = 1:NOBJECTS
    pos(:, obj) = pos(:, obj) + object_vel(:, obj);
    
    pos_int = int32(pos(:, obj));
    
    if((pos_int(1) > WIDTH) || (pos_int(2) < 1) || (pos_int(2) > HEIGHT))
      object_vel(1, obj) = rand(1) * 3;
      object_vel(2, obj) = (rand(1) - 0.5) * 3;
      
      % Initialise positions for next run
      pos(:, obj) = [double(1);
		     ceil(rand(1) * HEIGHT)];

      pos_int = int32(pos(:, obj));
    end
        
    frame(pos_int(2):pos_int(2)+SIDELENGTH-1, pos_int(1):pos_int(1)+SIDELENGTH-1, :) = ...
      ones(SIDELENGTH, SIDELENGTH, C) * 50;
  
    foreground(pos_int(2):pos_int(2)+SIDELENGTH-1, pos_int(1):pos_int(1)+SIDELENGTH-1, :) = ...
      ones(SIDELENGTH, SIDELENGTH);
  end

  frame = frame(1:HEIGHT, 1:WIDTH, :);
  foreground = foreground(1:HEIGHT, 1:WIDTH);

  groundtruth(:, :, tt) = foreground;
  data(:, :, tt) = reshape(frame, HEIGHT * WIDTH, C);
end
