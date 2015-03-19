function [] = pixel_mixture_stat(fig_mixture_stat, debug_coord, data, tt, ...
		 matched_gaussian, replaced_gaussian, ...
				 background_gaussians, mixparam); ...
% [] = pixel_mixture_stat(fig_mixture_stat, debug_coord, data, tt, ...
%      matched_gaussian, replaced_gaussian, background_gaussians, mixparam)
% Arguments are a figure handle, the pixel-coordinate of the debugging
% pixel the data-matrix of all images, the time index, a binary
% indicator matrix of which Gaussian matched at each pixel, an indicator
% matrix of replaced Gaussians, a binary
% indicator matrix of all background Gaussians at each pixel, and the
% parameter structure used to update the mixtures with. 

  global HEIGHT WIDTH C;
  global Mus Sigmas Weights;
  
  persistent prev_coord last prev_data prev_mean prev_deviation;
  
  % How far do the timeplots reach back in history. 
  NPREVPOINTS = 150;

  % Display every second deviation as error bar in the time plots. 
  NTH_DEVIATION = 15; 

  % Display timeplots for a specific channel.
  CHANNEL = 1;

  % If this is the first call to this function, initialise persistent data
  % Also initialise the data if the coordinate has changed. 
  if(isempty(last) || any(prev_coord ~= debug_coord))
    prev_coord = debug_coord;

    last = 0;
    
    % Inf marks invalid data
    prev_data = ones(1, NPREVPOINTS) * inf;
    prev_mean = ones(mixparam.K, NPREVPOINTS) * inf;
    prev_deviation = ones(mixparam.K, NPREVPOINTS) * inf;
  end

  last = mod(last, NPREVPOINTS) + 1;

  % The pixel we will monitor is indexed by pixel_index. 
  pixel_index = sub2ind([HEIGHT, WIDTH], debug_coord(1, 2), debug_coord(1, 1));

  set(0, 'CurrentFigure', fig_mixture_stat);

  if(background_gaussians(pixel_index, find(matched_gaussian(pixel_index, :))))
    suptitle(sprintf('Frame %04d: pixel (%d, %d) is background\n', tt, debug_coord(1, 2), debug_coord(1, 1)));
  else
    suptitle(sprintf('Frame %04d: pixel (%d, %d) is foreground\n', tt, debug_coord(1, 2), debug_coord(1, 1)));
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Draw the mean time plots for one channel                     %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Insert new data, we only deal with one channel
  prev_data(last) = data(pixel_index, CHANNEL, tt);
  prev_mean(:, last) = Mus(pixel_index, :, CHANNEL);
  prev_deviation(:, last) = sqrt(Sigmas(pixel_index, :, CHANNEL));

  % Draw the data
 
  % Get a vector of indeces that has the order of points permuted
  index = [last+1:NPREVPOINTS, 1:last];

  for kk = 1:mixparam.K
    subplot(mixparam.K, 2, (kk) * 2 - 1);
    % A specific colour channel of input data over time
    plot((1:NPREVPOINTS) + tt - NPREVPOINTS, prev_data(index), 'r-');
    hold on;
    % Don't plot all deviations as error bars. Pick every NTH_DEVIATION one
    nth_deviations = prev_deviation(kk, index) .* ~mod(index, NTH_DEVIATION);
    errorbar((1:NPREVPOINTS) + tt - NPREVPOINTS, prev_mean(kk, index), nth_deviations, 'k-');
    ylabel('Red component');
    xlabel('Frame');
    hold off;
    set(gca, 'XLim', [tt - NPREVPOINTS + 1, tt]);
  end
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Draw the statistics visualisation                              %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  for kk = 1:mixparam.K
    % A blank image
    image = ones(20, 55, 3, 'uint8') * 255;

    for cc = 1:C
      image(1:20, 1:30, cc) = ones(20, 30, 1) * Mus(pixel_index, kk, cc);
    end

    % If the component belongs to the background draw a blue frame
    % around it. Otherwise draw a black frame around it. 
    if(background_gaussians(pixel_index, kk) == 1)
      image(1:20, 1, 3) = ones(20, 1, 1, 'uint8') * 255;
      image(1:20, 1, 1:2) = zeros(20, 1, 2, 'uint8');
      image(1:20, 55, 3) = ones(20, 1, 1, 'uint8') * 255;
      image(1:20, 55, 1:2) = zeros(20, 1, 2);

      image(1, 1:55, 3) = ones(1, 55, 1, 'uint8') * 255;
      image(1, 1:55, 1:2) = zeros(1, 55, 2, 'uint8');
      image(20, 1:55, 3) = ones(1, 55, 1, 'uint8') * 255;
      image(20, 1:55, 1:2) = zeros(1, 55, 2, 'uint8');
    else
      image(1:20, 1, :) = zeros(20, 1, 3, 'uint8');
      image(1:20, 55, :) = zeros(20, 1, 3, 'uint8');

      image(1, 1:55, :) = zeros(1, 55, 3, 'uint8');
      image(20, 1:55, :) = zeros(1, 55, 3, 'uint8');
    end
    
    % If the component replaced another component then mark it red with
    % another frame, one pixel further in. 
    if(replaced_gaussian(pixel_index, kk))
      image(2:19, 2, 1) = ones(18, 1, 1, 'uint8') * 255;
      image(2:19, 2, 2:3) = zeros(18, 1, 2, 'uint8');
      image(2:19, 54, 1) = ones(18, 1, 1, 'uint8') * 255;
      image(2:19, 54, 2:3) = zeros(18, 1, 2);

      image(2, 2:54, 1) = ones(1, 53, 1, 'uint8') * 255;
      image(2, 2:54, 2:3) = zeros(1, 53, 2, 'uint8');
      image(19, 2:54, 1) = ones(1, 53, 1, 'uint8') * 255;
      image(19, 2:54, 2:3) = zeros(1, 53, 2, 'uint8');
    else % Or mark it green 
      if(matched_gaussian(pixel_index, kk))
	image(2:19, 2, 2) = ones(18, 1, 1, 'uint8') * 255;
	image(2:19, 2, [1, 3]) = zeros(18, 1, 2, 'uint8');
	image(2:19, 54, 2) = ones(18, 1, 1, 'uint8') * 255;
	image(2:19, 54, [1, 3]) = zeros(18, 1, 2);

	image(2, 2:54, 2) = ones(1, 53, 1, 'uint8') * 255;
	image(2, 2:54, [1, 3]) = zeros(1, 53, 2, 'uint8');
	image(19, 2:54, 2) = ones(1, 53, 1, 'uint8') * 255;
	image(19, 2:54, [1, 3]) = zeros(1, 53, 2, 'uint8');
      end
    end
    
    subplot(mixparam.K, 2, (kk) * 2);
    imagesc(image);
    axis off;
    hold on;
    
%    format short;

    t = text(5, 1 * 5 + 1, sprintf('R: %.3f', Mus(pixel_index, kk, 1)));
    t = text(35, 1 * 5 + 1, sprintf('R: %.3f', single(Sigmas(pixel_index, kk, 1))));
    
    t = text(5, 2 * 5 + 1, sprintf('G: %.3f', Mus(pixel_index, kk, 2)));
    t = text(35, 2 * 5 + 1, sprintf('G: %.3f', single(Sigmas(pixel_index, kk, 2))));
    
    t = text(5, 3 * 5 + 1, sprintf('B: %.3f', Mus(pixel_index, kk, 3)));
    t = text(35, 3 * 5 + 1, sprintf('B: %.3f', single(Sigmas(pixel_index, kk, 3))));
    
    if(background_gaussians(pixel_index, kk) == 1)
      % Background model
      title(sprintf('[BG] w_%d: %f', kk, Weights(pixel_index, kk)));
    else
      % Foreground model
      title(sprintf('[FG] w_%d: %f', kk, Weights(pixel_index, kk)));
    end
    
    %  title('Mean                   Variance');
    hold off;
end

drawnow;
