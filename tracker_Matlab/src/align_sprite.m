function [alignment, sprite_x, sprite_y] = align_sprite(siz, sprite, offset)
%[alignment, sprite_x, sprite_y] = align_sprite(siz, sprite, offset)
% Align a sprite within a matrix of zeros, of size siz using the given
% offset. The offset specifies where in the canvas the sprite center
% should be placed and must thus not lie outwith the dimensions specified
% in siz. Parts of the sprite that lie outwith the canvas after
% alignment are cropped away. siz is a two or three dimensional vector
% specifying the size of the final alignment. offset is always
% two-dimensional and only applies to the first two dimensions if siz is
% three dimensional. sprite_x and sprite_y are vectors that record the
% extent of the aligned and cropped sprite. 

sprite_height = size(sprite, 1);
sprite_width = size(sprite, 2);

if((length(siz) ~= 2) && (length(siz) ~= 3))
  error('siz must be a vector of length two or three');
end

% Oversized canvas. Align sprite in this matrix and then crop to
% normal size.

if(length(siz) == 2)
  alignment = zeros(siz(1) + 2 * ceil(sprite_height / 2), siz(2) + 2 * ceil(sprite_width / 2));
end

if(length(siz) == 3)
  alignment = zeros(siz(1) + 2 * ceil(sprite_height / 2), siz(2) + 2 * ceil(sprite_width / 2), siz(3));
end

sprite_x(1) = int32(ceil(sprite_width / 2) + offset(1) - ceil(sprite_width / 2));
sprite_x(2) = sprite_x(1) + sprite_width - 1;  

sprite_y(1) = int32(ceil(sprite_height / 2) + offset(2) - ceil(sprite_height / 2));
sprite_y(2) = sprite_y(1) + sprite_height - 1;

% Align the sprite

if(length(siz) == 2)
  alignment(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2)) = sprite;
end

if(length(siz) == 3)
  % Offset only operates on first two dimensions
  alignment(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2), :) = sprite;
end

%  appearance(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2), :) = objects(obj).appearance;
%  occupancy(sprite_y(1):sprite_y(2), sprite_x(1):sprite_x(2)) = objects(obj).occupancy;
  
xcropstart = ceil(sprite_width / 2);
xcropend = xcropstart + siz(2) - 1;
ycropstart = ceil(sprite_height / 2);
ycropend = ycropstart + siz(1) - 1;

% Crop the alignment to final size
if(length(siz) == 2)
  alignment = alignment(ycropstart:ycropend, xcropstart:xcropend);
end

if(length(siz) == 3)
  alignment = alignment(ycropstart:ycropend, xcropstart:xcropend, :);
end

% After cropping the displacement parameters have changed. Update all structure fields.
sprite_x(1) = max(1, int32(offset(1) - ceil(sprite_width / 2)) + 1);
sprite_x(2) = min(siz(2), sprite_x(1) + sprite_width - 1);

sprite_y(1) = max(1, int32(offset(2) - ceil(sprite_height / 2)) + 1);
sprite_y(2) = min(siz(1), sprite_y(1) + sprite_height - 1);
