function [] = store_pixel_coord(currpoint)
% [] = store_pixel_coord(currpoint)
% Helper function to update a global coordinate vector. 

global HEIGHT WIDTH debug_coord;

coord = [currpoint(1, 1), currpoint(1, 2)];

if((~any(coord < 0)) && (coord(1) <= WIDTH) && (coord(2) <= HEIGHT))
  debug_coord = int32(coord);
end
