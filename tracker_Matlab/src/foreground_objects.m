function [objects_map, num_objects, object_sizes, object_positions] = foreground_objects(foreground_map, num, size_threshold)
% [objects_map, num_objects, object_sizes, object_positions] = foreground_objects(foreground_map, num, size_threshold)
% Compute a map of object silhouettes using a connected components algoritm and 
% thresholding at size_threshold. foreground_map is a binary matrix
% indicating foreground pixels with unit-entries. num can take a value
% of 4 or 8 to select the 4- or 8-connected objects algorithm. The
% silhouettes will be indicated by different integers in the objects
% map. Object centroids are returned in object positions, where each
% column is a coordinate vector. 

global HEIGHT WIDTH;

objects_map = zeros(size(foreground_map), 'int32');
object_sizes = [];
object_positions = [];

new_label = 1;
[label_map, num_labels] = bwlabel(foreground_map, num);

for label = 1:num_labels
  object = (label_map == label);
  object_size = sum(sum(object));
  if(object_size >= size_threshold)
    % Component is big enough, mark it
    objects_map = objects_map + int32(object * new_label);
    object_sizes(new_label) = object_size;

    [X, Y] = meshgrid(1:WIDTH, 1:HEIGHT);    
    object_x = X .* object;
    object_y = Y .* object;
    
    object_positions(:, new_label) = [sum(sum(object_x)) / object_size;
				      sum(sum(object_y)) / object_size];

    new_label = new_label + 1;
  end
end

num_objects = new_label - 1;
