function indicator = positions_within_window(positions, x, y)
% indicator = positions_within_window(positions, x, y)
% Return an indicator vector, with each element indicating whether or not
% the position is within the window specified by x and y. 
% x and y both are vectors recording the first minimal and maximal
% window coordinates in x and y direction. 

npos = size(positions, 2);

% Must satisfy two constraints
constraints = (positions >= repmat([x(1); y(1)], 1, npos)) & (positions <= repmat([x(2); y(2)], 1, npos));

indicator = constraints(1, :) & constraints(2, :);
