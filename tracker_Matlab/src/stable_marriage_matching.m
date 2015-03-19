function [assignments] = stable_marriage_matching(affinities)
% [assignments] = marriage(affinities)
% affinities is a matrix indexed by kalman filters on the rows and
% potential points on the columns. The algorithm returns an assignment
% of points to filters. 
% Algorithm description: http://www.sandelman.ottawa.on.ca/mcr/thesis/subsubsection1.2.0.4.2.2.html

assignments = zeros(size(affinities), 'int32');

% While there are rows that have no match yet and have not checked all potential matched
while(any((sum(assignments == 1, 2) == 0) & (sum(assignments == 0, 2) >= 1)))
% Find one that hasn't been assigned yet
 unchecked = find((sum(assignments == 1, 2) == 0) & (sum(assignments == 0, 2) >= 1));

  kal = unchecked(1); % Pick one

  % Find best match which hasn't been checked yet.
  [junk, index] = sort(affinities(kal, :), 'ascend');

  % Duplicated elements in affinities(kal, :) have different values in a(). 
  a(index) = 1:size(affinities, 2); 

  % Assigned or previously checked values cannot be selected
  a(find(assignments(kal, :) ~= 0)) = inf;

  % Get the minimum element in a()
  [junk, index] = min(a);

  % Is that point already paired?
  if(any(assignments(:, index) == 1)) % yes
    kal2 = find(assignments(:, index) == 1);
    
    % Compare the value of the two and pick better one
    if(affinities(kal, index) < affinities(kal2, index))
      % Then take the new one
      assignments(kal2, index) = -1; % Unpair kal2 and index
      assignments(kal, index) = 1; % Pair kal and index
    else
      assignments(kal, index) = -1; % kal not paired with this one
    end
  else
    assignments(kal, index) = 1; % Not yet paired, pair kal with this one
  end
end

assignments = uint8(assignments == 1);
