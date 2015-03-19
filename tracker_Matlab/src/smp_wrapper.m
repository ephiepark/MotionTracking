function [assignments] = smp_wrapper(costs, dummy)
% [assignments] = smp_wrapper(costs, dummy)
% Wrapper to call the Stable Marriage Algorithm. Avoids unlikely matches
% by extending the matrix with dummy entries. 

% For the Stable Marriage Problem add one extra dummy row for each existing 
% Kalman Filter which will help us prevent unwanted matches. 

smp_costs = [costs; ones(size(costs)) * dummy];

% Run the SMP assignment algorithm. 
assignments = stable_marriage_matching(smp_costs);

assignments = assignments(1:end/2, :); % Crop out the dummy matches
