function [assignments, gate] = data_association(objects, pos_observed, object_sizes, kalparam)
% [assignments, gate] = data_association(objects, pos_observed, object_sizes, kalparam)
% Helper function to associate observed blobs with tracked
% objects. objects is a structure array of the different objects being
% tracked. pos_observed and object_sizes are the centroids and sizes of
% the observed blobs. kalparam is the parameter structure for the
% tracking subsystem. See kalman_parameters.m for details. 
% Depending on parameters in the structure, different cost functions and
% association algorithms are used. 

% Are we using distance-based costs
if(strcmp(kalparam.ASSOC_COST_TYPE, 'distance'))
 costs = distance_costs(objects, pos_observed);
end

% Or costs based on the predictive observation distribution. 
if(strcmp(kalparam.ASSOC_COST_TYPE, 'kalman_expectation'))
  costs = kalman_costs(objects, pos_observed, object_sizes);
end

% Return an indicator matrix of those entries we considered good. 
gate = uint8(costs < kalparam.ASSOC_DUMMY_COST);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stable Marriage Problem data association: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(strcmp(kalparam.ASSOC_ALG_TYPE, 'SMP'))
  assignments = smp_wrapper(costs, kalparam.ASSOC_DUMMY_COST);  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linear Assignment Problem data association. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(strcmp(kalparam.ASSOC_ALG_TYPE, 'LAP'))
  assignments = lap_wrapper(costs, kalparam.ASSOC_DUMMY_COST);
end

if(strcmp(kalparam.ASSOC_ALG_TYPE, 'MUNK'))
  assignments = munkres_wrapper(costs, kalparam.ASSOC_DUMMY_COST);
end
