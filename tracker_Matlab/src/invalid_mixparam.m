function invalid = invalid_parameters(parameters)
% invalid = invalid_parameters(parameters)
% Sanity check the parameter structure for automatic optimisation
% methods. For example, it constrains the alpha parameter to be between 0
% and 1. Returns 0 if all parameters are valid, and 1 otherwise. 

invalid = 0;

if(parameters.K <= 0)
  invalid = 1;
end

if((parameters.ALPHA <= 0) || (parameters.ALPHA >= 1))
  invalid = 1;
end

if((parameters.RHO <= 0) || (parameters.RHO >= 1))
  invalid = 1;
end

if(parameters.DEVIATION_SQ_THRESH <= 0)
  invalid = 1;
end

if(parameters.INIT_VARIANCE <= 0)
  invalid = 1;
end

if(parameters.INIT_MIXPROP <= 0)
  invalid = 1;
end

if(parameters.COMPONENT_THRESH <= 0)
  invalid = 1;
end

if((parameters.BACKGROUND_THRESH <= 0) || (parameters.BACKGROUND_THRESH >= 1))
  invalid = 1;
end
