clear 

cd src;
addpath(pwd);
cd lap;
addpath(pwd);
cd ..;
cd misc;
addpath(pwd);
cd ../..;

data = load_data('./data1/datasets/pets_2001_testing_cam1_scaled/', 'png');

mixparam = mixture_parameters();
kalparam = kalman_parameters(); 

% Run algorithm using different settings for the association dummy
% cost. Evaluating the results should tell us how robust the 
% cost-measure is. 

% Dummy costs we evaluate. Depending on the cost measure these can be
% squared distance measures, or negative log probabilities. 

%for tt = 1:100
%  dist_costs(tt) = tt^2;
%end
%kalparam.ASSOC_COST_TYPE = 'distance';

kal_costs = [10:100:10000]
kalparam.ASSOC_COST_TYPE = 'kalman_expectation';

for ii = 1:length(kal_costs)
  fprintf(1, 'Evaluating using dummy cost of %d\n', kal_costs(ii));
  kalparam.ASSOC_DUMMY_COST = kal_costs(ii)

  [junk, object_hist] = track(data, mixparam, kalparam);

  % Save the object history and mixture and kalman parameters. 
  save(sprintf('./data1/tmp/Tracking_kal_cost_%05d.mat', kal_costs(ii)), ...
       'object_hist', 'mixparam', 'kalparam');
end
