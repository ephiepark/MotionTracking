alpha =  0.6913;

w_init = [0.1 0.9]
w = w_init;
sigma_sq = [100, 90]

[junk, ind_init] = sort(w ./ sqrt(sigma_sq), 'descend');
junk
ind_init

alpha_list = [];

for sig = 110:1000:300000
w = w + alpha .* (([1:2] == ind_init(1)) - w);
w = w ./ sum(w);
  sigma_sq(ind_init(1)) = sig;
  %sigma_sq
  [junk, ind] = sort(w ./ sqrt(sigma_sq), 'descend');
%  junk 
%  ind

  w = w_init;
  % Find better alpha
  alpha = (w(ind_init(2)) * (sqrt(sigma_sq(ind_init(1))) / sqrt(sigma_sq(ind_init(2)))) - w(ind_init(1))) / ...
(1 - w(ind_init(1)));
  
  sigma_ratio = sqrt(sigma_sq(ind_init(1))) / sqrt(sigma_sq(ind_init(2)));
  alpha = (w(ind_init(2)) * sigma_ratio - w(ind_init(1))) / ...
  (w(ind_init(2)) * sigma_ratio - w(ind_init(1)) + 1);

  alpha_list(end+1) = alpha;
end
