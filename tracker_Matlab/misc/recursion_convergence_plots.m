% For the recusive equation to adapt means and covariances, plot the
% number of iterations required to get within epsilon error of the 
% final answer. It does not matter if we use the relative residual or a
% fixed error bound, as the magnitude would cancel. 

start = 0;

alpha_vec = 0.01:0.01:0.99;
final_vec = 10:10:1000;

epsilon = 0.001;

for alpha_ind = 1:length(alpha_vec)
  alpha = alpha_vec(alpha_ind);
  for final_ind = 1:length(final_vec)
    final = final_vec(final_ind);
    
    % Predict the time it takes for the mu to reach within a certain error
    t_mu(alpha_ind, final_ind) = log(epsilon / (final - start)) / log(1 - alpha);
  end
end

figure(1);
imagesc(log(t_mu));
colormap(gray);
colorbar;

title(sprintf('Logarith of time required to reach C within epsilon = %d error\n', epsilon));
xlabel('C');
ylabel('alpha');
XTick = get(gca, 'XTick');
YTick = get(gca, 'YTick');
set(gca, 'XTickLabel', final_vec(XTick));
set(gca, 'YTickLabel', alpha_vec(YTick));

stddev = 1;

for alpha_ind = 1:length(alpha_vec)
  alpha = alpha_vec(alpha_ind);
  for final_ind = 1:length(final_vec)
    final = final_vec(final_ind);

    tt = 0;
    maxsigma = 0;
    maxtime = 0;
    sigma_t_minus_1 = 1; % Start with unit standard deviation
    start_sigma = sigma_t_minus_1;
    start_mu = start;

    % The second recursion is hard to evaluate in closed form. We'll
    % evaluate and check directly when convergence is reached. 
    
    while(1)
      tt = tt + 1;

      % Mean value at time tt
      mu_t = (1 - alpha)^tt * (start - final) + final;
 
      % Sigma value at time tt
      sigma_t = (1 - alpha) * sigma_t_minus_1 + alpha * (mu_t - final)^2;
%      sigma_predicted = (1 - alpha)^tt * start_sigma + (start_mu - final)^2 * (1 - alpha)^(tt+1) * (1 - (1 - alpha)^tt)

      sigma_t_minus_1 = sigma_t;

      % Record maximum if we hit it
      if(sigma_t > maxsigma)
	maxtime = tt;
	maxsigma = sigma_t;
      end
      
      % Stop iterating when we achieve less than unit standard deviation. 
      if(sqrt(sigma_t) < stddev);
	break;
      end
    end

    % Record data in matrices
    % Time required to converge
    t_sigma(alpha_ind, final_ind) = tt;
    % Maximum value reached
    maxval_sigma(alpha_ind, final_ind) = maxsigma;
    % Time when maximum value was reached
    maxtime_sigma(alpha_ind, final_ind) = maxtime;
  end
end

figure(2);
subplot(1, 3, 1);
imagesc(log(t_sigma));
colormap(gray);
colorbar;

title('Logarithm of convergence time');
xlabel('C');
ylabel('alpha');
XTick = get(gca, 'XTick');
YTick = get(gca, 'YTick');
set(gca, 'XTickLabel', final_vec(XTick));
set(gca, 'YTickLabel', alpha_vec(YTick));

subplot(1, 3, 2);
imagesc(maxval_sigma);
colormap(gray);
colorbar;
title('Logarithm of the maximum value reached during adaption');
xlabel('C');
ylabel('alpha');
XTick = get(gca, 'XTick');
YTick = get(gca, 'YTick');
set(gca, 'XTickLabel', final_vec(XTick));
set(gca, 'YTickLabel', alpha_vec(YTick));

subplot(1, 3, 3);
imagesc(log(maxtime_sigma));
colormap(gray);
colorbar;
title('Logarithm of the time when the maximum value was reached');
xlabel('C');
ylabel('alpha');
XTick = get(gca, 'XTick');
YTick = get(gca, 'YTick');
set(gca, 'XTickLabel', final_vec(XTick));
set(gca, 'YTickLabel', alpha_vec(YTick));

% This shows that the two time constants are almost equivalent. 
figure(3)
% Residual between time constants
resid = abs(1 ./ [0.0001:0.0001:1] - abs(1 ./ log(ones(1, 10000) - [0.0001:0.0001:1])))
% Relative residual
relresid = resid .* 0.0001:0.0001:1
title('Absolute residual between the two time constants');
plot(resid)

figure(4);
title('Relative residual between the two time constants');
plot(relresid)

figure(5);
imagesc(maxval_sigma);
%colormap(gray);
colorbar;
title('Logarithm of maximum variance');
xlabel('C');
ylabel('alpha');
XTick = get(gca, 'XTick');
YTick = get(gca, 'YTick');
set(gca, 'XTickLabel', final_vec(XTick));
set(gca, 'YTickLabel', alpha_vec(YTick));
