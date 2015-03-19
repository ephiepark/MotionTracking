opengl neverselect

cutoffs = 1:50:201;
for cut = 1:length(cutoffs)
  for ii = 1:length(alpha)
    for jj = 1:length(rho)
      average(ii, jj) = sum(errors(ii, jj, cutoffs(cut):end)) / ...
	(size(errors, 3) - cutoffs(cut) + 1);
    end
  end

  [val, ind] = min(log(average(:)));
  [a, r] = ind2sub(size(average), ind);

  imagesc(log(average));
  axis ij;
  colormap('gray');
%  caxis([5 10]);
  caxis([5.5 6.2]);
  title({sprintf('Log of average errors for  frames %03d-%03d', ...
		 cutoffs(cut), size(errors, 3)), ...
	 sprintf('Minimum log error %f at alpha = %f, rho = %f', ...
		 val, alpha(a), rho(r))});
  ylabel('alpha');
  xlabel('rho');

  XTick = get(gca, 'XTick');
  YTick = get(gca, 'YTick');
  rho(XTick(1:2:end))
  set(gca, 'XTick', XTick(1:2:end));
  set(gca, 'XTickLabel', rho(XTick(1:2:end)));
  set(gca, 'YTickLabel', alpha(YTick));

  colorbar;
  
  drawnow;
  saveas(1, sprintf('param_eval_flat_%d-%d.eps', ...
		    cutoffs(cut), size(errors, 3)), 'psc2');

  [R, A] = meshgrid(rho, alpha);
  surf(R, A, log(average));
  hold on;
%  caxis([5 10]);
  caxis([5.5 6.1]);
  view(160, 32);
  title({sprintf('Log of average errors for  frames %03d-%03d', ...
		 cutoffs(cut), size(errors, 3)), ...
	 sprintf('Minimum log error %f at alpha = %f, rho = %f', ...
		 val, alpha(a), rho(r))})
    log(average(a, r))
    scatter3(rho(r), alpha(a), log(average(a, r)), 'ro', 'fill');
    ylabel('alpha');
    xlabel('rho');
    colorbar
    hold off;
    drawnow;
    
    saveas(1, sprintf('param_eval_3d_%d-%d.eps', ...
		      cutoffs(cut), size(errors, 3)), 'psc2');

end
