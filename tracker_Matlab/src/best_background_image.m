function [background] = best_background_image(Mus, best_background_gaussian)
% [background] = best_background_image(Mus, best_background_gaussian)
% Produce an image of the means of those mixture components which we are most
% confident in using the weight/stddev tradeoff. The component indeces
% are stored in best_background_gaussian, which we need to convert
% (rather ardously) into linear indeces. The first dimension of Mus is
% the (linear) pixel index (D elements), the second the gaussian component
% index (K elements), the last the colour channel index (1-C). 

global HEIGHT WIDTH D C

index = sub2ind(size(Mus), reshape(repmat([1:D], C, 1), D * C, 1), ...
    reshape(repmat(best_background_gaussian', C, 1), D * C, 1), repmat([1:C]', D, 1));

background = reshape(Mus(index), C, D);
background = reshape(background', HEIGHT, WIDTH, C);
