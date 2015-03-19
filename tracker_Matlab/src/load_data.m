function [data] = load_data(path, extention)
% [data] = load_data(path, extention)
% Load a set of images with given extention from a directory given in path
% Returns a data matrix of images. The first dimension indexes
% individual pixels, the second the three RGB components and the third the
% frame number. 

% A few globals
global HEIGHT WIDTH C D T 

% Fix up the path if slash was ommitted

if(path(end) ~= '/')
   path = [path '/'];
end

fprintf(1, 'Loading files from %s...\n', path);

files = dir([path '*.' extention]);
files = sort({files.name});

% Now read in every image

% Number of frames
T = size(files, 2);

% Get the dimension of one picture
sizes = size(imread([path files{1}]));

% If we only have greyscale
if(length(sizes) == 2)
  sizes(3) = 1; % Pretend we have a third dimension
end 

data = zeros([sizes, T], 'uint8');

for tt = 1:T
  fprintf(1, 'Reading file %d: %s\r', tt, files{tt});
  im = imread([path files{tt}]);
  data(:, :, :, tt) = im;
end

% If the data was grayscale then fill the remaining two dimensions with
% the same data. A quick way to get the code working on grayscale
% images. 

if(sizes(3) == 1) 
  data = repmat(data, [1, 1, 3, 1]);
end 

% How many colours has each pixel got (should be 3)
C = size(data, 3);

% Image dimensions
HEIGHT = size(data, 1);
WIDTH = size(data, 2);

% After reshaping, how many dimensions has each datapoint got
D = HEIGHT * WIDTH;

% Turn each picture into a column vector of color triplets
data = reshape(data, size(data, 1) * size(data, 2), size(data, 3), ...
               size(data, 4));

fprintf(1, '\n');
