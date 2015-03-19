function [im] = get_streaming_image(locator)
% [im] = get_streaming_image(locator)
% Read an image from a v4l file using the command "streamer" or a
% web-page using "wget". The first three characters of the locator
% string indicate whether it is a v4l device or a website we read from. 
% Example locators: 'v4l:/dev/video0' or 'web:http://www.test.com/image.png'
% Resets HEIGHT WIDTH D (= HEIGHT * WIDTH), and C (3 for RGB) global 
% variables. Returns a matrix that is D-by-C of size, or an empty matrix
% [], if an error occurred. 

global HEIGHT WIDTH D C T

if(strcmp(locator(1:3), 'v4l'))
  fprintf(1, 'Reading v4l image from %s\n', locator(5:end));
  unix(['streamer -c ', locator(5:end), ' -j 90 -s 120x160 -o stream.jpeg > /dev/null 2>&1']);
end

if(strcmp(locator(1:3), 'web'))
  fprintf(1, 'Reading web image from %s\n', locator(5:end));
  unix(['wget -q -T 20 -t 1 -O stream.jpeg ', locator(5:end)]);
end

% Sometimes the images we get are corrupted
try,
  im = imread('stream.jpeg');
catch,
  fprintf(1, 'Caught error while reading image: %s\n', lasterr);
  im = [];
end

HEIGHT = size(im, 1);
WIDTH = size(im, 2);
D = HEIGHT * WIDTH;
C = size(im, 3);

im = reshape(im, D, C);
