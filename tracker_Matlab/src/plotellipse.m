function [] = plotellipse(x,y,rx,ry,colour,width);
% [] = plotellipse(x,y,rx,ry,colour);
% This code was borrowed from Prof. Geoff Hinton's website at 
% http://www.cs.toronto.edu/~hinton/csc321/matlab/plotellipse.m

theta = 0 :.05: 2*pi;
plot(x+rx*sin(theta),y+ry*cos(theta), colour, 'Linewidth', width);
