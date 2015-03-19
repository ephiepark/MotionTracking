% Main script for generating a set of synthetic sequences. These can be
% used for developing and testing new code. The 

global HEIGHT WIDTH D T C;

HEIGHT = 120;
WIDTH = 160;
D = HEIGHT * WIDTH;
C = 3;
T = 250;

% Generate some occlusions for two objects. Object appearances and 
% movements are described by an array of structures. 

% Small grey object moves right quickly
objects(1).sidelength = 10;
objects(1).looming = [0 0];
objects(1).position = [-10; floor(HEIGHT/2) - 12];

% Object dynamics. Each column vector is an entry. Different dynamics
% are active at different times (see dynamics_start). 
objects(1).velocities = [15; 0];
objects(1).accellerations = [0; 0];

% After what frame do the velocity, accelleration etc. dynamics start taking
% effect. This should be a list in increasing order, only one dynamic will
% be active at any time.
objects(1).dynamics_start = [150]; 

objects(1).meancolour = [100; 100; 100];
objects(1).noiseamp = 1; % How much to amplify object colour gaussian noise

% Small grey object moves left slowly
objects(2).sidelength = 10;
objects(2).looming = [0 0];
objects(2).position = [WIDTH + 10; floor(HEIGHT/2)];
objects(2).velocities = [-6; 0];
objects(2).accellerations = [0; 0];
objects(2).dynamics_start = [150];
objects(2).meancolour = [100; 100; 100];
objects(2).noiseamp = 1; % How much to amplify object colour gaussian noise

% Generate sequence 1 for the object description
data_1 = synthetic_sequence(HEIGHT, WIDTH, T, objects);

% Large grey object moving right at first and slowing down. 
% When it stops, stay stationary. 
objects(1).sidelength = 20;
objects(1).looming = [0 0];
objects(1).position = [-50; floor(HEIGHT/2) - 13];
objects(1).velocities = [5 5 0; 0 0 0];
objects(1).accellerations = [0 -0.5 0; 0 0 0];
objects(1).dynamics_start = [200 211 218]; 
objects(1).meancolour = [100; 100; 100];
objects(1).noiseamp = 2; % How much to amplify object colour gaussian noise

% Small grey object (i.e. a "ball") moving left toward first
% object. When it reaches the objectand is occluded, it stops, pauses for
% a while, and then moves quickly up (it was "kicked"). 

objects(2).sidelength = 5;
objects(2).looming = [0 0];
objects(2).position = [WIDTH + 20; floor(HEIGHT/2)];
objects(2).velocities = [-8 6; 0 10];
objects(2).accellerations = [0 0; 0 0];
objects(2).dynamics_start = [200 218]; 
objects(2).meancolour = [100; 100; 100];
objects(2).noiseamp = 2; % How much to amplify object colour gaussian nois

data_2 = synthetic_sequence(HEIGHT, WIDTH, T, objects);

objects(1).sidelength = 6;
objects(1).looming = [0 0 0];
objects(1).position = [-36; floor(HEIGHT/2) - 4];
objects(1).velocities = [5 5 0; 0 0 0];
objects(1).accellerations = [0 -0.5 0; 0 0 0];
objects(1).dynamics_start = [200 211 217]; 
objects(1).meancolour = [100; 100; 100];
objects(1).noiseamp = 2; % How much to amplify object colour gaussian noise

objects(2).sidelength = 15;
objects(2).looming = [-0.5 -0.5];
objects(2).position = [WIDTH + 26; HEIGHT - 28];
objects(2).velocities = [-8 2; -2 -2];
objects(2).accellerations = [0 0; 0 0];
objects(2).dynamics_start = [200 217]; 
objects(2).meancolour = [100; 100; 100];
objects(2).noiseamp = 2; % How much to amplify object colour gaussian nois

data_3 = synthetic_sequence(HEIGHT, WIDTH, T, objects);

% Large grey object moving right, slightly smaller blue object moving left. 
% The grey object occludes the blue object. 

objects(1).sidelength = 25;
objects(1).looming = [0];
objects(1).position = [WIDTH + 10; floor(HEIGHT/2)];
objects(1).velocities = [-3; 0];
objects(1).accellerations = [0; 0];
objects(1).dynamics_start = [150]; 
objects(1).meancolour = [40; 40; 160];
objects(1).noiseamp = 2; % How much to amplify object colour gaussian nois
objects(1).holesidelength = 0; % Sometimes the object has a hole in the middle
objects(1).holetimes = [0, 0]; % During which frames does the object
                                   % have a hole in the middle

objects(2).sidelength = 21;
objects(2).looming = [0];
objects(2).position = [-40; floor(HEIGHT/2) - 10];
objects(2).velocities = [3; 0];
objects(2).accellerations = [0; 0];
objects(2).dynamics_start = [155]; 
objects(2).meancolour = [40; 160; 40];
objects(2).noiseamp = 2; % How much to amplify object colour gaussian noise
objects(2).holesidelength = 10; % Sometimes the object has a hole in the middle
objects(2).holetimes = [186, 190]; % During which frames does the object
                                   % have a hole in the middle

data_4 = synthetic_sequence(HEIGHT, WIDTH, T, objects);

% To integrate with run_tracker, let data_4 be data
data = data_4;
