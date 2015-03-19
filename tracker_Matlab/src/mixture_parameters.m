function [mixparam] = mixture_parameters()
% [mixparam] = mixture_parameters()
% Returns a structure of parameters which is used to update the
% pixel-wise mixtures according to the Stauffer and Grimson algorithm. 

% Component threshold. Smaller connected components will be considered
% to be noise. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for Ant sequence (8fps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mixparam.K = 3;
mixparam.ALPHA = 0.02;
mixparam.RHO = 0.01;
mixparam.DEVIATION_SQ_THRESH = 7^2;
mixparam.INIT_VARIANCE = 3;
mixparam.INIT_MIXPROP = 0.00001;
mixparam.BACKGROUND_THRESH = 0.9;
mixparam.COMPONENT_THRESH = 10;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Parameters for StGeorge sequence (8fps)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mixparam.K = 3;
% mixparam.ALPHA = 0.01;
% mixparam.RHO = 0.01;
% mixparam.DEVIATION_SQ_THRESH = 7^2;
% mixparam.INIT_VARIANCE = 3;
% mixparam.INIT_MIXPROP = 0.00001;
% mixparam.BACKGROUND_THRESH = 0.8;
% mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for StGeorge sequence (25fps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mixparam.K = 3;
% mixparam.ALPHA = 0.005;
% mixparam.RHO = 0.005;
% mixparam.DEVIATION_SQ_THRESH = 7^2;
% mixparam.INIT_VARIANCE = 3;
% mixparam.INIT_MIXPROP = 0.00001;
% mixparam.BACKGROUND_THRESH = 0.8;
% mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Testing parameters for pets2000 sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mixparam.K = 3;
%mixparam.ALPHA = 0.005;
%mixparam.RHO = 0.005;
%mixparam.DEVIATION_SQ_THRESH = 50;
%mixparam.INIT_VARIANCE = 3;
%mixparam.INIT_MIXPROP = 0.00001;
%mixparam.BACKGROUND_THRESH = 0.8;
%mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Testing parameters for pets2001 sequences (Camera1 and Camera2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mixparam.K = 3;
% mixparam.ALPHA = 0.001;
% mixparam.RHO = 0.001;
% mixparam.DEVIATION_SQ_THRESH = 67.46;
% mixparam.INIT_VARIANCE = 3;
% mixparam.INIT_MIXPROP = 0.00001;
% mixparam.BACKGROUND_THRESH = 0.8;
% mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for webcam stream
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mixparam.K = 3;
%mixparam.ALPHA = 0.02;
%mixparam.RHO = 0.02;
%mixparam.DEVIATION_SQ_THRESH = 67.46;
%mixparam.INIT_VARIANCE = 3;
%mixparam.INIT_MIXPROP = 0.00001;
%mixparam.BACKGROUND_THRESH = 0.8;
%mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nelder-Mead optimisation parameters for video7 sequence (confidence)%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mixparam.K = 3;
%mixparam.ALPHA = 0.00539174160791;
%mixparam.RHO = 0.00731056462468;
%mixparam.DEVIATION_SQ_THRESH = 44.69758262145663;
%mixparam.INIT_VARIANCE =  7.99692827140397;
%mixparam.INIT_MIXPROP = 0.00868042059550;
%mixparam.BACKGROUND_THRESH = 0.95614778887127;
%mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nelder-Mead optimisation parameters for video7 sequence (logresponsibility)%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mixparam.K = 3;
%mixparam.ALPHA = 0.00380212364495;
%mixparam.RHO = 0.00641584186138;
%mixparam.DEVIATION_SQ_THRESH = 79.40827518857728;
%mixparam.INIT_VARIANCE = 1.40761255886641;
%mixparam.INIT_MIXPROP =  0.00165114930763;
%mixparam.BACKGROUND_THRESH = 0.99176795735288;
%mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nelder-Mead optimisation parameters for video7 sequence (stddev)%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mixparam.K = 3;
%mixparam.ALPHA = 0.004364137567;
%mixparam.RHO = 0.001369226869;
%mixparam.DEVIATION_SQ_THRESH = 416.805089554341;
%mixparam.INIT_VARIANCE =  36.203379048012;
%mixparam.INIT_MIXPROP = 0.038123717683;
%mixparam.BACKGROUND_THRESH = 0.787113166968;
%mixparam.COMPONENT_THRESH = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nelder-Mead optimisation parameters for video7 sequence (diagonal cov.)%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mixparam.K = 3;
%mixparam.ALPHA = 0.00952342457047;
%mixparam.RHO = 0.00627820963217;
%mixparam.DEVIATION_SQ_THRESH = 64.91249404373625;
%mixparam.INIT_VARIANCE = 3.02260723292532;
%mixparam.INIT_MIXPROP = 0.00275625549733;
%mixparam.BACKGROUND_THRESH = 0.98843689165904;
%mixparam.COMPONENT_THRESH = 10;
