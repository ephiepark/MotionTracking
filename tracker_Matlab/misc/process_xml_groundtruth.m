function [trajectories] = process_xml_groundtruth(path)
% Given a directory path containing several XML files conforming to the 
% PETS 2001 XML trajectory schema, produce an array of structures, each
% of which contains among others the field pos to indicator object
% positions. This structure can be fed into plot_trajectories.m to 
% visualise the trajectories on an image background. 

% Fix up the path if slash was ommitted

if(path(end) ~= '/')
   path = [path '/'];
end

fprintf(1, 'Loading files from %s\n', path);

files = dir([path '*.xml']);
files = sort({files.name});

% Begin with empty trajectories structure
trajectories = struct([]);
next_trajectory = 1;

% For each xml tree structure, extract the relevant information 
% and record it in a trajectories structure which is can be plotted 
% more easily. 

for obj = 1:size(files, 2)
  fprintf(1, 'Reading file %d: %s\n', obj, files{obj});

  tree = parseXML([pwd '/' path files{obj}]);

  if(~strcmp(tree.Attributes(3).Name, 'FRAME_CREATED'))
    warning('FATAL: expected to find a start frame number here');
    return;
  end
  
  frame_created = str2num(tree.Attributes(3).Value);
  
  if(~strcmp(tree.Attributes(4).Name, 'FRAME_DESTROYED'))
    warning('FATAL: expected to find a start frame number here');
    return;
  end
  
  frame_destroyed = str2num(tree.Attributes(4).Value);
  
  if(~strcmp(tree.Attributes(5).Name, 'ID'))
    warning('FATAL: expected to find an object id here');
    return;
  end

  id = str2num(tree.Attributes(5).Value);
    
  for child = 1:size(tree.Children, 2)

    if(~strcmp(tree.Children(child).Attributes.Name, 'TIME'))
      warning('FATAL: expected to find a time element here');
      return;
    end

    current_time = str2num(tree.Children(child).Attributes.Value);

    if(~strcmp(tree.Children(child).Children(1).Children(2).Name, 'CENTROID'))
      warning('FATAL: expected to find a centroid element here');
      return;
    end
    
    centroid = tree.Children(child).Children(1).Children(2);
    
    x = centroid.Attributes(1);
    y = centroid.Attributes(2);
  
    if(~strcmp(x.Name, 'CENTER_X') || ~strcmp(y.Name, 'CENTER_Y'))
      warning('FATAL: expected to find centroid data here');
      return;
    end
    
    % Ids are indexed from zero
    trajectories(id + 1).label = id;
    trajectories(id + 1).frame_created = frame_created;
    trajectories(id + 1).frame_destroyed = frame_destroyed;

    trajectories(id + 1).pos(:, current_time) = [str2num(x.Value);
						 str2num(y.Value)];
  end

  % So that we can use this data structure with plot_trajectories.m we say
  % that this trajectory was updated at each frame. 

  trajectories(id + 1).ntimes_updated = frame_destroyed - frame_created + 1;
  trajectories(id + 1).traj_length = frame_destroyed - frame_created + 1;
  
  % And pretend the last obervation was also a match. Otherwise we will
  % not plot the whole trajectory using plot_trajectories.m 
  
  trajectories(id + 1).last_updated = frame_destroyed;
end
