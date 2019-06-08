clear all; close all; clc; 

% normalized data
load("all_data.mat");

y = all_data(:, 11);
x = all_data(:, 1:9);

% titles = ["layer_height"; "wall_thickness"; "infill_density"; ...
%           "infill_pattern"; "nozzle_temperature"; "bed_temperature";...
%           "print_speed"; "material"; "fan_speed"; "roughness",...
%           "tension_strength"; "elongation"];

for i = 1:9
    figure;
    plot(x(:,i),y.^3, "o");
    curr_title = strcat("feature number", int2str(i));
    title(curr_title);

end 