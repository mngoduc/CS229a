close all; clear all; clc;

load('data.mat');
data = table2array(data);
% random number generator to separate into train - cv - test 
idx = randperm(50,50);

train_idx = idx(1:30); 
cv_idx = idx(31:40); 
test_idx = idx(41:50); 

train = zeros(30, 12);
cv = zeros(10, 12);
test = zeros(10, 12); 

% 1  = grid, 2 = honeycomb
% 1 = abs, 2 = pla

for i = 1:length(train_idx)
    idx = train_idx(i);
    train(i,:) = data(idx,:);
end 

save('train.mat','train')

for i = 1:length(cv_idx)
    idx = cv_idx(i);
    cv(i,:) = data(idx,:);
end

save('cv.mat','cv')

for i = 1:length(test_idx)
    idx = test_idx(i);
    test(i,:) = data(idx,:);
end 
save('test.mat','test')

% column labels 
% inputs
layer_height = 1; 
wall_thickness = 2; 
infill_density = 3; 
infill_pattern = 4; 
nozzle_temperature = 5; 
bed_temperature = 6; 
print_speed = 7;
material = 8; 
fan_speed = 9; 

% outputs
roughness = 10; 
tension_strength = 11; 
elongation = 12;

% training data columns separated 
train_data.layer_height = train(:,1); 
train_data.wall_thickness = train(:,2); 
train_data.infill_density = train(:,3); 
train_data.infill_pattern = train(:,4); 
train_data.nozzle_temperature = train(:,5); 
train_data.bed_temperature = train(:,6); 
train_data.print_speed = train(:,7);
train_data.material = train(:,8); 
train_data.fan_speed = train(:,9); 

train_data.inputs = train(:,1:9);
train_data.normalized_inputs = featureNormalize(train_data.inputs);
train_data.tension_strength = train(:,11); 

save('train_data.mat', 'train_data')

% cross-validation data columns separated 
cv_data.layer_height = cv(:,1); 
cv_data.wall_thickness = cv(:,2); 
cv_data.infill_density = cv(:,3); 
cv_data.infill_pattern = cv(:,4); 
cv_data.nozzle_temperature = cv(:,5); 
cv_data.bed_temperature = cv(:,6); 
cv_data.print_speed = cv(:,7);
cv_data.material = cv(:,8); 
cv_data.fan_speed = cv(:,9); 

cv_data.inputs = cv(:,1:9);
cv_data.normalized_inputs = featureNormalize(cv_data.inputs);

cv_data.tension_strength = cv(:,11); 

save('cv_data.mat', 'cv_data')
