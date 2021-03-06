close all; clear all; clc;

load ('all_data.mat');

normal_all = featureNormalize(all_data); 

dupl_data =  featureNormalize(all_data);
random_noise = -0.02 + 0.04 * rand(size(dupl_data));

%%


idx = randperm(50,50);

train_idx = idx(1:30); 
cv_idx = idx(31:40); 
test_idx = idx(41:50); 

train = zeros(30, 12);
cv = zeros(10, 12);
test = zeros(10, 12); 



% 1  = grid, 2 = honeycomb
% 1 = abs, 2 = pla
%% Data has been normalized, now separate into train-validate-stress


for i = 1:length(train_idx)
    idx = train_idx(i);
    train(i,:) = all_data(idx,:);
end 

save('train.mat','train')

for i = 1:length(cv_idx)
    idx = cv_idx(i);
    cv(i,:) = all_data(idx,:);
end

save('cv.mat','cv')

for i = 1:length(test_idx)
    idx = test_idx(i);
    test(i,:) = all_data(idx,:);
end 

save('test.mat','test')

%% Separate the things into columns for later 
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
train_data.UTS = train(:,11); 

train_data.mu = mu; 
train_data.sig = sigma; 

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
cv_data.UTS = cv(:,11); 

cv_data.mu = mu; 
cv_data.sig = sigma;
save('cv_data.mat', 'cv_data')


test_data.inputs = test(:,1:9);
test_data.UTS = test(:,11); 
test_data.mu = mu; 
test_data.sig = sigma; 
save('test_data.mat', 'test_data')