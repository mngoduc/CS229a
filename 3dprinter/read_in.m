close all; clear all; clc;

data = load('data.mat');
% eh idk 
data = data.data; 
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
