close all; clear all; clc;

% neural net implementation 

load('train_data.mat');
load('cv_data.mat');

train.X = train_data.inputs; 
train.y = train_data.UTS;

cv.X = cv_data.inputs; 
cv.y = cv_data.UTS;
cv.m = size(cv.X, 1);

train.svr_Mdl = fitrsvm(train.X, train.y, 'KernelFunction','gaussian');
%,'Standardize',true,'KFold',5,'KernelFunction','gaussian')

cv.ypred = predict(train.svr_Mdl, cv.X);
cv.err = cv.ypred - cv.y;
cv.J_total = 1/(2*cv.m) * (cv.err)' * cv.err; 
fprintf('cv_gpr_total_loss : %d \n', cv.J_total);

train.training_gpr_loss = resubLoss(train.svr_Mdl);
fprintf('training_gpr_loss : %d \n',train.training_gpr_loss);

