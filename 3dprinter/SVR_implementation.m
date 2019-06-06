close all; clear all; clc;

% neural net implementation 

load('train_data.mat');
load('cv_data.mat');

train.X = train_data.inputs; 
train.y = train_data.UTS;

cv.X = cv_data.inputs; 
cv.y = cv_data.UTS;
cv.m = size(cv.X, 1);

train.svr_Mdl = fitrsvm(train.X, train.y)% , 'KernelFunction','gaussian');
%,'Standardize',true,'KFold',5,'KernelFunction','gaussian')

% without kernel function
% cv_gpr_total_loss : 2.646741e-01 
% training_gpr_loss : 4.275899e-01 

% with gaussian kernel 
% cv_gpr_total_loss : 2.932041e-01 
% training_gpr_loss : 2.956287e-02 

cv.ypred = predict(train.svr_Mdl, cv.X);
cv.err = cv.ypred - cv.y;

figure;
plot(cv.ypred);
hold on; 
plot(cv.y);
legend('predicted' , 'actual')

cv.J_total = 1/(2*cv.m) * (cv.err)' * cv.err; 
fprintf('cv_gpr_total_loss : %d \n', cv.J_total);

train.training_gpr_loss = resubLoss(train.svr_Mdl);
fprintf('training_gpr_loss : %d \n',train.training_gpr_loss);

