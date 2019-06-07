close all; clear all; clc;

% neural net implementation 

load('train_data.mat');
load('cv_data.mat');

train.X = train_data.inputs; 
train.y = train_data.UTS;
train.m = size(train.X, 1);

cv.X = cv_data.inputs; 
cv.y = cv_data.UTS;
cv.m = size(cv.X, 1);

train.svr_Mdl = fitrsvm(train.X, train.y);% , 'KernelFunction','gaussian');
%,'Standardize',true,'KFold',5,'KernelFunction','gaussian')

% without kernel function
% cv_gpr_total_loss : 2.646741e-01 
% training_gpr_loss : 4.275899e-01 

% with gaussian kernel 
% cv_gpr_total_loss : 2.932041e-01 
% training_gpr_loss : 2.956287e-02 

train.ypred = resubPredict(train.svr_Mdl);
train.err = train.ypred - train.y; 

cv.ypred = predict(train.svr_Mdl, cv.X);
cv.err = cv.ypred - cv.y;

% cv_data.cv_svr_Mdl = crossval(train.svr_Mdl, cv.X);
% cv_data.L = kfoldLoss(cv_data.cv_svr_Mdl);
% 
% fprintf('k folds loss %d', cv_data.L);

figure;
plot(cv.ypred, 'o');
hold on; 
plot(cv.y, '*');
legend('predicted' , 'actual')
title('predict and actual normalized UTS')
cv.J_total = 1/(2*cv.m) * (cv.err)' * cv.err; 
fprintf('cv_svr_total_loss : %d \n', cv.J_total);

train.J_total = 1/(2*train.m) * (train.err)' * train.err; 
fprintf('train_svr_total_loss : %d \n', train.J_total);

train.training_svr_loss = resubLoss(train.svr_Mdl);
fprintf('training_svr_loss : %d \n',train.training_svr_loss);

plotfixer;

%% reduced features 

train.reduced.inputs = reduced_features(train_data.inputs, [6,9]);
cv.reduced.X = reduced_features(cv_data.inputs, [6,9])
train.reduced.svr_Mdl = fitrsvm(train.reduced.inputs, train.y);% , 'KernelFunction','gaussian');

train.reduced.ypred = resubPredict(train.reduced.svr_Mdl);
train.reduced.err = train.reduced.ypred - train.y; 

cv.reduced.ypred = predict(train.reduced.svr_Mdl, cv.reduced.X);
cv.reduced.err = cv.reduced.ypred - cv.y;

figure;
plot(cv.reduced.ypred, 'o');
hold on; 
plot(cv.y, '*');
legend('predicted' , 'actual')
title('predict and actual normalized UTS')
cv.reduced.J_total = 1/(2*cv.m) * (cv.reduced.err)' * cv.reduced.err; 
fprintf('cv_svr_total_loss : %d \n', cv.reduced.J_total);

train.reduced.J_total = 1/(2*train.m) * (train.reduced.err)' * train.reduced.err; 
fprintf('reduced_train_svr_total_loss : %d \n', train.reduced.J_total);

train.reduced.training_svr_loss = resubLoss(train.reduced.svr_Mdl);
fprintf('reduced_training_svr_loss : %d \n',train.reduced.training_svr_loss);

plotfixer;