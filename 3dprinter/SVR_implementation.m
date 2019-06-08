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
train.abs_err = abs(train.y) - abs(train.ypred); 

cv.ypred = predict(train.svr_Mdl, cv.X);
cv.err = cv.ypred - cv.y;
cv.abs_err = abs(cv.y) - abs(cv.ypred);
% cv_data.cv_svr_Mdl = crossval(train.svr_Mdl, cv.X);
% cv_data.L = kfoldLoss(cv_data.cv_svr_Mdl);

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

%% reduced features 6,7,9
fprintf("reduced features 6,7,9 \n")
[cv.reduced679.err, train.reduced679.err] = svr_exclude_feat(...
                            [train.X, train.y], [cv.X, cv.y], [6,7,9]);


%% reduced features 4,6,7,9
fprintf("reduced features 4,6,7,9 \n")
[cv.reduced4679.err, train.reduced4679.err] = svr_exclude_feat(...
                            [train.X, train.y], [cv.X, cv.y], [4,6,7,9]);

%% comparing removing features

figure;
hold on; 
plot(cv.abs_err )
plot(cv.reduced679.err)
plot(cv.reduced4679.err)
legend('all features', 'removed 6,7,9', 'removed 4,6,7,9');
title('|y_{actual}| - |y_{pred}|')
plotfixer;

figure;
hold on; 
plot(train.abs_err)
plot(train.reduced679.err)
plot(train.reduced4679.err)
title('|y_{actual}| - |y_{pred}|')
legend('all features', 'removed 6,7,9', 'removed 4,6,7,9');
plotfixer;

%% fitsvr with k-folds 
train.svr_Mdl = fitrsvm(train.X, train.y,'KFold',15);%,...
                        %'KernelFunction','gaussian')
mseLin = kfoldLoss(train.svr_Mdl)
fprintf('k folds loss %d', mseLin);
