close all ; clear all; clc; 

load('train_data.mat'); 
load('cv_data.mat'); 
GPR_kernel = 'squaredexponential'; %'ardsquaredexponential';

% 'squaredexponential': cross validation loss: 3.981976e-01 ;loss from introducing new data: 3.487219e-01 
gprMdl = fitrgp([train_data.inputs; cv_data.inputs],...
        [train_data.UTS; cv_data.UTS],...
        'KernelFunction', GPR_kernel ,...
        'FitMethod','sr','PredictMethod','fic','Standardize',1);
ypred = resubPredict(gprMdl);
train_data.training_gpr_loss = resubLoss(gprMdl);
disp(train_data.training_gpr_loss)

cv_data.cvMdl = crossval(gprMdl);
cv_data.L = kfoldLoss(cv_data.cvMdl);

fprintf('cross validation loss: %d \n', cv_data.L);


gprMdl1 = fitrgp(train_data.inputs,...
        train_data.UTS,... %'KernelFunction','ardsquaredexponential',...
        'KernelFunction',GPR_kernel,...
        'FitMethod','sr','PredictMethod','fic','Standardize',1);
cv_data.pred = predict(gprMdl1,cv_data.inputs);
cv_data.error = cv_data.UTS - cv_data.pred;
cv_data.mse = sqrt(sum(cv_data.error.^2))/10;
fprintf('loss from introducing new data: %d \n', cv_data.mse)
%% 
% checking direction and general error for prediction  %
%histogram(cv_data.pred - train_data.normalized_young) 
