close all ; clear all; clc; 

load('train_data.mat'); 
load('cv_data.mat'); 

gprMdl = fitrgp(train_data.normalized_inputs,train_data.normalized_young,...
        'KernelFunction','ardsquaredexponential',...
        'FitMethod','sr','PredictMethod','fic','Standardize',1);
ypred = resubPredict(gprMdl);

train_data.training_gpr_loss = resubLoss(gprMdl);

cvMdl = crossval(gprMdl)
cv_data.L = kfoldLoss(cvMdl)
%%
disp(train_data.training_gpr_loss)

%% 
% checking direction and general error for prediction  
histogram(ypred - train_data.normalized_young) 
