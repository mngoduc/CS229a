close all; clear all; clc; 
load('train_data.mat');

train.X = train_data.inputs;
train.y = train_data.UTS;
[train.mu, train.sigma2] = estimateGaussian(train.X);                  
%  Estimate mu and sigma2

train.p = multivariateGaussian(train.X, train.mu, train.sigma2);     
%  Returns the density of the multivariate normal 
%  at each data point (row) of X
[train.epsilon, train.F1] = selectThreshold(train.y, train.p);
fprintf('Best epsilon found using cross-validation: %e\n', train.epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', train.F1);

load('cv_data.mat');
cv.X = cv_data.inputs;
cv.y = cv_data.UTS;
[cv.mu, cv.sigma2] = estimateGaussian(cv.X);                  
%  Estimate mu and sigma2

cv.p = multivariateGaussian(cv.X, cv.mu, cv.sigma2);                 

[cv.epsilon, cv.F1] = selectThreshold(cv.y, cv.p);
fprintf('Best epsilon found using cross-validation: %e\n', cv.epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', cv.F1);

outliers = find(cv.p < cv.epsilon);                %  Find the outliers in the training set and plot the
