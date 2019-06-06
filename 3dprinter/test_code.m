close all; clear all; clc

load('train_data.mat'); 
theta_init = zeros(9, 1);
num_iters = 200; 
alpha = 7.5e-6; 

[train.theta, train.J_history] = gradientDescentMulti(train_data.inputs, ...
                                          train_data.tension_strength,...
                                          theta_init, alpha, num_iters);

% learned the things, now cross-validate

load('cv_data.mat');

cv.J = zeros(num_iters, 1);
for i = 1:length(train.theta)
    cv.J(i) = computeCostMulti(cv_data.inputs, cv_data.tension_strength, ...
                        train.theta(:,i));
end 

plot(train.J_history)
hold on; 
plot(cv.J)
legend('training error', 'cross-validation')
plotfixer;
