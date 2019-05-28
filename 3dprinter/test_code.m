close all; clear all; clc

load('train_data.mat'); 
theta_init = zeros(9, 1);
num_iters = 700; 
alpha = 7.5e-3; 
lambda = 10; 

% [train.theta, train.J_history] = gradientDescentMulti(train_data.normalized_inputs, ...
%                                           train_data.normalized_young,...
%                                           theta_init, alpha, num_iters);

[train.theta, train.J_history] = regGradDescent(train_data.normalized_inputs, ...
                                     train_data.normalized_young,...
                                     theta_init, alpha, ...
                                     lambda, num_iters);


% learned the things, now cross-validate

load('cv_data.mat');

cv.J = zeros(num_iters, 1);
for i = 1:length(train.theta)
    cv.J(i) = computeCostMulti(cv_data.normalized_inputs, ...
                               cv_data.normalized_young, ...
                               train.theta(:,i));
end 

plot(train.J_history)
hold on; 
plot(cv.J)
legend("training error", "cross-validation")
plotfixer;

%%
norm.lambda = 25;
norm.X = train_data.normalized_inputs; 
norm.y = train_data.normalized_young; 

XTX = norm.X' * norm.X;
Inxn = eye(size(XTX)); 
norm.Theta = pinv(XTX + norm.lambda * Inxn) * norm.X' * norm.y;

norm.y_pred = cv_data.normalized_inputs * norm.Theta;
norm.cv_error = cv_data.normalized_young - norm.y_pred; 
norm.mse = mean(sqrt(norm.cv_error.^2));
figure; 
histogram(norm.cv_error, 10)
disp(norm.mse)

figure; 
plot(norm.y_pred, 'ro')
hold on; 
plot(cv_data.normalized_young, 'b*')
legend('predicted', 'actual')
