close all; clear all; clc

load('train_data.mat'); 
theta_init = zeros(9, 1);
num_iters = 700; 
alpha = 7.5e-3; 
lambda = 1;

% [train.theta, train.J_history] = gradientDescentMulti(train_data.normalized_inputs, ...
%                                           train_data.normalized_young,...
%                                           theta_init, alpha, num_iters);
lambda = 10; 

[train.theta, train.J_history] = regGradDescent(train_data.inputs, ...
                                     train_data.UTS,...
                                     theta_init, alpha, ...
                                     lambda, num_iters);

train.actual_J = train.J_history * train_data.sig(11) + train_data.mu(11); 

train.theta(:, end)
% learned the things, now cross-validate

load('cv_data.mat');

cv.J = zeros(num_iters, 1);
for i = 1:length(train.theta)
    cv.J(i) = computeCostMulti(cv_data.inputs, ...
                               cv_data.UTS, ...
                               train.theta(:,i));
end 

cv.actual_J = cv.J * cv_data.sig(11) + cv_data.mu(11); 
cv.y_pred = cv_data.inputs * train.theta(:, end);
%cv.pred_err = cv.y_pred - cv_data.UTS
figure; 
plot(cv.y_pred,'o');
hold on; 
plot(cv_data.UTS,'*'); 
legend('predicted', 'actual')
title('Predicted and Actual Output')

figure;
plot(train.J_history)
hold on; 
plot(cv.J)
legend('training error', 'cross-validation')
title('Using all available features')
xlabel('Iterations')
ylabel('Cost')
plotfixer;
savefig('Cost_all_features.fig')


figure; 
plot(train.actual_J)
hold on; 
plot(cv.actual_J)
legend('training error', 'cross-validation')
title('actual error')
xlabel('Iterations')
ylabel('Cost')
plotfixer;

%%
norm.lambda = 25;
norm.X = train_data.inputs; 
norm.y = train_data.UTS; 

XTX = norm.X' * norm.X;
Inxn = eye(size(XTX)); 
norm.Theta = pinv(XTX + norm.lambda * Inxn) * norm.X' * norm.y;

norm.y_pred = cv_data.inputs * norm.Theta;
norm.cv_error = cv_data.UTS - norm.y_pred; 
norm.mse = mean(sqrt(norm.cv_error.^2));
figure; 
histogram(norm.cv_error, 10)
disp(norm.mse)

figure; 
plot(norm.y_pred, 'ro')
hold on; 
plot(cv_data.UTS, 'b*')
legend('predicted', 'actual')

%% what if we removed some features, namely features #6 and #9

train_reduced.inputs = [train_data.inputs(:,1:5),...
                            train_data.inputs(:,7:8)];
theta_init_new = zeros(1,7);

[train_reduced.theta, train_reduced.J_history] = ...
                        regGradDescent(train_reduced.inputs, ...
                                       train_data.UTS,...
                                       theta_init_new, alpha, ...
                                       lambda, num_iters);
                                   
train_reduced.theta(:, end)

% learned the things, now cross-validate
cv_reduced_data.inputs = [cv_data.inputs(:,1:5),...
                               cv_data.inputs(:,7:8)];
cv_reduced.J = zeros(num_iters, 1);
for i = 1:length(train.theta)
    cv_reduced.J(i) = computeCostMulti(cv_reduced_data.inputs, ...
                               cv_data.UTS, ...
                               train_reduced.theta(:,i));
end 

figure;
plot(train_reduced.J_history)
hold on; 
plot(cv_reduced.J)
legend("training error", "cross-validation")
title("Removing Features #6 and #9")
xlabel('Iterations')
ylabel('Cost')
plotfixer;
savefig("Cost_without_6n9.fig")