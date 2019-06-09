% auto-encoder neural network

close all; clear all; clc;

load('train_data.mat');
load('cv_data.mat');
load('test_data.mat');
train.X = train_data.inputs; 

train.y = train_data.UTS;
train.m = size(train.X, 1);
layer_size = 11;
train.autoenc = trainAutoencoder(train.X', layer_size);

cv.X = cv_data.inputs; 

%% 
train.X_recon = predict(train.autoenc,train.X');
mseError = mse(train.X'-train.X_recon)


train.Encoder_b = train.autoenc.EncoderBiases;
train.Encoder_W = train.autoenc.EncoderWeights;

train.Encoded_X = train.Encoder_W * train.X' + train.Encoder_b;
cv.Encoded_X = train.Encoder_W * cv.X' + train.Encoder_b;
%%

theta_init = zeros(layer_size, 1);
num_iters = 900; 
alpha =1e-3; 
lambda = 15; 

[train.theta, train.J_history] = regGradDescent(train.Encoded_X', ...
                                     train_data.UTS,...
                                     theta_init, alpha, ...
                                     lambda, num_iters);

cv.J = zeros(num_iters, 1);
for i = 1:length(train.theta)
    cv.J(i) = computeCostMulti(cv.Encoded_X', ...
                               cv_data.UTS, ...
                               train.theta(:,i));
end 
%%
test.Encoded_X = train.Encoder_W * test_data.inputs' + train.Encoder_b;

test_cost = computeCostMulti(test.Encoded_X', test_data.UTS, ...
                                train.theta(:,end));

figure;
plot(train.J_history)
hold on; 
plot(cv.J)
legend("training error", "cross-validation")
title("Using all available features")
xlabel('Iterations')
ylabel('Cost')
plotfixer;
savefig("Cost_auto_encoder.fig")
