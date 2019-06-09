% auto-encoder neural network

close all; clear all; clc;

load('train_data.mat');
<<<<<<< HEAD
load('cv_data.mat');
load('test_data.mat');
=======
>>>>>>> parent of 144aa9d... lol somehow 0.15 loss lets fukin go
train.X = train_data.inputs; 

train.y = train_data.UTS;
train.m = size(train.X, 1);

train.autoenc = trainAutoencoder(train.X', 4);


%% 
train.X_recon = predict(train.autoenc,train.X');
mseError = mse(train.X'-train.X_recon)


train.Encoder_b = train.autoenc.EncoderBiases;
train.Encoder_W = train.autoenc.EncoderWeights;

train.Encoded_X = train.Encoder_W * train.X' + train.Encoder_b
%%

<<<<<<< HEAD
theta_init = zeros(layer_size, 1);
num_iters = 900; 
alpha =1e-3; 
lambda = 15; 

[train.lin_reg.theta, train.J_history] = regGradDescent(train.Encoded_X', ...
=======
theta_init = zeros(4, 1);
num_iters = 700; 
alpha = 7.5e-3; 
lambda = 15; 

[train.theta, train.J_history] = regGradDescent(train.Encoded_X, ...
>>>>>>> parent of 144aa9d... lol somehow 0.15 loss lets fukin go
                                     train_data.UTS,...
                                     theta_init, alpha, ...
                                     lambda, num_iters);

<<<<<<< HEAD
cv.J = zeros(num_iters, 1);
for i = 1:length(train.lin_reg.theta)
    cv.J(i) = computeCostMulti(cv.Encoded_X', ...
                               cv_data.UTS, ...
                               train.lin_reg.theta(:,i));
end 



test.Encoded_X = train.Encoder_W * test_data.inputs' + train.Encoder_b;

test_cost = computeCostMulti(test.Encoded_X', test_data.UTS, ...
                                train.lin_reg.theta(:,end));

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


%%

train.svr_Mdl = fitrsvm(train.Encoded_X', train.y);

=======
%%
plot(train.J_history)
>>>>>>> parent of 144aa9d... lol somehow 0.15 loss lets fukin go
