% auto-encoder neural network

close all; clear all; clc;

load('train_data.mat');
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

theta_init = zeros(4, 1);
num_iters = 700; 
alpha = 7.5e-3; 
lambda = 15; 

[train.theta, train.J_history] = regGradDescent(train.Encoded_X, ...
                                     train_data.UTS,...
                                     theta_init, alpha, ...
                                     lambda, num_iters);

%%
plot(train.J_history)
