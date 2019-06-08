close all; clear all; clc;

% neural net implementation 

load('train_data.mat');
train.X = train_data.inputs; 

train.y = train_data.UTS;
train.m = size(train.X, 1);

train.net = newgrnn(train.X, train.y);

load('cv_data.mat');
cv.X = cv_data.inputs;
cv.y_pred = sim(train.net,cv.X); 
cv.y_actual = cv_data.UTS; 



% 
%                           % Setup the parameters you will use for this exercise
% input_layer_size  = 9*30; % 20x20 Input Images of Digits
% hidden_layer_size = 30;   % 25 hidden units
% num_labels = 1;          
% 
% initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
% initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% 
% % Unroll parameters
% initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
% 
% options = optimset('MaxIter', 50);
% lambda = 1;                          %  You should also try different values of lambda
%                                      % Create "short hand" for the cost function to be minimized
% costFunction = @(p) nnCostFunction(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, X, y, lambda);
% 
% % Now, costFunction is a function that takes in only one argument (the
% % neural network parameters)
% [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% 
% % Obtain Theta1 and Theta2 back from nn_params
% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                  hidden_layer_size, (input_layer_size + 1));
% 
% Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                  num_labels, (hidden_layer_size + 1));
% 
% pred = predict(Theta1, Theta2, X);
% Accuracy =  mean(double(pred == y)) * 100 