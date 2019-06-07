clear all; close all; clc; 

load('train_data.mat');

train.X = train_data.inputs; 
train.Y = train_data.UTS;

train.Mdl = fitcknn(train.X,train.Y) ;