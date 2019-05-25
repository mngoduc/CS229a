function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
err = X * theta - y;
J = 1/(2*m) * (err)' * err; 
% ============================================================
end