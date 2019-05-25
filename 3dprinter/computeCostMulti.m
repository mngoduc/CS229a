function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples
J = 0;         % compute the cost and set it to J

% ====================== YOUR CODE HERE ======================
err = X * theta - y;
J = 1/(2*m) * (err)' * err; 
% ============================================================
end