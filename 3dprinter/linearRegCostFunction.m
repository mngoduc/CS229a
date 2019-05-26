% GRADED FUNCTION: linearRegCostFunction
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y);                         % number of training examples
                                       % Return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
h = X * theta; 
resid = h - y; 

% regularization skips the bias term
theta_j = theta;
theta_j(1) = 0;

J = 1/(2*m) * (sum( resid.^2)  + lambda * sum(theta_j.^2));

% regularization term in gradient
grad_reg = 1/m * lambda * theta_j;
grad_reg(1) = 0; 

grad = 1/m * ( X' * resid) + grad_reg; 
% ============================================================
grad = grad(:);
end
