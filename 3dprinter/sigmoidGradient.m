% GRADED FUNCTION: sigmoidGradient
function g = sigmoidGradient(z)

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================

gz = sigmoid(z); 

g = gz.*(1 - gz);

% =============================================================

end