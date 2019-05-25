function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y);                    % number of training examples
J_history = zeros(num_iters, 1);  % vector to store the cost at every iteration

    for iter = 1:num_iters

    % ====================== YOUR CODE HERE =====================


    theta = theta - alpha*(1/m) * (X' * (X * theta - y));


    % ===========================================================
    J_history(iter) = computeCostMulti(X, y, theta); % Save the cost J in every iteration    

    end

end