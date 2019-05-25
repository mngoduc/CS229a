function [theta, J_history] = gradientDescentMulti(X, y, theta_init, alpha, num_iters)

m = length(y);                    % number of training examples
J_history = zeros(num_iters, 1);  % vector to store the cost at every iteration

theta = zeros(length(theta_init), num_iters); 
theta(:,1) = theta_init;

    for iter = 2:num_iters+1
    prev_theta = theta(:, iter - 1);
    theta(:, iter) = prev_theta ...
                    - alpha*(1/m) * (X' * (X * prev_theta - y));

    J_history(iter-1) = computeCostMulti(X, y, theta(:, iter)); 
    % Save the cost J in every iteration    

    end

end