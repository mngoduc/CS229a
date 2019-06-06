% GRADED FUNCTION: nnCostFunction
function [J grad] = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);                               % Setup some useful variables
J = 0;                                        % You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% PART 1: FEED FORWARD PROPAGATION
I = eye(num_labels);
Y = zeros(m, num_labels);

% for i = 1:m
%     y(i)
%     Y(i,:) = I(y(i),:);
% end 


% feedforward 
a1 = [ones(m, 1), X];

size1 = size(Theta1') 
size2 = size(a1)

z2 = a1 * Theta1' ;

a2 = [ones(size(z2,1),1 ), sigmoid(z2)];
z3 = a2 * Theta2'; 
a3 = sigmoid(z3); 
h_theta = a3;
K = num_labels;

y_k = eye(K);
cost = zeros(K,1);

for i=1:m
	value = -y_k(:,y(i)) .* log(h_theta(:,i)) - ... 
		(1 - y_k(:,y(i))) .* log(1 - h_theta(:,i)) ;
	cost = cost + value ;
end

J = sum(cost) / m;
regularizationTerm = sum(sum(Theta1(:,2:end) .^2)) ... 
	+ sum(sum(Theta2(:,2:end).^2)); 
J = J + regularizationTerm * lambda / (2 * m);

Delta1_2 = zeros(size(Theta2));
Delta1_1 = zeros(size(Theta1));

    for t=1:m

        a1 = [1 ; X(t,:)'];
        z2 = Theta1 *a1 ;
        a2 = sigmoid(z2);
        a2 = [1;a2];
        z3 = Theta2 *a2;
        a3 = sigmoid(z3);

        delta_3 = a3 - y_k(:,y(t));
        delta_2 = (Theta2' * delta_3 ) .* ...
            [0 ; sigmoidGradient(z2)];
        delta_2 = delta_2(2:end,1);

        Delta1_2 = Delta1_2 +   delta_3 * a2';   
        Delta1_1 = Delta1_1 +   delta_2 * a1';
    end
Theta1_grad = Delta1_1 /m ...
    + ( lambda / m ) * [zeros(size(Theta1,1),1) , Theta1(:,2:end)];

Theta2_grad = Delta1_2 /m ...
    + ( lambda / m ) * [zeros(size(Theta2,1),1) , Theta2(:,2:end)];  

grad = [Theta1_grad(:) ; Theta2_grad(:)];
end


% penalty = (lambda/2/m) * (sum(sum(Theta1(:, 2:end).^2), 2) + sum(sum(Theta2(:, 2:end).^2),2));
% h = a3;
% J = 1/m *( sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h))) ) + penalty; 
% 
% % g'(z)
% gz = sigmoid([ones(size(z2, 1), 1), z2]); 
% g_prime = gz.*(1 - gz);
% 
% % PART 2: BACK PROPAGATION
% delta3 = a3 - Y;
% delta2 = (delta3 * Theta2) .* g_prime;
% delta2 = delta2(:, 2:end);
% 
% Delta_1 = delta2'*a1;
% Delta_2 = delta3'*a2;
% 
% % PART 3: WEIGHT REGULARIZATION
% 
% Theta1_grad = 1/m * (Delta_1 + lambda * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)] );
% Theta2_grad = 1/m * (Delta_2 + lambda * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)] );
% 
% % ============================================================
% 
% grad = [Theta1_grad(:) ; Theta2_grad(:)];     % Unroll gradients
% 
% end