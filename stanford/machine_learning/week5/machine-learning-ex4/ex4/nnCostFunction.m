function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ ones(size(X, 1), 1) X ];

% using loop
%for i = 1:m,

%    z1 = Theta1*X(i,:)';
%    a2 = sigmoid(z1);

%    a2 = [ 1; a2 ];
%    z2 = Theta2*a2;
%    a3 = sigmoid(z2);

%    yk = zeros(num_labels, 1);
%    yk(y(i)) = 1;

%    J = J + sum( -yk.*log(a3) - (1 - yk).*log(1 - a3) );


%end

%J = J/m;


% vectorized
z2 = Theta1*X';
a2 = sigmoid(z2);

a2 = [ones(1, size(a2, 2)); a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);

yk = zeros(m, num_labels);
for i = 1:m,
    yk(i, y(i)) = 1;
end

J = sum( sum( ( -yk.*log(a3)' - (1 - yk).*log(1-a3)' ) , 1 ) ) / m  ;


sigma = 0;

for j = 1:size(Theta1, 1),
    for k = 2:size(Theta1, 2),
        sigma = sigma + Theta1(j, k)^2;
    end
end

for j = 1:size(Theta2, 1),
    for k = 2:size(Theta2, 2),
        sigma = sigma + Theta2(j, k)^2;
    end
end

J = J + ( lambda/(2*m) ) * sigma;


% ============================================================================
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.


a1 = X;

d3 = ( a3 - yk' );

d2 = ( Theta2'*d3 ) .* ( a2.*(1-a2) );
d2 = d2(2:end, :);

Theta2_grad = ( d3*(a2)' ) / m;
Theta1_grad = ( d2*(a1) ) / m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = Theta1_grad + (lambda/m) * ...
                                [ zeros(size(Theta1, 1), 1) Theta1(:, 2:end) ];

Theta2_grad = Theta2_grad + (lambda/m) * ...
                                [ zeros(size(Theta2, 1), 1) Theta2(:, 2:end) ];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
