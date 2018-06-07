function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
% Using for
%    s0 = 0;
%    s1 = 0;
%    for i = 1:m
%        s0 = s0 + theta(1) + theta(2)*X(i, 2) - y(i);
%        s1 = s1 + (theta(1) + theta(2)*X(i, 2) - y(i))*X(i,2);
%    end

%    theta(1,1) = theta(1,1) -alpha*s0/m;
%    theta(2,1) = theta(2,1) -alpha*s1/m;

% Using vectorization

    s = zeros(size(theta, 1), 1);
    for i = 1:m
        s = s + (theta'*X(i,:)'-y(i))*X(i, :)';
    end

    theta = theta - alpha/m*s;


    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
