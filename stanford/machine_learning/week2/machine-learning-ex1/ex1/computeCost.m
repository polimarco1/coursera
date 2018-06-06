function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Loading and setting up the data
data = load('ex1data1.txt');
y = data(:, 2);

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% Only to test the script. It should be removed to submit
X = [ones(m, 1), data(:, 1)];   % Add a column of ones to x
theta = zeros(2, 1);            % Initialize fitting parameters
iterations = 1500;
alpha = 0.01;

sigma = 0;
for i = 1:m,
    sigma = sigma + (theta(1, 1)*X(i, 1) + theta(2, 1)*X(i, 2) - y(i))^2;
end

J = (1/(2*m))*sigma;






% =========================================================================

end
