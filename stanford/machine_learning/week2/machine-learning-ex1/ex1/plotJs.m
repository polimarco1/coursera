%% Clear and Close Figures
clear ; close all; clc


%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
num_iters = 50;
alpha1 = 0.01;
alpha2 = 0.1;
alpha3 = 1.1;

% Init Theta and Run Gradient Descent
%theta = zeros(3, 1);
%[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

theta1 = zeros(3, 1);
[theta1, J1] = gradientDescentMulti(X, y, theta1, alpha1, num_iters);
theta2 = zeros(3, 1);
[theta2, J2] = gradientDescentMulti(X, y, theta2, alpha2, num_iters);
theta3 = zeros(3, 1);
[theta3, J3] = gradientDescentMulti(X, y, theta3, alpha3, num_iters);
% Plot the convergence graph
figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
plot(1:numel(J1), J1, '-b', 'LineWidth', 2);
hold on;
plot(1:numel(J2), J2, '-r', 'LineWidth', 2);
plot(1:numel(J3), J3, '-k', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

