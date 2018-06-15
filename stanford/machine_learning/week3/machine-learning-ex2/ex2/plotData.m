function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Importing data
%data = load('ex2data1.txt');
%X = data(:, [1,2]);
%y = data(:, 3);

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% Find indices of positive and negative examples
pos = find(y == 1);
neg = find(y == 0);

% Plotting
plot(X(pos, 1), X(pos, 2) , 'k+', 'linewidth', 3, 'markersize', 10);
plot(X(neg, 1), X(neg, 2), 'ko', 'markerfacecolor', 'y', 'markersize', 10);

xlabel('Exam 1 score', 'fontsize', 15);
ylabel('Exam 2 score', 'fontsize', 15);

legend('Admitted', 'Not admitted');


% =========================================================================



hold off;

end
