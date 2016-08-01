function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% Prediction is made in logistic regression using
% if z > 0, then y  = 1, where z = theta'X

% all_theta is an K x (n+1) matrix, where K is the number of classes
% X is the feature/design matrix having dimensions m x (n+1)
% The product is a K x m matrix

% Given a new x, p(y = i | x; theta) = h_i(x) = sigmoid(X theta)
% x belongs to class i if argmax over i of h_i(x)
H = sigmoid(all_theta * X');

% The argmax of each row of H is the class that that training example most
% likely corresponds to
[~, p] = max(H, [], 1);
p = p';
% =========================================================================


end
