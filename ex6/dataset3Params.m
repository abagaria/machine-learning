function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_best = 1;
sigma_best = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
possible_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]';
best_error = 9999; 
for C = possible_vals
    for sigma = possible_vals
        % train an svm classifier with X and y
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

        % Calculate the error rate of the cross-validation set
        ypred = svmPredict(model, Xval);
        error = mean(double(ypred ~= yval));

        % Compare the errror to the old recorded best_error
        if error < best_error
            best_error = error;
            C_best = C;
            sigma_best = sigma;
        end

    end
end

C = C_best;
sigma = sigma_best;
% =========================================================================

end
