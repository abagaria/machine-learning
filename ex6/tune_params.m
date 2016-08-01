function [C_best, sigma_best] = tune_params(X, y, Xval, yval)
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
end