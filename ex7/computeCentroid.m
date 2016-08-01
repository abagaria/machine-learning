function centroid = computeCentroid(X, id, idx)
%COMPUTECENTROID given a single id, find all the n-d points in that are
%assigned to that id. Then, find the centroid of those points.
    relavant_points = [];
    for i = 1 : size(idx, 1)
        if idx(i) == id
            relavant_points = [relavant_points; X(i,:)];
        end
    end
    
    % sum all the rows of the relavant_points matrix
    sumPoints = 0;
    for row = 1:size(relavant_points, 1)
        sumPoints = sumPoints + relavant_points(row, :);
    end
    centroid = sumPoints / size(relavant_points, 1);
end
            