close all; clear all; clc;
% Features used:
% ave_area, ave_aspect_ratio, ave_dihedral_angle,
% num_faces, num_vertices, num_connected_components

load('import_data.mat')

% Normalize
ave_area_norm = (ave_area - mean(ave_area)) / std(ave_area);
num_faces_norm = (num_faces - mean(num_faces)) / std(num_faces);
num_connected_components_norm = (num_connected_components - mean(num_connected_components)) / std(num_connected_components);

X = [ave_area_norm, num_faces_norm, num_connected_components_norm];

plot3(X(:,1),X(:,2),X(:,3), '.')

figure
histogram(X(:,3))

% Initialize clusters
num_clusters = 4;
num_features = 3;
iterations = 0;

centroids = rand(num_clusters, num_features);
idx_old = zeros(size(X,1),1);

while(true)
    % Cluster Assignment
    distance = zeros(size(X,1), num_clusters);

    for i = 1:size(X,1)
        for j = 1:num_clusters
            distance(i,j) = sum((X(i,:) - centroids(j,:)).^2);
        end
    end

    [minValues, idx_new] = min(distance,[],2);

    % Move Centroids
    for i = 1:num_clusters
        sel = find(idx_new == i);
        centroids(i,:) = mean(X(sel,:));
    end

    if idx_new == idx_old
        break
        
    else
        idx_old = idx_new;
        iterations = iterations + 1;
    end
end
