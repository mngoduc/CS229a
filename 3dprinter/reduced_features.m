function reduced_input = reduced_features(orig_data, exclude_vec)
    m = size(orig_data, 1); % number of rows
    n = size(orig_data, 2); % number of columns 
    reduced_input = zeros(m, n - length(exclude_vec));
    curr_col = 1; 
    for i = 1:n
        if(~ismember(i,exclude_vec))
            reduced_input(:,curr_col) = orig_data(:, i);
            curr_col = curr_col + 1;
        end
    end 
end 