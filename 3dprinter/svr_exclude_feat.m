function [cv_abs_err, train_abs_err] = svr_exclude_feat(train, cv, exclude)
    
    train_inputs = train(:, 1:9); 
    cv_inputs = cv(:, 1:9);
    
    train_m = size(train_inputs, 1);
    cv_m = size(cv_inputs, 1);
    
    train_y = train(:, 10);
    cv_y = cv(:, 10); 
    
    train_reduced_X = reduced_features(train_inputs, exclude);
    cv_reduced_X = reduced_features(cv_inputs, exclude);
    
    train_svr_Mdl = fitrsvm(train_reduced_X, train_y);% , 'KernelFunction','gaussian');

    train_reduced_ypred = resubPredict(train_svr_Mdl);
    train_reduced_err = train_reduced_ypred - train_y; 

    cv_reduced_ypred = predict(train_svr_Mdl, cv_reduced_X);
    cv_reduced_err = cv_reduced_ypred - cv_y;

    figure;
    plot(cv_reduced_ypred, 'o');
    hold on; 
    plot(cv_y, '*');
    legend('predicted' , 'actual')
    plot(cv_reduced_err);
    title_name = strcat('UTS with removed features', mat2str(exclude)); 
    title(title_name);

    cv_reduced_J_total = 1/(2*cv_m) * (cv_reduced_err)' * cv_reduced_err; 
    fprintf('cv_svr_total_loss : %d \n', cv_reduced_J_total);

    train_reduced_J_total = 1/(2*train_m) * (train_reduced_err)' ...
                                    * train_reduced_err; 
    fprintf('reduced_train_svr_total_loss : %d \n', train_reduced_J_total);

    train_reduced_training_svr_loss = resubLoss(train_svr_Mdl);
    fprintf('reduced_training_svr_loss : %d \n',train_reduced_training_svr_loss);

    plotfixer;
    
    cv_abs_err = abs(cv_y) - abs(cv_reduced_ypred);
    train_abs_err = abs(train_y) - abs(train_reduced_ypred);
end 