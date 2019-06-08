% GRADED FUNCTION: selectThreshold
function [bestEpsilon bestF1] = selectThreshold(yval, pval)

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================

    % true pos : both detector and ground says it's correct
    tp = sum((yval==1) & (pval<epsilon));
    
    % false neg: detector says neg; ground says pos
    fn = sum((yval==1) & (pval>=epsilon));
    
    % false pos: detector says pos; ground says neg
    fp = sum((yval==0) & (pval<epsilon));
    
    
    % precision
    prec = tp / (tp + fp); 
    % recall
    rec = tp / (tp + fn); 
    
    F1 = 2 * prec * rec / (prec + rec);
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end
end