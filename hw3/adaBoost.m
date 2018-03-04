function [alpha,threshold] = adaBoost(x,y,T)
% convert x to binary data
threshold = mean(x,1);
x = x >= repmat(threshold,[size(x,1) 1]);

% make x and y to be (-1,+1)
x = x - ~x; % conversion to (-1,+1) is only for convenience
y = y - ~y; % label must be (-1,+1)

m = size(x,1); % sample size
n = size(x,2); % dimension of input
errMatrix = x~=repmat(y,[1,n]); % error of directly set a component as prediction, 1-->1 / 0-->0
errMatrixInv = ~errMatrix; % error of invert a component as prediction, 1-->0 / 0-->1

alpha = zeros(n,2); % weight, column vector
D = 1/m * ones(1,m); % initialize distribution, a row vector

nowPredict = zeros(m,1); % ht(xi)
for t=1:1:T,
    [min1,min1_index] = min(D*errMatrix);
    [min2,min2_index] = min(D*errMatrixInv);
    if min1 <= min2,
        minErr = min1;
        nowIndex = min1_index;
        invOrNot = 1; % not invert
    else
        minErr = min2;
        nowIndex = min2_index;
        invOrNot = 2; % invert
    end
    
    alphaIncre = 0.5*log((1-minErr)/minErr);
    alpha(nowIndex,invOrNot) = alpha(nowIndex,invOrNot) + 0.5*log((1-minErr)/minErr); % update weight
    Z = 2*sqrt(minErr*(1-minErr)); % update normalization factor
    if invOrNot == 1, % predict over x with non-invert hypothesis
        nowPredict = x(:,nowIndex); % +1-->+1, -1-->-1
    elseif invOrNot == 2, % predict over x with invert hypothesis
        nowPredict = -1*x(:,nowIndex); % +1-->-1, -1-->+1
    end
    D = D.*exp(-1*alphaIncre*y.'.*nowPredict.') / Z; % update distribution  
end

end