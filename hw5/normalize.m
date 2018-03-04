%%% min max normalization
function [X_n,minMax] = normalize(X,Xminmax)
if ~exist('Xminmax', 'var'),
    minMax = [min(X); max(X)];
    Xmin = repmat(minMax(1,:),[size(X,1) 1]);
    Xmax = repmat(minMax(2,:),[size(X,1) 1]);
else
    Xmin = repmat(Xminmax(1,:),[size(X,1) 1]);
    Xmax = repmat(Xminmax(2,:),[size(X,1) 1]);
end

X_n = (X-Xmin) ./ (Xmax-Xmin);

end
%%% standarlization
% function [X_n,para] = normalize(X,Xpara)
% if ~exist('Xpara', 'var'),
%     para = [mean(X); var(X)];
%     Xmean = repmat(para(1,:),[size(X,1) 1]);
%     Xvar = repmat(para(2,:),[size(X,1) 1]);
% else
%     Xmean = repmat(Xpara(1,:),[size(X,1) 1]);
%     Xvar = repmat(Xpara(2,:),[size(X,1) 1]);
% end
% 
% X_n = (X - Xmean)./(2*sqrt(Xvar));
% end