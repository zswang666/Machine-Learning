%%% min max normalization
function X_n = denormalize(X,Xminmax)
Xmin = repmat(Xminmax(1,:),[size(X,1) 1]);
Xmax = repmat(Xminmax(2,:),[size(X,1) 1]);

X_n = X.*(Xmax-Xmin) + Xmin;

end

%%% standardlization
% function X_n = denormalize(X,Xpara)
% Xmean = repmat(Xpara(1,:),[size(X,1) 1]);
% Xvar = repmat(Xpara(2,:),[size(X,1) 1]);
% 
% % X_n = (X - Xmean)./(2*sqrt(Xvar));
% X_n = X.*(2*sqrt(Xvar)) + Xmean;
% 
% end