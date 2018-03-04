function [beta,x,b] = SVRtrain(x,y,tol,C,myEps,kernel)
m = size(x,1);
%%% initialization
beta = zeros(m,1);
F = -1.*y;

maxIter = 1500;
iter = 1;
numChanged = 1;
while numChanged>0 && iter<=maxIter,
    [numChanged,beta,F] = SVRexamine(x,y,tol,C,myEps,kernel,beta,F);
    iter = iter + 1;
end
if iter>=maxIter,
%     fprintf('reach max iteration\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compute b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Iplus_e = (beta>-C) & (beta<0);
Iminus_e = (beta>0) & (beta<C);
if sum(Iplus_e)==0,    fprintf('Iplus_e is empty.\n');  end
if sum(Iminus_e)==0,    fprintf('Iminus_e is empty.\n');  end
b = (myEps*(sum(Iplus_e)-sum(Iminus_e)) - sum(F(Iplus_e | Iminus_e))) / (sum(Iplus_e)+sum(Iminus_e));

% pe_index = find(Iplus_e);
% b = -myEps - F(pe_index(1));
if isnan(b),
    error('failed to compute offset b, try other parameters.');
end

end