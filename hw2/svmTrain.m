function [alpha,b,x,y] = svmTrain(x,y,C,sig,tol,maxIter,mode)

m = size(x,1);
% initialize alpha (a column vector) to zeros
alpha = zeros(m,1);
% initialize error cache no b
F = -1.*y; 

% computing kernel matrix
% K = zeros(m,m);
% for i=1:1:m,
%     x_i = ones(m,1)*x(i,:);
%     K(i,:) = kernel(x,x_i,sig,mode);
% end

iter = 0;
numChanged = 0;
examineAll = 1;
while (numChanged>0 || examineAll) && iter<maxIter,
   numChanged = 0;
   if examineAll,
       for i=1:1:m, % loop over entire training set
           [count,alpha,F] = examineExample(i,alpha,y,C,x,tol,F,sig,mode);
           numChanged = numChanged + count;
%            fprintf('examineAll %d\n',i);
       end
   else
       index = find(alpha(i,1)~=0 && alpha(i,1)~=C);
       for i=1:1:length(index), % loop over examples where alpha is not 0 & not C, repeated pass non-bound alpha           
           [count,alpha,F] = examineExample(index(i),alpha,y,C,x,eps,F,sig,mode);
           numChanged = numChanged + count;
%            fprintf('examineNon0NonC %d\n',i);
       end
   end
   if examineAll==1,
       examineAll = 0;
   elseif numChanged==0,
       examineAll = 1;
   end     
   iter = iter + 1;
end

% calculating b
sup_vec_index = find(alpha>0 & alpha<C);
m = size(x,1);
x_sup1 = ones(m,1)*x(sup_vec_index(1),:);
K = kernel(x,x_sup1,sig,mode);
b = y(sup_vec_index(1)) - sum(alpha.*y.*K);
% b = y(sup_vec_index(1)) - sum(alpha.*y.*K(:,sup_vec_index(1)));

alphaNot0 = alpha~=0;
alpha = alpha(alphaNot0);
y = y(alphaNot0);
x = x(alphaNot0,:);

end