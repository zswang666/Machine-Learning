function [alpha] = kernelPerceptron(x,y,T,kernel)

m = size(x,1);
% check sample size of input x is equal to T
% if m>T,
%     warning('sample size of x should be smaller T in kernelPerceptron function.\n');
%     return;
% end

%%%% compute index of x corresponding to t
x_index = zeros(T,1);
for i=1:m:T,
    if i+m-1>T,
        break;
    end
    sequence = 1:1:m;
    x_index(i:i+m-1,:) = sequence;
end
x_index(i:T,:) = 1:1:T-i+1;

alpha = zeros(m,1);
for t=1:1:T,
    % sequentially choose received x
    now_index = x_index(t);
    x_received = x(now_index,:);
    y_received = y(now_index);
    
    xt_mat = ones(m,1) * x_received; % used for calculating kernel
    
    yt_predict = (alpha.'*(y.*kernel(xt_mat,x))) > 0; % compute prediction of yt
   
    if  y_received~=yt_predict,
        alpha(now_index) = alpha(now_index) + 1;
    end
end

end