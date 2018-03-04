function [count,alpha,F] = takeStep(i1,i2,alpha,y,C,x,F,sig,mode)
if i1 == i2,
    count = 0;
    return;
end

alph1 = alpha(i1,1);    alph2 = alpha(i2,1);
y1 = y(i1,1);   y2 = y(i2,1);
s = y1 * y2;

% compute L and H
gamma = alph1 + s*alph2;
if s == 1,
    L = max(0, gamma-C);
    H = min(C, gamma);
else
    L = max(0, -1*gamma);
    H = min(C, C-gamma);
end
if L == H,
    count = 0;
    return;
end

m = size(x,1);
x_i1 = ones(m,1)*x(i1,:);
K(1,:) = kernel(x,x_i1,sig,mode);
x_i2 = ones(m,1)*x(i2,:);
K(2,:) = kernel(x,x_i2,sig,mode);

% k11 = K(i1,i1);
% k12 = K(i1,i2);
% k22 = K(i2,i2);
k11 = K(1,i1);
k12 = K(1,i2);
k22 = K(2,i2);
eta = k11 + k22 - 2*k12; % eta always >= 0

myEps = eps;

% compute new clipping alph2
if eta>0,
    a2 = alph2 + y2*(F(i1)-F(i2))/eta;
    if a2 < L,
        a2 = L;
    elseif a2 > H,
        a2 = H;
    end
else % eta==0
    Lobj = y2*(F(i1)-F(i2))*L; % objective function at a2=L
    Hobj = y2*(F(i1)-F(i2))*H; % objective function at a2=H
    if Lobj < Hobj-myEps,
        a2 = L;
    elseif Lobj > Hobj+myEps,
        a2 = H;
    else
        a2 = alph2;
    end
end

if a2 < 1E-8,
    a2 = 0;
elseif a2 > C-1E-8,
    a2 = C;
end

if abs(a2-alph2) < myEps*(a2+alph2+myEps),
% if abs(a2-alph2) < 1E-8,
    count = 0;
    return;
end

% compute new alph1
a1 = alph1 + s*(alph2-a2);

% update error cache
% F = F + y1*(a1-alph1)*K(:,i1) + y2*(a2-alph2)*K(:,i2);
F = F + y1*(a1-alph1)*(K(1,:).') + y2*(a2-alph2)*(K(2,:).');
% store a1 and a2 in alpha
alpha(i1,1) = a1;   alpha(i2,1) = a2;

count = 1;
end
    