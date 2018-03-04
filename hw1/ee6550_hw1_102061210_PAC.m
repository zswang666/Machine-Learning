% PAC learning
% P(R(hs)<epsilon) > 1-delta
% gaurantee parameter
myEps = 0.1; % epsilon ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
delt = 0.01; % delta ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Random samples S={x1, x2, x3...xm} of size m are drawn iid
% according to fixed but unknown distribution P over input space R^2.
% Now we assume P as 2 dimensional Gaussian distribution.
m = round(4/myEps*log(4/myEps)); % number of samples ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
r_xy = 0.4; % correlation coefficient of 2 dimensional Gaussian ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
sigma_x = 0.1; % standard deviation of x ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
sigma_y = 0.15; % standard deviation of y ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MU = [5 15]; % mean of Gaussian (P) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SIGMA = [sigma_x^2 r_xy*sigma_x*sigma_y; r_xy*sigma_x*sigma_y sigma_y^2]; % covariance matrix of Gaussion (P)x
x = mvnrnd(MU,SIGMA,m); % m-by-d matrix representing random samples S

% Select a qualified unknown concept c:input space->{0,1} such that P(c)>=2myEps,
% where c = [v u] refers to axis-aligned rectangular area constained
% by lower left corner v and upper right corner u. c is randomly chosen
% with criteria p_hat=(1/m)*(sigma i=1 to m, c(xi)) >= 3myEps
direct_input = 1; % direct-input enable ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if direct_input,
    c = [[4.7954; 14.9089] [5.3170; 15.3026]]; % directly assigned c +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    p_hat = 1/m*sum(ee6550_hw1_102061210_concept(c,x));
    if p_hat < 3*myEps,
        warning('Directly assigned c does not statify P(c)>2epsilon');
        return;
    end
end
c_notFound = 1;
while c_notFound && ~direct_input,
    c = [[5; 15]-rand(2,1)./3 [5; 15]+rand(2,1)./3]; % random chosen c ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    p_hat = 1/m*sum(ee6550_hw1_102061210_concept(c,x));
    if p_hat >= 3*myEps,
        c_notFound = 0;
    end
end
y = ee6550_hw1_102061210_concept(c,x); % correct labels

% positively labeled x
% points which are located inside concept rectangle
x_pos = zeros(size(y,1),2);
j = 1;
for i=1:1:size(x,1)
    if y(i,1)==1,
        x_pos(j,:) = x(i,:);
        j = j + 1;
    end
end
x_pos(j:size(x_pos,1),:) = [];
    
% PAC learning algorithm
% A minimum rectangle is found to encircle all
% positively-labeled points, where its lower-left
% corner vector may be smallest value of x and smallest
% value of y among all positively-labeled points, and
% upper-right corner vector may be largest value of x 
% and largest value of y among all positively-labeled points
init = 1;
for i=1:1:size(x,1)
    if y(i,1)==1,
        if init==1,
            hs = [x(i,:)' x(i,:)']; 
            init = 0;
        end
        if init==0,
            less_than = x(i,:)'<hs(:,1);
            greater_than = x(i,:)'>hs(:,2);
            hs(:,1) = ~less_than.*hs(:,1) + less_than.*x(i,:)';
            hs(:,2) = ~greater_than.*hs(:,2) + greater_than.*x(i,:)';
        end
    end
end
hs

% calculate q_hat
% q_hat is an estimator of generalization error R(hs),
% calculating the difference between c and hs, i.e. E(q_hat) = P(c\hs)
% y is a new sample to validate hs
y = mvnrnd(MU,SIGMA,m);
q_hat = 1/m*sum(xor(ee6550_hw1_102061210_concept(c,y),ee6550_hw1_102061210_concept(hs,y)));
q_hat

% figure of input points in scatter plot
% with blue points labeled negative (outside concept rectangle)
% and red point labeled positive (inside concept rectangle).
% Concept is represented as yellow rectangle.
% My target hypothesis hs is represented as magenta rectangle.
figure
scatter(x(:,1),x(:,2));
hold on; 
scatter(x_pos(:,1),x_pos(:,2), 'r');
rectangle('Position',[c(:,1)' (c(:,2)-c(:,1))'],'EdgeColor','y');
rectangle('Position',[hs(:,1)' (hs(:,2)-hs(:,1))'],'EdgeColor','m');
hold off;
xlabel('x');
ylabel('y');
txt1 = sprintf('sample point(blue points), +sample(red points)\nc(yellow rect), hs(magenta rect)');
title(txt1);
