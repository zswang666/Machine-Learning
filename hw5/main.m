clear
%%%% parameters %++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nCross= 5; % n-fold cross validation, n = nCross 
kernelChoice = 2; % 1-->linear kernel, 2-->gaussian kernel, 3-->polynomial kernel
sigChoice = [0.2 0.5 1.2]; % sigma for gaussian kernel
Cchoice = [1 4 10 20 50]; % penalty coefficient in SVR
dChoice = [1 2 3 4 5]; % d for polynomial kernel
normOrNot = 1; % 1--> normalize y, 0--> use raw data

%%%% load data
load yacht_training
x = yacht_training(:,1:end-1);
y_raw = yacht_training(:,end);
% testing data
load yacht_testing
x_test = yacht_testing(:,1:end-1);
y_test = yacht_testing(:,end);
% size of data
m = size(x,1); % sample size of x
n = size(x,2); % dimension of x
%%%% normalize data
if normOrNot,
    [y,y_normPara] = normalize(y_raw);
else
    y = y_raw;
end
%%%% epsilon for SVR and tolerance
myEps = (max(y)-min(y))*0.01; %++++++++++++++++++++++++++++++++++++++++++++++++++++++++
tol = myEps*0.01;             %++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%%% index of cross validation
foldSize = round(m/nCross); % size of every fold
foldIndex = zeros(2,nCross);
for i=1:1:nCross,
   foldIndex(1,i) = (i-1)*foldSize + 1; % starting index of ith fold
   foldIndex(2,i) = min(i*foldSize,m); % ending index of ith fold
end
%%%% start cross validation
mincrossErr = 1E8;
for C = Cchoice,
    if kernelChoice==1,
        secChoice = 1;
    elseif kernelChoice==2.
        secChoice = sigChoice;
    else
        secChoice = dChoice;
    end
    for sec = secChoice,
        crossErr = 0;
        %%%% kernel
        if kernelChoice==1,
            kernel = @linear_kernel;
        elseif kernelChoice==2,
            kernel = @(x1,x2)gaussian_kernel(x1,x2,sec);
        else
            kernel = @(x1,x2)polynomial_kernel(x1,x2,sec);
        end
        for i=1:1:nCross,
            if i == 1,
                xFold = x(foldIndex(1,2):end,:);
                yFold = y(foldIndex(1,2):end);
            elseif i == nCross,
                xFold = x(1:foldIndex(2,nCross-1),:);
                yFold = y(1:foldIndex(2,nCross-1));
            else
                xFold = [x(1:foldIndex(2,i-1),:); x(foldIndex(1,i+1):end,:)];
                yFold = [y(1:foldIndex(2,i-1)); y(foldIndex(1,i+1):end)];
            end
            [beta,xFold,b] = SVRtrain(xFold,yFold,tol,C,myEps,kernel);
            y_predict_raw = SVRpredict(x(foldIndex(1,i):foldIndex(2,i),:),beta,xFold,b,kernel);
            if normOrNot,   y_predict = denormalize(y_predict_raw,y_normPara);
            else            y_predict = y_predict_raw;  end
            crossErr = crossErr + sum( max(abs(y_predict-y(foldIndex(1,i):foldIndex(2,i),:))-myEps,0) )/length(y_predict);
        end
        if kernelChoice==1,
            fprintf('C=%3.1f error: %f\n',C,crossErr);
        elseif kernelChoice==2,
            fprintf('(C,sigma)=(%3.1f,%3.1f) error: %f\n',C,sec,crossErr);
        else
            fprintf('(C,d)=(%3.1f,%3.1f) error: %f\n',C,sec,crossErr);
        end
        if crossErr < mincrossErr,
            bestC = C;
            bestSec = sec;
            mincrossErr = crossErr;
        end
    end
end
%%%% train entire training set
fprintf('\n');
if kernelChoice==1,
    kernel = @linear_kernel;
    fprintf('Using linear kernel with C=%3.1f\n',bestC);
elseif kernelChoice==2,
    kernel = @(x1,x2)gaussian_kernel(x1,x2,bestSec);
    fprintf('Using gaussian kernel with sigma=%3.1f, C=%3.1f\n',bestSec,bestC);
else
    kernel = @(x1,x2)polynomial_kernel(x1,x2,bestSec);
    fprintf('Using polynomial kernel with d=%3.1f, C=%3.1f\n',bestSec,bestC);
end
[beta,x,b] = SVRtrain(x,y,tol,bestC,myEps,kernel);
y_predict_raw = SVRpredict(x_test,beta,x,b,kernel);
if normOrNot,   y_predict = denormalize(y_predict_raw,y_normPara);
else            y_predict = y_predict_raw;  end
fprintf('Error=%f\n',sum( max(abs(y_predict-y_test)-myEps,0) )/length(y_predict));
%%%% visualize
drawPoints(x_test,y_predict,y_test);
% printdata(y_predict,y_test);