% parameters
nCross = 5; % nth fold cross validation
TChoice = 100; % choices of T while doing cross validation
whichData = 1; % 1-->adult, 2-->iris

if whichData == 1,
    load adult_training
    x = adult_training(:,1:end-1);
    y = adult_training(:,end);
    load adult_testing
    x_test = adult_testing(:,1:end-1);
    y_test = adult_testing(:,end);
elseif whichData == 2,
    load iris_set_ver_training
    x = iris_set_ver_training(:,1:end-1);
    y = iris_set_ver_training(:,end);
    load iris_set_ver_testing
    x_test = iris_set_ver_testing(:,1:end-1);
    y_test = iris_set_ver_testing(:,end);
end
m = size(x,1); % sample size of x
n = size(x,2); % dimension of x

% index of cross validation
foldSize = round(m/nCross); % size of every fold
foldIndex = zeros(2,nCross);
for i=1:1:nCross,
   foldIndex(1,i) = (i-1)*foldSize + 1; % starting index of ith fold
   foldIndex(2,i) = min(i*foldSize,m); % ending index of ith fold
end

%% perform cross validation
maxAcc = 0;
bestT = 1;
bestT2 = 1;
bestAlpha2 = zeros(n,1);
bestThreshold2 = zeros(1,n);
mincrossErr = m;
for T = TChoice,
    crossErr = 0;
    for i=1:1:nCross,
        if i == 1,
            [alpha,threshold] = adaBoost(x(foldIndex(1,2):end,:),y(foldIndex(1,2):end),T);
        elseif i == nCross,
            [alpha,threshold] = adaBoost(x(1:foldIndex(2,nCross-1),:),y(1:foldIndex(2,nCross-1)),T);
        else
            [alpha,threshold] = adaBoost([x(1:foldIndex(2,i-1),:); x(foldIndex(1,i+1):end,:)],[y(1:foldIndex(2,i-1)); y(foldIndex(1,i+1):end)],T);
        end
        y_predict = adaBoost_predict(x(foldIndex(1,i):foldIndex(2,i),:),alpha,threshold);
        crossErr = crossErr + sum(y_predict~=y(foldIndex(1,i):foldIndex(2,i)));
        accuracy = sum(y_predict==y(foldIndex(1,i):foldIndex(2,i))) / length(y(foldIndex(1,i):foldIndex(2,i)));
        fprintf('Accuracy of %d fold with T = %d is %.3f%%\n',i,T,accuracy*100);
        if accuracy > maxAcc,
            bestAlpha2 = alpha;
            bestT2 = T;
            bestFold = i;
            bestThreshold2 = threshold;
            maxAcc = accuracy;
        end
    end
%     crossErr/nCross/foldSize
    if crossErr < mincrossErr,
        bestT = T;
        mincrossErr = crossErr;
    end
end

%% training over entire data set with bestT obtained from cross validation
fprintf('\n\nBy %d-fold cross validation, best T = %d.\n',nCross,bestT);
fprintf('Now we are going to train entire data set with best T...\n\n');
[bestAlpha,bestThreshold] = adaBoost(x,y,bestT);

%% prediction over testing data with classifier derived from entire training set
fprintf('Following shows accuracy of prediction over testing data,\n');
y_predict = adaBoost_predict(x_test,bestAlpha,bestThreshold);
fprintf('Accuracy = %.3f%% with classifier derived from entire training set\n',sum(y_predict==y_test)/length(y_test)*100);
%% prediction over testing data with classifier derived from specific fold that has highest accuracy during cross validation
y_predict = adaBoost_predict(x_test,bestAlpha2,bestThreshold2);
fprintf('Accuracy = %.3f%% with classifier derived from %dth fold and T = %d\n', ...
                        sum(y_predict==y_test)/length(y_test)*100,bestFold,bestT2);
