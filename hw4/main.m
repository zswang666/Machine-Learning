whichData = 2; % 1-->adult, 2-->iris
nCross = 5; % nth fold cross validation
TChoice = [1 2 3 5];
sigChoice = [0.1 0.4 1.1];
kernelChoice = 2; % 1-->linear kernel, 2-->gaussian kernel

if whichData == 1,
    load adult_training
    x = adult_training(1:3000,1:end-1);
    y = adult_training(1:3000,end); y = y - ~y;
    load adult_testing
    x_test = adult_testing(:,1:end-1);
    y_test = adult_testing(:,end);
elseif whichData == 2,
    load iris_set_ver_training
    x = iris_set_ver_training(:,1:end-1);
    y = iris_set_ver_training(:,end); y = y - ~y;
    load iris_set_ver_testing
    x_test = iris_set_ver_testing(:,1:end-1);
    y_test = iris_set_ver_testing(:,end);
end
m = size(x,1); % sample size of x
n = size(x,2); % dimension of x

%%
% index of cross validation
foldSize = round(m/nCross); % size of every fold
foldIndex = zeros(2,nCross);
for i=1:1:nCross,
   foldIndex(1,i) = (i-1)*foldSize + 1; % starting index of ith fold
   foldIndex(2,i) = min(i*foldSize,m); % ending index of ith fold
end

mincrossErr = m;
for T = TChoice.*(foldSize*(nCross-1)),
    for sig = sigChoice,
        if kernelChoice==1,
            kernel = @linear_kernel;
        else
            kernel = @(x1,x2)gaussian_kernel(x1,x2,sig);
        end
        crossErr = 0;
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
            [alpha] = kernelPerceptron(xFold,yFold,T,kernel);
            y_predict = kernelPerceptron_predict(x(foldIndex(1,i):foldIndex(2,i),:),xFold,yFold,kernel,alpha);
            y_predict = y_predict - ~y_predict;
            crossErr = crossErr + sum(y_predict~=y(foldIndex(1,i):foldIndex(2,i)));
            accuracy = sum(y_predict==y(foldIndex(1,i):foldIndex(2,i))) / length(y(foldIndex(1,i):foldIndex(2,i)));
            if kernelChoice==1,
%                 fprintf('Accuracy of %d fold with T = %d is %.3f%%\n',i,T/(foldSize*(nCross-1)),accuracy*100);
            else
%                 fprintf('Accuracy of %d fold with T = %d, sigma = %f is %.3f%%\n',i,T/(foldSize*(nCross-1)),sig,accuracy*100);
            end
        end
        fprintf('T=%d, sig=%f, err= %f\n', T/(foldSize*(nCross-1)), sig, crossErr/nCross/foldSize);
        if crossErr < mincrossErr,
            bestT = T/(foldSize*(nCross-1));
            bestSig = sig;
            mincrossErr = crossErr;
        end
        if kernelChoice==1,
            break;
        end
    end
end

%%
%%% training over entire data set with bestT obtained from cross validation
if kernelChoice==1,
    fprintf('\n\nBy %d-fold cross validation, using linear kernel, best T = %d.\n',nCross,bestT);
    kernel = @linear_kernel;
else
    fprintf('\n\nBy %d-fold cross validation, using gaussiana kernel, best T = %d, best sigma = %f.\n',nCross,bestT,bestSig);
    kernel = @(x1,x2)gaussian_kernel(x1,x2,bestSig);
end
fprintf('Now we are going to train entire data set with best T...\n\n');
[bestAlpha] = kernelPerceptron(x,y,bestT*m,kernel);

%%% prediction over testing data with classifier derived from entire training set
fprintf('Following shows accuracy of prediction over testing data,\n');
y_predict = kernelPerceptron_predict(x_test,x,y,kernel,bestAlpha);
fprintf('Accuracy = %.3f%% with classifier derived from entire training set\n',sum(y_predict==y_test)/length(y_test)*100);
