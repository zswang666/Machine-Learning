data = 1; % choose data, data=1-->adult, data=2-->iris_set_ver
mode = 1; % mode of kernel, mode=1-->gaussian kernel, mode=2-->linear kernel 
tol = 0.01; % tolerance

maxIter = 1E4;
miniSize = 2000; % size of mini set
numMiniSet = 10; % number of mini set
C_test = [1.2 1.7]; % slack penalty coefficient for cross validation
sig_test = [1.4 1.7 2.1 2.5 2.7]; % variance of Gaussian kernel for cross validation

if data == 1,
    load adult_training
    x = adult_training(:,1:end-1);
    y = adult_training(:,end); y = y - ~y; % make y to be +1 or -1
    % testing data
    load adult_testing
    x_test = adult_testing(:,1:end-1);
    y_test = adult_testing(:,end);
elseif data == 2,
    load iris_set_ver_training
    x = iris_set_ver_training(:,1:end-1);
    y = iris_set_ver_training(:,end); y = y - ~y; % make y to be +1 or -1
    % testing data
    load iris_set_ver_testing
    x_test = iris_set_ver_testing(:,1:end-1);
    y_test = iris_set_ver_testing(:,end);
end

if data == 1,
    if mode == 1, % gaussian kernel
        outestLoop1 = length(C_test);
        outestLoop2 = length(sig_test);
        accuracy = zeros(outestLoop1,outestLoop2);
        for k=1:1:outestLoop1,
            C = C_test(k); % try different parameter
            for m=1:1:outestLoop2,
                sig = sig_test(m); % try different parameter
                sumPredict = 0;
                randStart = randi([1 round(size(x,1)-miniSize*numMiniSet-1)],1);
                % non-overlap validation sample
                x_valid = [x(1:randStart-1,:); x(randStart+miniSize*numMiniSet+1:end,:)];
                y_valid = [y(1:randStart-1,end); y(randStart+miniSize*numMiniSet+1:end,end)]; y_valid = y_valid==1;
                for i=randStart:miniSize:randStart+numMiniSet*miniSize-1,
                    x_mini = x(i:i+miniSize-1,1:end);
                    y_mini = y(i:i+miniSize-1,end);
                    [alpha,b,x_mini_trim,y_mini_trim] = svmTrain(x_mini,y_mini,C,sig,tol,maxIter,mode);
                    y_predict = svmPredict(x_mini_trim,y_mini_trim,x_valid,alpha,b,sig,mode);
                    sumPredict = sumPredict + sum(y_predict==y_valid)./length(y_valid);
                end
                accuracy(k,m) = sumPredict/numMiniSet;
            end
        end
    elseif mode == 2, % linear kernel
        outestLoop1 = length(C_test);
        accuracy = zeros(outestLoop1,1);
        for k=1:1:outestLoop1,
            C = C_test(k); % try different parameter
            sumPredict = 0;
            randStart = randi([1 round(size(x,1)-miniSize*numMiniSet-1)],1);
            % non-overlap validation sample
            x_valid = [x(1:randStart-1,:); x(randStart+miniSize*numMiniSet+1:end,:)];
            y_valid = [y(1:randStart-1,end); y(randStart+miniSize*numMiniSet+1:end,end)]; y_valid = y_valid==1;
            for i=randStart:miniSize:randStart+numMiniSet*miniSize-1,
                x_mini = x(i:i+miniSize-1,1:end);
                y_mini = y(i:i+miniSize-1,end);
                [alpha,b,x_mini_trim,y_mini_trim] = svmTrain(x_mini,y_mini,C,0,tol,maxIter,mode);
                y_predict = svmPredict(x_mini_trim,y_mini_trim,x_valid,alpha,b,0,mode);
                sumPredict = sumPredict + sum(y_predict==y_valid)./length(y_valid);
            end
            accuracy(k) = sumPredict/numMiniSet;
        end
    end
elseif data == 2,
    if mode == 1,
        outestLoop1 = length(C_test);
        outestLoop2 = length(sig_test);
        accuracy = zeros(outestLoop1,outestLoop2);
        for k=1:1:outestLoop1,
            C = C_test(k); % try different parameter
            for m=1:1:outestLoop2,
                sig = sig_test(m); % try different parameter 
                [alpha,b,x_trim,y_trim] = svmTrain(x,y,C,sig,tol,maxIter,mode);
                y_predict = svmPredict(x_trim,y_trim,x_test,alpha,b,sig,mode);
                accuracy(k,m) = sum(y_predict==y_test)./length(y_test);
            end
        end
    elseif mode == 2,
        outestLoop1 = length(C_test);
        accuracy = zeros(outestLoop1,1);
        for k=1:1:outestLoop1,
            C = C_test(k); % try different parameter
            [alpha,b,x_trim,y_trim] = svmTrain(x,y,C,0,tol,maxIter,mode);
            y_predict = svmPredict(x_trim,y_trim,x_test,alpha,b,0,mode);
            accuracy(k) = sum(y_predict==y_test)./length(y_test);
        end
    end
end