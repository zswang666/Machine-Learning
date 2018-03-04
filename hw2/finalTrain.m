load adult_training
x = adult_training(:,1:end-1);
y = adult_training(:,end); y = y - ~y;
load adult_testing
x_test = adult_testing(:,1:end-1);
y_test = adult_testing(:,end);

C = 1.7;
sig = 2.5;
tol = 0.01; 
maxIter = 1E8;
mode = 1;

% best_acc = 0;
% miniSize = 1000; % size of mini set
% numMiniSet = 15; % number of mini set
% for i=1:miniSize:miniSize*numMiniSet,
%     x_mini = x(i:i+miniSize-1,1:end);
%     y_mini = y(i:i+miniSize-1,end);
%     [alpha,b,x_mini_trim,y_mini_trim] = svmTrain(x_mini,y_mini,C,sig,tol,maxIter,mode);
%     y_predict = svmPredict(x_mini_trim,y_mini_trim,x_test,alpha,b,sig,mode);
%     accuracy = sum(y_predict==y_test)/length(y_test)
%     if accuracy > best_acc,
%         best_b = b;
%         best_alpha = alpha;
%         best_x = x;
%         best_y = y;
%     end
% end
% save best.mat best_b best_alpha best_x best_y C sig tol maxIter mode

[alpha,b,x_trim,y_trim] = svmTrain(x,y,C,sig,tol,maxIter,mode);
y_predict = svmPredict(x_trim,y_trim,x_test,alpha,b,sig,mode);
fprintf('Gaussian kernel with C=1.7, sigma=2.5 correctness : %f\n',sum(y_predict==y_test)./length(y_test));

% save result.mat C sig mode x_trim y_trim alpha b