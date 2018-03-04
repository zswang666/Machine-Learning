clear
%%% parameters +++++++++++++++++++++++++++++++++++++++
C = 25;
kernelChoice = 3;
sig = 0.8;
d = 1;
if kernelChoice==1,
    kernel = @linear_kernel;
elseif kernelChoice==2,
    kernel = @(x1,x2)gaussian_kernel(x1,x2,sig);
else
    kernel = @(x1,x2)polynomial_kernel(x1,x2,d);
end
%%% load data
load yacht_training
x = yacht_training(:,1:end-1);
y_raw = yacht_training(:,end);
% testing data
load yacht_testing
x_test = yacht_testing(:,1:end-1);
y_test = yacht_testing(:,end);
[y,y_normPara] = normalize(y_raw);
%%% epsilon and tolerance 
myEps = (max(y)-min(y))*0.01;
tol = myEps*0.01;
%%% train and predict
[beta,x,b] = SVRtrain(x,y,tol,C,myEps,kernel);
y_predict = SVRpredict(x_test,beta,x,b,kernel);
%%% visualize
fprintf('Error = %f\n',sum( max(abs(y_predict-y_test)-myEps,0) )/length(y_predict));
drawPoints(x_test,y_predict,y_test);