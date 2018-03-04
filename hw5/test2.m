%%% predicting sine function
clear
C = 25;
kernelChoice = 2;
sig = 0.8;
d = 3;

if kernelChoice==1,
    kernel = @linear_kernel;
elseif kernelChoice==2,
    kernel = @(x1,x2)gaussian_kernel(x1,x2,sig);
else
    kernel = @(x1,x2)polynomial_kernel(x1,x2,d);
end

x = randi(100,1,200)./10; x = x.';
y = sin(x);
myEps = (max(y)-min(y))*0.01;
tol = myEps*0.01;

x_test = randi(100,1,50)./10; x_test = x_test.';
y_test = sin(x_test);

[beta,x,b] = SVRtrain(x,y,tol,C,myEps,kernel);
y_predict = SVRpredict(x_test,beta,x,b,kernel);
fprintf('Error = %f\n',sum( max(abs(y_predict-y_test)-myEps,0) )/length(y_predict));
drawPoints(x_test,y_predict,y_test);