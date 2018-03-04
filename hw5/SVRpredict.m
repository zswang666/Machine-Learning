function y_predict = SVRpredict(x_test,beta,x,b,kernel)

m = size(x,1);
mt = size(x_test,1);
y_predict = zeros(mt,1);
for i=1:mt,
	x_test_i = ones(m,1)*x_test(i,:);
	y_predict(i) = (beta.')*kernel(x,x_test_i) + b;
end

end