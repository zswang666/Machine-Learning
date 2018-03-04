function y_predict = svmPredict(x,y,x_test,alpha,b,sig,mode)

m = size(x,1);
mt = size(x_test,1);
y_predict = zeros(mt,1);
for i=1:mt,
	x_test_i = ones(m,1)*x_test(i,:);
	y_predict(i) = (((alpha.*y).')*kernel(x,x_test_i,sig,mode)+b) >= 0;
end

end
