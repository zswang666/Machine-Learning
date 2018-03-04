function y_predict = kernelPerceptron_predict(x_test,x,y,kernel,alpha)

m = size(x_test,1);
y_predict = zeros(size(x_test,1),1);
for i=1:1:m,
    x_test_i_mat = ones(size(x,1),1) * x_test(i,:);
    y_predict(i) = (alpha.'*(y.*kernel(x_test_i_mat,x))) > 0;
end
    
end