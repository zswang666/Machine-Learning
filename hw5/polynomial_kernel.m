function a = polynomial_kernel(x1,x2,d)

a = (1+sum(x1.*x2,2)).^d;

end