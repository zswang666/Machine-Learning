function a = gaussian_kernel(x1,x2,sig)

a = exp(-0.5*sum((x1-x2).^2,2)./(sig^2));

end