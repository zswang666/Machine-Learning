function a = kernel(x1,x2,sig,mode)

if mode==1,
    a = exp(-sum((x1-x2).^2,2)./2./(sig.^2));
elseif mode==2,
    a = sum(x1.*x2,2);
end

end