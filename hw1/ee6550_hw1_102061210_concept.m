function c_of_x = ee6550_hw1_102061210_concept(c, x)
% this function give label 1 to sample point xi located
% in concept(or hypothisis h) c, otherwise 0. The return value
% c_of_x is a length-of-x vector labeling x, ie. c_of_x(i) = label(xi)
    % lower_left = c(:,1) upper_right = c(:,2)
    c_of_x = zeros(size(x,1),1);
    for i=1:1:size(x,1)
        if x(i,1)>=c(1,1) && x(i,2)>=c(2,1) && x(i,1)<=c(1,2) && x(i,2)<=c(2,2),
            c_of_x(i,1) = 1;
        end
    end
end

