function printdata(y_predict,y_test)

fprintf('\n');
for i=1:1:length(y_test),
    fprintf('(predict,true) = (%5f,%5f)\n',y_predict(i),y_test(i));
end

end