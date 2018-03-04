function drawPoints(x,y_predict,y_test)
if size(x,2)==1,
    figure,
    p1 = plot(x,y_predict,'bo');
    hold on;
    p2 = plot(x,y_test,'r*');
    hold off;
    title('values of all points');
    legend([p1,p2],'predict','test');
    
    figure,
    plot(y_predict,y_test,'x');
    title('predict-true');
    xlabel('test');
    ylabel('predict');
else
%     for i=1:1:size(x,2),
%         figure,
%         p1 = plot(x(:,i),y_predict,'bo');
%         hold on;
%         p2 = plot(x(:,i),y_test,'r*');
%         hold off;
%         title('values of all points');
%         legend([p1,p2],'predict','test');
%     end
    figure,
    plot(y_predict,y_test,'x');
    title('predict-true');
    xlabel('test');
    ylabel('predict');
end

end