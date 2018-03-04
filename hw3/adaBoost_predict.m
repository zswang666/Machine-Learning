function y_predict = adaBoost_predict(x_test,alpha,threshold)
% convert x_test to binary data
x_test = x_test >= repmat(threshold,[size(x_test,1) 1]);

% make x_test from 0/1 to -1/+1
x_test = x_test - ~x_test; % conversion to (-1,+1) is only for convenience

% % make h a row vector
% h = h.';
% 
% m = size(x_test,1); % size of x_test
% 
% % trim zero component for computational efficiency
% hNot0 = find(h~=0); % row vector
% x_test = x_test(:,hNot0); % only non-zero part will be taken into consideration
% alpha = alpha(hNot0.'); % still column vector
% h = h(hNot0);
% 
% h = h - 1; % 0-->not invert, 1-->invert
% h = repmat(h,[m,1]);

% x_test is converted into (-1,+1), so the following computation is valid
% predictMatrix = x_test.*~h + -x_test.*h; % non-invert term + invert term
% y_predict = (predictMatrix*alpha) > 0; % column vector
y_predict = (x_test*alpha(:,1) + -1*x_test*alpha(:,2)) > 0;
end