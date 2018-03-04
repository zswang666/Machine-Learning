% This program is to validate gaurantee of my PAC
% learning algorithm (hw1.m) as P(R(hs)<epsilong) > 1-delta.
% ***COMMENT ALL CODE CREATING FIGURE IN hw1.m BEFORE RUNNING THIS PROGRAM
delt = 0.01; % must be set to the same value as delt in hw1.m
numFail = 0; % number of hs that have R(hs)>epsilon 
for i=1:1:10/delt
    ee6550_hw1_102061210_PAC;
    if q_hat>0.9*myEps,
        numFail = numFail + 1;
    end
end
numFail
