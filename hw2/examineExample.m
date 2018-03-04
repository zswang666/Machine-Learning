function [count,alpha,F] = examineExample(i2,alpha,y,C,x,tol,F,sig,mode)

I0 = (alpha>0 & alpha<C);
I012 = (I0==1) | ((y==1) & (alpha==0)) | ((y==-1) & (alpha==C));
I034 = (I0==1) | ((y==1) & (alpha==C)) | ((y==-1) & (alpha==0));
Bup = min(F(I012)) + tol;
Blow = max(F(I034)) - tol;
violate = 0;
if (I012(i2) && (F(i2)<Blow)) || (I034(i2) && (F(i2) > Bup)),
    violate = 1;
end

if violate, % alph2 with index i2 violates KKT condition
    index = find(I0);
%     index.'
%     alpha.'
    
    % non-zero and non-C alpha with largest step
    if sum(I0)>1,
        [~,i1] = max(abs(F(index)-F(i2))); % second heuristic maximize abs(F1-F2)
        [en,alpha,F] = takeStep(index(i1),i2,alpha,y,C,x,F,sig,mode);
        if en,
            count = 1;
            return;
        end
    end
    % if largest step fails, loop over all non-zero and non-C alpha
    indexLength = length(index);
    if indexLength>1,
        randSeq = (circshift( index, randi([1 indexLength],1))).';
    else
        randSeq = index;
    end
    for i=1:1:indexLength,
        [en,alpha,F] = takeStep(randSeq(i),i2,alpha,y,C,x,F,sig,mode);
        if en,
            count = 1;
            return;
        end
    end
    
    % if the above 2 all fails, loop over all possible i1
    alphaLength = length(alpha);
    randSeq = (circshift( (1:1:alphaLength).', randi([1 alphaLength],1))).';
    for i=1:1:alphaLength,
        [en,alpha,F] = takeStep(randSeq(i),i2,alpha,y,C,x,F,sig,mode);
        if en,
            count = 1;
            return;
        end
    end
end

count = 0;
return;
end