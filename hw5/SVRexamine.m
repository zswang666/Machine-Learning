function [count,beta,F] = SVRexamine(x,y,tol,C,myEps,kernel,beta,F)

I0 = (beta==0);
Iplus_e = (beta>-C) & (beta<0);
Iminus_e = (beta>0) & (beta<C);
Iplus = (beta==-C);
Iminus = (beta==C);

[Bup,bu] = max(F(I0));
[Blow,bl] = min(F(I0));

if (Bup-Blow)>2*(myEps+tol),
    [count,beta,F] = SVRstep(bu,bl,x,y,tol,C,myEps,kernel,beta,F);
    if count==1,
%         fprintf('v1\n');
        return;
    end
end

[minFp,p] = min(F(Iplus_e | Iplus));
if isempty(minFp),
    count = 1;
%     fprintf('v2empty\n');
    return;
end
if (minFp<(Bup-tol)),
    [count,beta,F] = SVRstep(p,bu,x,y,tol,C,myEps,kernel,beta,F);
    if count==1,
%         fprintf('v2\n');
        return;
    end
end

[maxFm,m] = max(F(Iminus_e | Iminus));
if isempty(maxFm),
    count = 1;
%     fprintf('v3empty\n');
    return;
end
if (maxFm>(Blow+tol)),
    [count,beta,F] = SVRstep(m,bl,x,y,tol,C,myEps,kernel,beta,F);
    if count==1,
%         fprintf('v3\n');
        return;
    end
end

% fprintf('noV\n');
count = 0;

end