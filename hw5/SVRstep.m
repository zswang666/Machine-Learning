function [count,beta,F] = SVRstep(i,j,x,y,tol,C,myEps,kernel,beta,F)
% fprintf('(%d,%d)|(%f,%f):',i,j,beta(i),beta(j));
if i == j,
    count = 0;
    return;
end
m = size(x,1);

gamma = beta(i) + beta(j);

Kii = kernel(x(i,:),x(i,:));
Kjj = kernel(x(j,:),x(j,:));
Kij = kernel(x(i,:),x(j,:));
eta = Kii + Kjj - 2*Kij;

L = max(-C,gamma-C);
H = min(C,gamma+C);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update beta(j) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% delta = 2*myEps/eta;
% beta_jNew = beta(j) + 1/eta*(F(i)-F(j));
% beta_iNew = gamma - beta_jNew;
% if beta_iNew*beta_jNew<0,
%     if abs(beta_jNew)>=delta && abs(beta_iNew)>=delta,
%         beta_jNew = beta_jNew - sign(beta_jNew)*delta;
%     else
%         beta_jNew = (abs(beta_jNew)>abs(beta_iNew)) * gamma;
%     end
% end
% beta_jNew = min(max(beta_jNew,L),H);

if eta==0, % eta = 0
%     fprintf('\teta=0')
    F_idj = F(i)-F(j);
    if gamma==0,
        if F_idj<-2*myEps,                            beta_jNew = L;
        elseif F_idj>2*myEps,                         beta_jNew = H;
        else                                          beta_jNew = 0;
        end
    elseif (gamma>0) && (gamma<C),
        if F_idj<-2*myEps,                            beta_jNew = L;
        elseif (F_idj>0) && (F_idj<2*myEps),          beta_jNew = gamma;
        elseif F_idj>=2*myEps,                        beta_jNew = H;
        else                                          beta_jNew = 0;
        end
    elseif gamma==C,
        if F_idj<=0,     beta_jNew = L;                        
        else             beta_jNew = H;
        end
    elseif gamma>C,
        if F_idj<0,       beta_jNew = L;                        
        else              beta_jNew = H;
        end
    elseif (gamma>-C) && (gamma<0),
        if F_idj<-2*myEps,                     beta_jNew = L;                          
        elseif (F_idj>-2*myEps) && (F_idj<0),  beta_jNew = gamma;
        elseif F_idj>2*myEps,                  beta_jNew = H;
        else                                   beta_jNew = 0;
        end
    elseif gamma==-C,
        if F_idj<0,       beta_jNew = L;                        
        else              beta_jNew = H;
        end
    elseif gamma<-C,
        if F_idj<=0,       beta_jNew = L;                        
        else               beta_jNew = H;
        end
    end
elseif eta>0,  % eta ~= 0
    F_idj = F(i)-F(j);
    beta_j0 = beta(j) + (F_idj)/eta;
    beta_j2 = beta(j) + (F_idj+2*myEps)/eta;
    beta_jm2 = beta(j) + (F_idj-2*myEps)/eta;
    if gamma==0,
%         fprintf('\tgamma1')
        if beta_j2<=L,                          beta_jNew = L; 
        elseif (beta_j2>L) && (beta_j2<0),      beta_jNew = beta_j2;  
        elseif (beta_jm2>0) && (beta_jm2<H),    beta_jNew = beta_jm2; 
        elseif beta_jm2>=H,                     beta_jNew = H; 
        else                                    beta_jNew = 0; 
        end
    elseif (gamma>0) && (gamma<C),
%         fprintf('\tgamma2')
        if beta_j2<=L,                           beta_jNew = L;
        elseif (beta_j2>L) && (beta_j2<0),       beta_jNew = beta_j2;
        elseif (beta_j0<=0),                     beta_jNew = 0;
        elseif (beta_j0>0) && (beta_j0<gamma),   beta_jNew = beta_j0;
        elseif (beta_jm2>gamma) && (beta_jm2<H), beta_jNew = beta_jm2;
        elseif beta_jm2>=H,                      beta_jNew = H;
        else                                     beta_jNew = gamma;
        end
    elseif gamma==C,
%         fprintf('\tgamma3')
        if beta_j0<=L,                     beta_jNew = L;
        elseif (beta_j0>L) && (beta_j0<H), beta_jNew = beta_j0;
        else                               beta_jNew = H;
        end
    elseif gamma>C,
%         fprintf('\tgamma4')
        if beta_j0<L,                         beta_jNew = L;
        elseif (beta_j0>=L) && (beta_j0<=H),  beta_jNew = beta_j0;
        else                                  beta_jNew = H;
        end
    elseif (gamma>-C) && (gamma<0),
%         fprintf('\tgamma5')
        if beta_j2<=L,                               beta_jNew = L;
        elseif (beta_j2>L) && (beta_j2<gamma),       beta_jNew = beta_j2;
        elseif (beta_j0<=gamma),                     beta_jNew = gamma;
        elseif (beta_j0>gamma) && (beta_j0<0),       beta_jNew = beta_j0;
        elseif (beta_jm2>0) && (beta_jm2<H),         beta_jNew = beta_jm2;
        elseif beta_jm2>=H,                          beta_jNew = H;
        else                                         beta_jNew = 0;
        end
    elseif gamma==-C,
%         fprintf('\tgamma6')
        if beta_j0<=L,                      beta_jNew = L;
        elseif (beta_j0>L) && (beta_j0<H),  beta_jNew = beta_j0;
        else                                beta_jNew = H;
        end
    elseif gamma<-C,
%         fprintf('\tgamma7')
        if beta_j0<L,                        beta_jNew = L;
        elseif (beta_j0>=L) && (beta_j0<=H), beta_jNew = beta_j0;
        else                                 beta_jNew = H;
        end
    end
else % eta<0, never happen
    error('eta smaller than 0.');
end
if abs(beta_jNew-beta(j)) <= eps,
%     fprintf('\tdelta_betaj close to 0\n')
    count = 0;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update beta(i) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_iNew = gamma - beta_jNew;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update F %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_i = ones(m,1)*x(i,:);
Kiall = kernel(x,x_i);
x_j = ones(m,1)*x(j,:);
Kjall = kernel(x,x_j);
% fprintf('\tdelta_bj = %d',(beta_jNew-beta(j)));
F = F + (beta_jNew-beta(j))*(Kjall-Kiall);

beta(i) = beta_iNew;
beta(j) = beta_jNew;

count = 1;
% fprintf('\n');
end