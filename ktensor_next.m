function t = ktensor_next(A,dims)

N=length(A);

reconst=A{N};

for n=N:-1:3
    reconst=kr(reconst,A{n-1});
end

sA=size(A{N});
d=sA(2);

x=1:d;
%lam=exp(-0.2*x) +0.0001;
lam=exp(0.2*x) +0.0001;
%lam=exp(-x); %+0.0001;
lam=diag(lam);

t=A{1}*lam*reconst';
t=A{1}*reconst';
t=reshape(t,dims);

return;
