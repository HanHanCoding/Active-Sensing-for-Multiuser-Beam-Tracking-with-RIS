function y=MAXMIN_rate_direct(x,A,B,omega,K)
    tmp=zeros(K,K);
    for i0=1:K
        for k0=1:K
            tmp(i0,k0)=abs(x'*A(:,:,i0,k0)+B(i0,k0))^2;
        end
    end
    yt=zeros(K,1);
    for k0=1:K
        tmp1=sum(tmp(:,k0));
        tmp2=tmp1-tmp(k0,k0);
        yt(k0)=log(1+tmp(k0,k0)/(tmp2+1));
    end
    % max min R_k is equavalent to min max -R_k
    % y=-omega*yt; % this is the sum rate objective

    % Redefine your cost function as max(-U(x)) and use the CG solver to minimize it.
    % The CG solver is designed to minimize the cost function
    y=max(-1*yt); % this is the MAXMIN rate objective
end

