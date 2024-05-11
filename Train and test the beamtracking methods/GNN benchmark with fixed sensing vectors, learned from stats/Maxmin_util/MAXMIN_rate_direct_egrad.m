function yA=MAXMIN_rate_direct_egrad(x,A,B,omega,K,N,AA)
    tmp=zeros(K,K);
    for i0=1:K
        for k0=1:K
            tmp(i0,k0)=abs(x'*A(:,:,i0,k0)+B(i0,k0))^2;
        end
    end
    % compute the rate for each of the user
    yt_rate_all=zeros(K,1);
    for k0=1:K
        tmp1_rate=sum(tmp(:,k0));
        tmp2_rate=tmp1_rate-tmp(k0,k0);
        yt_rate_all(k0)=log(1+tmp(k0,k0)/(tmp2_rate+1));
    end
    % find out which user has the min rate at the current x (theta)
    [~,index] = min(yt_rate_all);

    % compute the euclidean gradient using the subgradient method
    vA=zeros(N,1,K);
    for k0=1:K
        Bt=zeros(N,1);
        for i0=1:K
            Bt=Bt+AA(:,:,i0,k0)*x+(A(:,:,i0,k0)*B(i0,k0)');
        end
        Bt2=Bt-AA(:,:,k0,k0)*x-(A(:,:,k0,k0)*B(k0,k0)');
        tmp1=sum(tmp(:,k0));
        tmp2=tmp1-tmp(k0,k0);
        vA(:,:,k0)=2*omega(k0)*(Bt./(tmp1+1)-Bt2./(tmp2+1)); % vA: (N,1,K), computed euclidean-grad for all the K users
    end
    
    % we choose the e_grad w.r.t. the user index that has the min rate ----- refer to Tao's interference nulling paper equation (39) 
    % yA=-sum(vA,3); % this is the e_grad w.r.t. sum rate objective
    yA=-1*vA(:,:,index);
end

