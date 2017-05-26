load data_train;
rand('seed',131313);
rand_num = rand(90000,1);
MSE = 100;
time = 0;
for fold = 1:10
    tic
    training_set = data_train(find((rand_num<=0.1*(fold-1)) | (rand_num>=0.1*fold)),:);
    testing_set = data_train(find((rand_num>0.1*(fold-1)) & (rand_num<0.1*fold)),:);
    m = 943;
    n = 1682;
    R = zeros(943, 1682);
    R(sub2ind(size(R),training_set(:,1),training_set(:,2)))=training_set(:,3);
    W = (R~=0);
    k = 9;
    [Uh, ss, V] = svds(R,k);
    V = V*sqrt(ss);
    Uh = Uh*sqrt(ss);
    % Optimization
    lambda = 15; 
    maxit = 300;
    %step_size = 0.001;
    thr = 5;
    err = inf; err_inV = inf; err_hap = inf;
    err_hist = zeros(1,maxit);
    iter=0; tic
    while iter < maxit && err > thr && err_inV > thr
        iter = iter + 1;
        for i =1:m
            Uh(i,:) = pinv(V'*diag(W(i,:))*V+lambda*eye(k))*(V'*diag(W(i,:))*R(i,:)');
        end
        for j =1:n
            V(j,:) = pinv(Uh'*diag(W(:,j))*Uh+lambda*eye(k))*(Uh'*diag(W(:,j))*R(:,j));%pinv((Uh'*diag(W(:,j))*R(:,j))\(Uh'*diag(W(:,j))*Uh+lambda*eye(k)),0);
        end    
        err = norm((R-Uh*V').*W,'fro');
        err_hist(iter) = err;
        if iter > 1
            err_inV = abs(err_hist(iter) - err_hist(iter-1));
        end
    end
    recovered = Uh*V';
    time = time + toc;
    Test_R = zeros(943, 1682);
    Test_R(sub2ind(size(Test_R),testing_set(:,1),testing_set(:,2)))=testing_set(:,3);
    Test_W = (Test_R ~= 0);
    MSE_new = sum(sum(Test_W.*(Test_R - recovered).^2)/sum(sum(Test_W)))
    if MSE_new < MSE
        MSE = MSE_new
    %MSE = sum(sum(Test_W.*abs(Test_R - recovered))/sum(sum(Test_W)))
        X=recovered;
        save('ALSOut.mat','X');
    end
end
time = time / 10;

