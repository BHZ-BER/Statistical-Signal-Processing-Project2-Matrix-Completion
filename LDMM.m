load data_train;
rand('seed',131313);
randn('seed',1313)
rand_num = rand(90000,1);
MSE = 100;
time = 0;
for fold = 1:10
    tic
    training_set = data_train(find((rand_num<=0.1*(fold-1)) | (rand_num>=0.1*fold)),:);
    testing_set = data_train(find((rand_num>0.1*(fold-1)) & (rand_num<0.1*fold)),:);
    R = zeros(1682, 943);
    v = zeros(1682, 943);
    lambda = 0.3;
    R(sub2ind(size(R),training_set(:,2),training_set(:,1)))=training_set(:,3);
    Index = (R~=0);
    uw = R;
    [r,k]=size(uw);
    outerloop = 6;
    Test_R = zeros(1682, 943);
    Test_R(sub2ind(size(Test_R),testing_set(:,2),testing_set(:,1)))=testing_set(:,3);
    Test_W = (Test_R ~= 0);
    uw(~Index)=mean(uw(Index))+std(uw(Index))*randn(sum(sum(Index==0)),1);
    for ii=1:outerloop   
        
        W = weight_ann(uw');

    %   Assemble the coefficient matrix

        DW =diag(sum(W,2));
        W1=DW-W;
        coe_matrix=full(W1+lambda*W);

        for kk=1:10 % Bregman iteration, usually the number of iterations is set to be 1

            uw(Index)=R(Index); % assign the value at sample points to be the given value

            b=lambda*W*(uw-v);
            
            [uw,flag,relres]=gmresm(coe_matrix,b,[],1e-2,50,[],[],uw); % solving the linear system using GMRES


            uw_old=uw;
            v=v+uw_old-uw; % update the Lagrange multiplier

        end
        recovered = uw;
    end
    
    recovered = uw;
    time = time + toc;
    Test_R = zeros(1682, 943);
    Test_R(sub2ind(size(Test_R),testing_set(:,2),testing_set(:,1)))=testing_set(:,3);
    Test_W = (Test_R ~= 0);
    MSE_new = sum(sum(Test_W.*(Test_R - recovered).^2)/sum(sum(Test_W)))
    if MSE_new < MSE
        MSE = MSE_new
        X=recovered;
        save('LDMMOut.mat','X');
    end
end
MSE
time = time / 10