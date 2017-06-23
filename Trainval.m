load data_train;
load AGDOut
R = zeros(1682, 943);
R(sub2ind(size(R),data_train(:,2),data_train(:,1)))=data_train(:,3);
W = (R~=0);
MSE_overall = sum(sum(W.*(R - X).^2)/sum(sum(W)))
