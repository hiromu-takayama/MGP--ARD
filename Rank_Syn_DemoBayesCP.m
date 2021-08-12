% A demo of Bayesian CP Factorization on synthetic data
% Written by Hiromu Takayama
%
% In this demo, we provide two Bayesian CP Factorization algorithms: one for
% incomplete tensor and tensor completion ("BCPF_TC.m") and the other one for
% complete tensor, which is a more efficient implementation when tensor is 
% fully observed ("BCPF.m").  
% The parameter settings are optional, please refer to the help by command 
% >> help BCPF_TC  
% Two important settings are initialization method and initilization of maximum rank.
% 
% This demo is used for evaluation of CP Rank determination and predictive
% perofrmance under varying conditions including Tensor size, True rank, 
% Signal type, Observation ratio, and Noise SNRs.   

% After the model learning, the results can be shown by
% 1. Visualization of true latent factors and the estimated factors
% 2. Visualization of observed tensor Y,  the estimated low-CP-rank tensor X

close all;
randn('state',1); rand('state',1); %#ok<RAND>
%% Generate a low-rank tensor
DIM = [256,256,3];     % Dimensions of data
R = 3;                % True CP rank
DataType = 2;         % 1: Random factors   2: The deterministic factors (sin, cos, square)

Z = cell(length(DIM),1);   
if DataType ==1
    for m=1:length(DIM)
          Z{m} =  gaussSample(zeros(R,1), eye(R), DIM(m));  
    end
end
if DataType == 2
    for m=1:length(DIM)
        temp = linspace(0, m*2*pi, DIM(m));
        part1 = [sin(temp); cos(temp); square(linspace(0, 16*pi, DIM(m)))]';
        part2 = gaussSample(zeros(DIM(m),1), eye(DIM(m)), R-size(part1,2))';
        Z{m} = [part1 part2];
        Z{m} = Z{m}(:,1:R);
        Z{m} = zscore(Z{m});
    end
end

% Generate tensor by factor matrices
X = double(ktensor_next(Z,DIM));

%% Random missing values
ObsRatio = 0.2;               % Observation rate: [0 ~ 1]
Omega = randperm(prod(DIM)); 
Omega = Omega(1:round(ObsRatio*prod(DIM)));
O = zeros(DIM); 
O(Omega) = 1;

%% Add noise
SNR = 10;                     % Noise levels
sigma2 = var(X(:))*(1/(10^(SNR/10)));
GN = sqrt(sigma2)*randn(DIM);

%% Generate observation tensor Y
Y = X + GN;
Y = O.*Y;

%% Run BayesCP
fprintf('------Bayesian CP Factorization---------- \n');


% Initialization
TimeCost = zeros(4,1);
RSElist = zeros(4,3);
RMSElist = zeros(4,1);
RRSElist = zeros(4,1);
RankEst = zeros(4,1);
nd=1;

params.R = 20;

%{

%% MGP-t for natural images
tStart = tic;
params.a = 2;
params.binary = 0;
params.tau_eps = 1;
params.normalize = 1;
params.maxiters = 100;
params.burnin = 80;
[data_train data_test subs_train subs_test] = generator_data(X,Y);
[U lambda prob_avg recover] = mu_mgpcp_gibbs_cp_t(double(data_train),subs_train,double(data_test),subs_test,params);
X_FBCPS = double(recover);

RSElist(1,1) = perfscore(X_FBCPS, X);
RSElist(1,2) = perfscore(X_FBCPS(O==1), X(O==1));
RSElist(1,3) = perfscore(X_FBCPS(O==0), X(O==0));

X_FBCPS(O==1) = X(O==1);
err = X_FBCPS(:) - X(:);
RMSElist(1) = sqrt(mean(err.^2));
RRSElist(1) = sqrt(sum(err.^2)/sum(X(:).^2));
RankEst(1) = params.R;
TimeCost(1) = toc(tStart);

%% Visualization of data and results
plotYXS(Y, X_FBCPS);
%factorCorr = plotFactor(Z,model.X.U);


%% MGP-a for natural images
tStart = tic;
params.R = 1;
[U lambda prob_avg recover R] = mu_mgpcp_gibbs_cp_a(double(data_train),subs_train,double(data_test),subs_test,params);
X_FBCPS = double(recover);

RSElist(2,1) = perfscore(X_FBCPS, X);
RSElist(2,2) = perfscore(X_FBCPS(O==1), X(O==1));
RSElist(2,3) = perfscore(X_FBCPS(O==0), X(O==0));

X_FBCPS(O==1) = X(O==1);
err = X_FBCPS(:) - X(:);
RMSElist(2) = sqrt(mean(err.^2));
RRSElist(2) = sqrt(sum(err.^2)/sum(X(:).^2));
RankEst(2) = params.R;
TimeCost(2) = toc(tStart);

%% Visualization of data and results
plotYXS(Y, X_FBCPS);
%factorCorr = plotFactor(Z,model.X.U);


%}

%% ARD-BCPF-MP (mixture priors) for natural images
tStart = tic;
fprintf('------Bayesian CP with Mixture Priors for Image Completion---------- \n');
[model] = BCPF_TC(Y, 'obs', O, 'init', 'ml', 'maxRank', 2*R, 'dimRed', 1, 'tol', 1e-4, 'maxiters', 60, 'verbose', 2);
%max([DIM 2*R])
X_FBCPS = double(model.X);

RSElist(3,1) = perfscore(X_FBCPS, X);
RSElist(3,2) = perfscore(X_FBCPS(O==1), X(O==1));
RSElist(3,3) = perfscore(X_FBCPS(O==0), X(O==0));

X_FBCPS(O==1) = X(O==1);
err = X_FBCPS(:) - X(:);
RMSElist(3) = sqrt(mean(err.^2));
RRSElist(3) = sqrt(sum(err.^2)/sum(X(:).^2));
RankEst(3) = model.TrueRank;
TimeCost(3) = toc(tStart);

%% Visualization of data and results
plotYXS(Y, X_FBCPS);
%factorCorr = plotFactor(Z,model.X.U);


%% MGP-ARD-BCPF-TC
tStart = tic;
fprintf('------Bayesian CP with Mixture Priors for Image Completion---------- \n');
[model] = MGP_BCPF_TC(Y, 'obs', O, 'init', 'ml', 'maxRank', 2*R, 'dimRed', 1, 'tol', 1e-4, 'maxiters', 60, 'verbose', 2);
%1e-4

%max([DIM 2*R])
X_FBCPS = double(model.X);

RSElist(4,1) = perfscore(X_FBCPS, X);
RSElist(4,2) = perfscore(X_FBCPS(O==1), X(O==1));
RSElist(4,3) = perfscore(X_FBCPS(O==0), X(O==0));

X_FBCPS(O==1) = X(O==1);
err = X_FBCPS(:) - X(:);
RMSElist(4) = sqrt(mean(err.^2));
RRSElist(4) = sqrt(sum(err.^2)/sum(X(:).^2));
RankEst(4) = model.TrueRank;
TimeCost(4) = toc(tStart);

%% Visualization of data and results
plotYXS(Y, X_FBCPS);
%factorCorr = plotFactor(Z,model.X.U);

RankEst
RSElist
RMSElist
RRSElist
TimeCost

