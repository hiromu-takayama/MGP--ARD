
% A demo of Bayesian CP factorization for image completion
% Written by Hiromu Takayama
%
% In this demo, we provide two algorithms including BCPF_IC and BCPF_MP.
% BCPF_IC is a Bayesian CP for image completion; BCPF_MP is a Bayesian CP
% using mixture priors, which is particularly useful for natural image
% completion. For algorithm settings, please refer to the detailed help by 
% >> help BCPF_MP

% The experimental data can be tested with
% 1) Different image files
% 2) Observation rate (1-missing rate)
% The predictive image can be online visualized during model learning. 
% The performance of RSE, PSNR, SSIM, Time Cost are evaluated and reported.

close all; clear all;

image_list={'lena','baboon','sailboat','airplane','barbara','facade','house','peppers'};

SNR=10;
count=1;

h_1=figure();
h_1.Position=[2,2,3000,500];
h_2=figure();
h_2.Position=[2,2,3000,500];

for name=image_list
    
    name=name{1};
    randn('state',1); rand('state',1); %#ok<RAND>
    %% Load image data
    filename=strcat('./TestImages/',name,'.bmp');    % Image file
    ObsRatio = 0.1;                      % Observation rate

    X = double(imread(filename));
    DIM = size(X);

    Omega = randperm(prod(DIM));
    Omega = Omega(1:round(ObsRatio*prod(DIM)));
    O = zeros(DIM);
    O(Omega) = 1;
    
    sigma2 = var(X(:))*(1/(10^(SNR/10)));
    GN = sqrt(sigma2)*randn(DIM);
    
    Y_N=X+GN;
    Y=O.*Y_N;

    % plot images
    row =3; col =8;
    set(0,'CurrentFigure',h_1);
    figure(h_1);
    subplot(3,8,0+count);
    imshow(uint8(X));
    subplot(3,8,8+count);
    imshow(uint8(Y_N));
    subplot(3,8,16+count);
    imshow(uint8(Y));
    drawnow;

    % Initialization
    TimeCost = zeros(4,1);
    RSElist = zeros(4,3);
    PSNRlist = zeros(4,1);
    SSIMlist = zeros(4,1);
    RankEst = zeros(4,1);

    if ~isempty(strfind(filename,'fecade.bmp'))
        nd=0.1;    % low-rank structural images
    else
        nd=1;      % natural images
    end
    
    %{

    %% MGP-t for natural images
    tStart = tic;
    params.R = 50;
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
    PSNRlist(1) = PSNR_RGB(X_FBCPS,X);
    SSIMlist(1) = ssim_index(rgb2gray(uint8(X_FBCPS)),rgb2gray(uint8(X)));
    RankEst(1) = params.R;
    TimeCost(1) = toc(tStart);
    figure; imshow(uint8(X_FBCPS)); title('FBCP-MP-t','FontWeight','bold'); drawnow;


    %% MGP-a for natural images
    tStart = tic;
    params.R = 1;
    [U lambda prob_avg recover R] = mu_mgpcp_gibbs_cp_a(double(data_train),subs_train,double(data_test),subs_test,params);
    X_FBCPS = double(recover);

    RSElist(2,1) = perfscore(X_FBCPS, X);
    RSElist(2,2) = perfscore(X_FBCPS(O==1), X(O==1));
    RSElist(2,3) = perfscore(X_FBCPS(O==0), X(O==0));

    X_FBCPS(O==1) = X(O==1);
    PSNRlist(2) = PSNR_RGB(X_FBCPS,X);
    SSIMlist(2) = ssim_index(rgb2gray(uint8(X_FBCPS)),rgb2gray(uint8(X)));
    RankEst(2) = R;
    TimeCost(2) = toc(tStart);
    figure; imshow(uint8(X_FBCPS)); title('FBCP-MP-a','FontWeight','bold'); drawnow;

    %}
    
    %% ARD-BCPF-MP (mixture priors) for natural images
    tStart = tic;
    fprintf('------Bayesian CP with Mixture Priors for Image Completion---------- \n');
    [model] = BCPF_MP(Y, 'obs', O, 'init', 'ml', 'maxRank', 100, 'maxiters', 30, ...
        'tol', 1e-4, 'dimRed', 1, 'verbose', 2, 'nd', nd);
    X_FBCPS = double(model.X);

    RSElist(3,1) = perfscore(X_FBCPS, X);
    RSElist(3,2) = perfscore(X_FBCPS(O==1), X(O==1));
    RSElist(3,3) = perfscore(X_FBCPS(O==0), X(O==0));

    X_FBCPS(O==1) = X(O==1);
    PSNRlist(3) = PSNR_RGB(X_FBCPS,X);
    SSIMlist(3) = ssim_index(rgb2gray(uint8(X_FBCPS)),rgb2gray(uint8(X)));
    RankEst(3) = model.TrueRank;
    TimeCost(3) = toc(tStart);
    pause(0.1)
    figure(h_2);
    subplot(2,8,0+count); imshow(uint8(X_FBCPS)); title('ARD','FontWeight','bold'); drawnow;

    
    %% MGP-ARD-BCPF-MP (mixture priors) for natural images
    tStart = tic;
    fprintf('------Bayesian CP with Mixture Priors for Image Completion---------- \n');
    [model] = MGP_BCPF_MP(Y, 'obs', O, 'init', 'rand', 'maxRank', 100, 'maxiters', 30, ...
        'tol', 1e-4, 'dimRed', 1, 'verbose', 2, 'nd', nd);
    X_FBCPS = double(model.X);

    RSElist(4,1) = perfscore(X_FBCPS, X);
    RSElist(4,2) = perfscore(X_FBCPS(O==1), X(O==1));
    RSElist(4,3) = perfscore(X_FBCPS(O==0), X(O==0));

    X_FBCPS(O==1) = X(O==1);
    PSNRlist(4) = PSNR_RGB(X_FBCPS,X);
    SSIMlist(4) = ssim_index(rgb2gray(uint8(X_FBCPS)),rgb2gray(uint8(X)));
    RankEst(4) = model.TrueRank;
    TimeCost(4) = toc(tStart);
    pause(0.1)
    figure(h_2);
    subplot(2,8,8+count); imshow(uint8(X_FBCPS)); title('MGP-ARD','FontWeight','bold'); drawnow;

    %%
    RankEst
    RSElist
    PSNRlist
    SSIMlist
    TimeCost
    
    count=count+1;

end

