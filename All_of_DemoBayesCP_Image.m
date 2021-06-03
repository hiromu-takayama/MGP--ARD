% Written by Hiromu Takayama

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

