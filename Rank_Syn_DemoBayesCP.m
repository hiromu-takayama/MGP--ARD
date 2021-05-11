% A demo of Bayesian CP Factorization on synthetic data
% Written by Hiromu Takayama


total_plot=50;

plot_list=cell(total_plot,2);
plot_list_An=cell(total_plot,2);
plot_list_tau=cell(total_plot,2);
RankEst = zeros(total_plot,4);
RSE_list_list = cell(total_plot,1);

rng('shuffle','philox')
s = rng;

obs_rate_list=[15];

rate_N=length(obs_rate_list);
rate_iter=1;

total_list=cell(rate_N,1);
total_list_ave=cell(rate_N,1);

a_del0_list=[0.8];

for a_del0=a_del0_list

    for obs_rate=obs_rate_list

        for iter=1:total_plot

            close all;
            %randn('state',1); rand('state',1); %#ok<RAND>

            %% Generate a low-rank tensor
            DIM = [30,30,30];     % Dimensions of data
            R = obs_rate;          % True CP rank
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

            figure;

            answer=0;
            for n=1:length(DIM)
                subplot(length(DIM),1,n); heatmap(Z{n},'GridVisible','off');
                answer = answer + diag(Z{n}'*Z{n});
            end

            % Generate tensor by factor matrices
            X = double(ktensor_next(Z,DIM));

            %% Random missing values
            ObsRatio = 0.5;            % Observation rate: [0 ~ 1]
            Omega = randperm(prod(DIM)); 
            Omega = Omega(1:round(ObsRatio*prod(DIM)));
            O = zeros(DIM); 
            O(Omega) = 1;

            %% Add noise
            SNR = 20;                     % Noise levels
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
            nd=1;

            %% MGP-ARD-BCPF-TC
            tStart = tic;
            fprintf('------Bayesian CP with Mixture Priors for Image Completion---------- \n');
            [model] = MGP_BCPF_TC(Y, a_del0, 'obs', O, 'init', 'ml', 'maxRank', 2*R, 'dimRed', 1, 'tol', 1e-4, 'maxiters', 100, 'verbose', 2);
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

            %illigal
            RankEst(iter,2) = RRSElist(4);

            RankEst(iter,4) = model.TrueRank;
            TimeCost(4) = toc(tStart);


            if length(model.gammas) < 3
                model.gammas=model.gammas';
                model.gammas(3)=max(model.gammas);
                plot_list{iter,2}=sort(model.gammas,'descend');
                model.An=model.An';
                model.An(3)=0;  %max(model.gammas);
                plot_list_An{iter,2}=sort(model.An,'descend');
            else
                plot_list{iter,2}=sort(model.gammas(1:3),'descend')';
                plot_list_An{iter,2}=sort(model.An(1:3),'descend')';
            end

            RSE_list_list{iter}=RRSElist;

            %% Visualization of data and results
            %plotYXS(Y, X_FBCPS);
            %factorCorr = plotFactor(Z,model.X.U);

            %RankEst
            %RSElist
            %RMSElist
            %RRSElist
            %TimeCost
            
            RankEst(iter,:)
        
            if RankEst(iter,1)>RankEst(iter,2) && RankEst(iter,4)==5
                RankEst(iter,1)
                RankEst(iter,2)
                pause
            end


        end

        %{

        plot_list_all=cat(1,plot_list{:});
        figure;
        plot3(plot_list_all(1:total_plot,1),plot_list_all(1:total_plot,2),plot_list_all(1:total_plot,3),'o')
        hold on
        plot3(plot_list_all(total_plot+1:2*total_plot,1),plot_list_all(total_plot+1:2*total_plot,2),plot_list_all(total_plot+1:2*total_plot,3),'+')
        legend('ARD','ARD-MGP')
        xlabel('lambda_1')
        ylabel('lambda_2')
        zlabel('lambda_3')
        grid on

        plot_list_all_An=cat(1,plot_list_An{:});
        figure;
        plot3(answer(1),answer(2),answer(3),'*')
        hold on
        plot3(plot_list_all_An(1:total_plot,1),plot_list_all_An(1:total_plot,2),plot_list_all_An(1:total_plot,3),'o')
        hold on
        plot3(plot_list_all_An(total_plot+1:2*total_plot,1),plot_list_all_An(total_plot+1:2*total_plot,2),plot_list_all_An(total_plot+1:2*total_plot,3),'+')
        legend('True','ARD','ARD-MGP')
        xlabel('lambda_1')
        ylabel('lambda_2')
        zlabel('lambda_3')
        grid on

        %}

        list_1=zeros(total_plot,1);
        list_2=zeros(total_plot,1);

        for i=1:total_plot
        temp=RSE_list_list{i};
        list_1(i)=temp(3);
        list_2(i)=temp(4);
        end

        total_list{rate_iter}=RankEst;
        total_list_ave{rate_iter}=[mean(list_1),mean(list_2)];
        
        rate_iter=rate_iter+1;
    end
    
    str=string(a_del0);
    str2=string(obs_rate);
    
    csvwrite('my'+str2+'File'+str+'.csv',[RankEst(:,2) RankEst(:,4)]);
    
end
