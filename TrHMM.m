function [AccTrHMM] = TrHMM()
% Input: Trace data, loaded from WiFiTimeData10.mat
%   e.g. (1) 'd0826' denotes the data collected at 08:26, with rows as examples,
%            first 10 columns as features (signal strengths from 10 access
%            points), the 11th column as the example's corresponding grid index.
%        (2) 'trn0826' denotes the training traces that we designed for our
%            experiments. Each cell is a trace, with rows as features,
%            columns as trace length (or, # of grids travelled).
%            ''trn0826label' denotes the corresponding labels for the
%            'trn0826' traces. Similarly, we have 'trn1112/trn1354/trn1621/trn910'.
%        (3) 'tst1112' denotes the testing traces, and 'tst1112label' denotes
%            the corresponding labels. Similarly, we have 'tst1354/tst1621/tst910'.
%        (4) 'hallway' denotes which grids are covered at each hallway.
%        (5) 'location' denotes the 2-D physical coordinates for all the grids.
%        (6) 'nref' denotes the number of reference points (RPs) at each hallway.
%            Row i denotes that, if we select (i*10) percents of the grids
%            as RPs, how many grids at each hallway need to be selected.
%            Here, we try to select RPs evenly at each hallway, which is a
%            manual way to do RP placement.
%        (7) 'nex' denotes the number of examples collected at each grid for
%            each time period. Similarly, 'ntrn' denotes the number of
%            examples out of all examples are used as training data; and
%            'ntst' denotes as the number of examples as testing data.

% Output: AccTrHMM - Localization accuracies for different number of RPs

% Note: This code uses the HMM toolbox from Kevin Murphy.
%       See http://www.cs.ubc.ca/~murphyk/Software/HMM/hmm.html.

% Copyright by Vincent W. Zheng (http://www.cse.ust.hk/~vincentz).
% Any question, please send email to vincentz_AT_cse.ust.hk.
% October 19th, 2008.
% ===============================================================

load WiFiTimeData10.mat;

% Define the output
AccTrHMM = zeros(10,1);

% Set the number of features O, and the number of states Q
O = 10;
% Grid 1 and grid 119 are referring to a same grid. So totally 118 different grids.
Q = 119;

% Set the number of unlabeled traces used at time t.
ntrace = 15;

% When the number of reference points are varying from 10% to 90% by 10%,
% of all the grids.
for r=1:9
    % ----------------------------------------------
    % Step 1. Applying regression analysis at time 0.

    % Randomly select Reference points and Non-reference points
    refpt = cell(5,1);
    for i=1:5
        a = randperm(length(hallway{i}));
        refpt{i} = hallway{i}(a(1:nref(r,i))); % employ (10*r)% locations
    end
    nonrefpt = cell(5,1);
    index = zeros(Q,1);
    for i=1:5
        index(refpt{i})=1;
    end
    for i=1:5
        s = length(hallway{i});
        for j=1:s
            if index(hallway{i}(j))==0
                nonrefpt{i} = [nonrefpt{i} hallway{i}(j)];
            end
        end
    end

    % Regression on time 0. We use 08:26 data as time 0 through the paper.
    W = RefptRegress(d0826, refpt, nonrefpt, 'linear');

    % ----------------------------------------------
    % Step 2. Rebuilding the radio map at time t.
    % We show using 16:21 data as time t here. One can change to d1121/d1354/d1910.
    Regd1621 = RecoverMatrix(d1621, W, refpt, nonrefpt);

    % ----------------------------------------------
    % Step 3. Using EM on unlabeled traces at time t.

    % Calculate HMM parameters
    %% build base model on 08:26
    %% give prior knowledge about hidden states: prior0
    %% transition matrix between hidden states: transmat0
    %% construct gauss hmm for output observation p(o|q): mu0, Sigma0
    M = 1; % 1 mixture components = Gaussian Output
    mu0826 = zeros(O,Q,M);
    Sigma0826 = zeros(O,O,Q,M);

    [prior0826, transmat0826, mu0, Sigma0] = ...
        gausshmm_train_observed(trn0826, trn0826label, Q, 'cov_type', 'diag');

    prior0826 = ones(Q,1)*(1/Q);
    mu0826(:,:,M) = mu0;
    Sigma0826(:,:,:,M) = Sigma0;
    mixmat0826 = ones(Q,M);

    % Step 3.1. Update HMM parameter at time t by regression analysis.
    [mu0 Sigma0 mixmat0] = RecoverParameter(Regd1621, mu0826, Sigma0826, mixmat0826);
    prior0 = prior0826;
    transmat0 = transmat0826;

    % Step 3.2. Further update HMM parameter by EM through time t trace data.
    a = randperm(100);
    index = a(1:ntrace); % randomly select 'ntrace' traces for EM
    [prior1 transmat1 mixmat1 mu1 Sigma1] ...
        = MAPadapt(trn1621, index, prior0, transmat0, mu0, Sigma0, mixmat0);

    % Use Viterbi algorithm to infer the trace labels.
    [AccTrMM(r) accrec1621] = TestViterbi(tst1621, tst1621label, prior1,...
        transmat1, mu1, Sigma1, mixmat1, location);
end

% When the number of reference points are 100% of all the grids.
r = 10;

M = 1;
mu1621 = zeros(O,Q,M);
Sigma1621 = zeros(O,O,Q,M);

% Directly calculate the HMM parameters, since time t data are all known.
[prior1621, transmat1621, mu0, Sigma0] = ...
    gausshmm_train_observed(trn1621, trn1621label, Q, 'cov_type', 'diag');

prior1621 = ones(Q,1)*(1/Q);
mu1621(:,:,M) = mu0;
Sigma1621(:,:,:,M) = Sigma0;
mixmat1621 = ones(Q,M);

a = randperm(100);
index = a(1:ntrace); % randomly select 'ntrace' traces for EM
[prior1 transmat1 mixmat1 mu1 Sigma1] ...
    = MAPadapt(trn1621, index, prior1621, transmat1621, mu1621, Sigma1621, mixmat1621);

% Use Viterbi algorithm to infer the trace labels.
[AccTrHMM(r) accrec1621] = TestViterbi(tst1621, tst1621label, prior1,...
    transmat1, mu1, Sigma1, mixmat1, location);

end


% ===============================================================
function [W] = RefptRegress(data, refpt, nonrefpt, method)
% Input: data - a Num*Fea matrix
%        refpt, nonrefpt - indices of the RPs and non-PRs
%        method - 'linear'/'quadratic'/'cubic' for different order regression
% Output: W - the regression weight matrix

ntrn = 40;
O = 10;
Q = 119;

% each hallway has its regression function, i.e. weight matrix
W = cell(O,5);

switch method
    case 'linear'
        for k=1:5
            s1 = length(refpt{k});
            s2 = length(nonrefpt{k});

            if s2~=0
                % for each AP, each hallway, XW = Y, X(ntrn*s1), Y(ntrn*s2)
                for i=1:O
                    X = [];
                    for j=1:s1
                        index = find(data(:,end)==refpt{k}(j));
                        X = [X data(index(1:ntrn),i)];
                    end
                    Y = [];
                    for j=1:s2
                        index = find(data(:,end)==nonrefpt{k}(j));
                        Y = [Y data(index(1:ntrn),i)];
                    end
                    W{i,k} = pinv(X'*X)*X'*Y;
                end
            end
        end
    case 'quadratic'
        for k=1:5
            s1 = length(refpt{k});
            s2 = length(nonrefpt{k});
            % for each AP, each hallway, XW = Y, X(ntrn*s1), Y(ntrn*s2)
            for i=1:O
                X = [];
                for j=1:s1
                    index = find(data(:,end)==refpt{k}(j));
                    X = [X data(index(1:ntrn),i).^2 data(index(1:ntrn),i)];
                end
                Y = [];
                for j=1:s2
                    index = find(data(:,end)==nonrefpt{k}(j));
                    Y = [Y data(index(1:ntrn),i)];
                end
                W{i,k} = pinv(X'*X)*X'*Y;
            end
        end
    case 'cubic'
        for k=1:5
            s1 = length(refpt{k});
            s2 = length(nonrefpt{k});
            % for each AP, each hallway, XW = Y, X(ntrn*s1), Y(ntrn*s2)
            for i=1:O
                X = [];
                for j=1:s1
                    index = find(data(:,end)==refpt{k}(j));
                    X = [X data(index(1:ntrn),i).^3 data(index(1:ntrn),i).^2 data(index(1:ntrn),i)];
                end
                Y = [];
                for j=1:s2
                    index = find(data(:,end)==nonrefpt{k}(j));
                    Y = [Y data(index(1:ntrn),i)];
                end
                W{i,k} = pinv(X'*X)*X'*Y;
            end
        end
    otherwise
        disp('Unknown method.');
end
end

% ===============================================================
function [NewData] = RecoverMatrix(data, W, refpt, nonrefpt)
% Input: data - time t's data, a Num*Fea matrix
%        W - regression weight learned from time 0
%        refpt, nonrefpt - indices of RPs and non-RPs
% Output: NewData - updated data matrix (updating the nonrefpt's data)

O = 10;
Q = 119;
ntrn = 40;

NewData = data;
for k=1:5
    % for each AP, Y = XW, X(ntrn*s1), Y(ntrn*s2)
    X = cell(O,1);
    Y = cell(O,1);
    s1 = length(refpt{k});
    s2 = length(nonrefpt{k});
    if s2~=0
        for i=1:O
            X{i} = [];
            for j=1:s1
                index = find(data(:,end)==refpt{k}(j));
                X{i} = [X{i} data(index(1:ntrn),i)];
            end
            Y{i} = X{i} * W{i,k};
        end
    end
    % re-construct GMM for new time slice
    if s2~=0
        for i=1:s2
            temp = [];
            for j=1:O
                temp = [temp Y{j}(:,i)]; % temp is now ntrn*O
            end
            index = find(data(:,end)==nonrefpt{k}(i));
            NewData(index(1:ntrn),1:O) = temp;
        end
    end
end
end

% ===============================================================
function [mu_hat Sigma_hat mixmat_hat] = RecoverParameter(data, pre_mu, ...
    pre_Sigma, pre_mixmat)
% Input: data - updated data matrix after regression at time t
%        pre_mu, pre_Sigma, pre_mixmat - HMM parameters for time 0
% Output: mu_hat Sigma_hat mixmat_hat - updated HMM parameters for time t

beta = 0.4;
O = 10;
Q = 119;
ntrn = 40;

M = 1; % 1 mixture components = Gaussian Output
mu0 = zeros(O,Q,M);
Sigma0 = zeros(O,O,Q,M);

% for each feature dimension, re-fit the gaussian distribution
for i=1:Q
    index = find(data(:,end)==i);
    temp = data(index(1:ntrn),1:O);
    [mm,ss] = mixgauss_init(M,temp','diag');
    mu0(:,i,:) = mm;
    Sigma0(:,:,i,:) = ss;
end

mu_old = pre_mu;
mu_new = mu0;
Sigma_old = pre_Sigma;
Sigma_new = Sigma0;
mixmat_old = pre_mixmat;

mu_hat = zeros(O,Q,M);
Sigma_hat = zeros(O,O,Q,M);
mixmat_hat = zeros(Q,M);

for j=1:M
    mu_hat(:,:,j) = mu_new;
    Sigma_hat(:,:,:,j) = Sigma_new;
    mixmat_hat(:,j) = mixmat_old;
end

end


% ===============================================================
function [prior1 transmat1 mixmat1 mu1 Sigma1] ...
    = MAPadapt(trn, index, prior0, transmat0, mu0, Sigma0, mixmat0)
% Input: trn - training traces
%        index - indices of the selected traces
%        prior0/transmat0/mu0/Sigma0/mixmat0 - HMM parameters before update
% Outpu: prior1/transmat1/mixmat1/mu1/Sigma1 - HMM parameters after update

[LL, prior1, transmat1, mu1, Sigma1, mixmat1]...
    = mhmm_em(trn(index), prior0, transmat0, mu0, Sigma0, mixmat0,...
    'max_iter', 5, 'cov_type', 'diag', 'adj_prior', 0);

end


% ===============================================================
function [accuracy tst_path] = TestViterbi(tst, tst_label, prior1, ...
    transmat1, mu1, Sigma1, mixmat1, location)
% Input: tst/tstlabel - testing traces and their labels
%        prior1/transmat1/mu1/Sigma1/mixmat1 - HMM parameters
%        location - 2-D coordinates for all the grids
% Output: accracy - localization accuracy under some error distance
%         tst_path - output labels for test paths.

% Use Viterbi algorithm to predict labels for the test paths
n = 20;
tst_path = cell(n,1);
for i=1:n
    B = mixgauss_prob(tst{i}, mu1, Sigma1, mixmat1);
    path = viterbi_path(prior1, transmat1, B);
    tst_path{i} = path; % each row of "tst_path" is an output label
end

% Calculate the localizaiton accuracy under 3-meter error distance
errdist = 3;
ncorrect = 0;
ntotal = 0;
for i=1:n
    gndtruth = location(tst_label{i},:);
    ntotal = ntotal + size(gndtruth,1);
    predict = location(tst_path{i},:);
    temp = sum((predict-gndtruth).^2,2);
    ncorrect = ncorrect + length(find(temp<(errdist^2)));
end
accuracy = ncorrect/ntotal;
end
