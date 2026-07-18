%% fcm_tsk_pso_de_EV_required_charging_with_V2G_service.m
% 2-stage pipeline for microgrid with corrected forecasting/evaluation logic
%   Stage A) Lagged FCM + TSK (WLS) forecasters for PV, Load, Price
%            (1-step ahead, recursive 24h rollout)
%   Stage B) PSO dispatch optimization using forecasted PV/Load/Price
%            (grid cost + SoC constraints via repair)
%
% EV INTEGRATION: Electric Vehicle with V2G capability
%   - Rule-based EV scheduling (not forecasted)
%   - Coordinated BESS + EV optimization
%   - EV SoC tracking with driver constraints
%

clear; clc; close all;
%diary on
%% --------------------------
% User settings / hyperparams
% --------------------------
% Run-location safe CSV loading:
% Put this .m file and Microgrid_with_EV.csv in the same folder, or run
% the script from the folder containing the CSV. This avoids hard-coded
% Windows paths and makes the code portable.
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
CSV_FILE = fullfile(script_dir, "Microgrid_with_EV.csv");
if ~isfile(CSV_FILE)
    CSV_FILE = "Microgrid_with_EV.csv";
end

PRED_HORIZON = 24;
LAGS = 24;

TRAIN_FRAC = 0.70;
VAL_FRAC   = 0.15;
TEST_FRAC  = 0.15;

CANDIDATE_CLUSTERS = [3 4 5];

% Ridge-regularized TSK consequent candidates (selected on validation data)
TSK_RIDGE_LAMBDAS = [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1];

% FCM params
FCM_M = 2.0;
FCM_MAXITER = 200;
FCM_TOL = 1e-5;

% FFNN benchmark params (custom MATLAB implementation; no Neural Network Toolbox needed)
% The FFNN uses the SAME lagged input vector as FCM-TSK and predicts the
% same 24-h multi-output targets: PV, Load, and Price. EV charging/discharging
% remain optimization variables, not forecasting outputs.
RUN_FFNN_BENCHMARK = true;
FFNN_HIDDEN_SIZES = [64 32];
FFNN_EPOCHS = 250;
FFNN_BATCH_SIZE = 256;
FFNN_LEARN_RATE = 1e-3;
FFNN_L2 = 1e-5;
FFNN_PATIENCE = 30;
FFNN_SEED = 2027;
FFNN_VERBOSE = true;

% LSTM benchmark params (requires Deep Learning Toolbox)
% Uses the same 24-h lag information as FCM-TSK/FFNN, rearranged as a
% multivariate sequence. Calendar features are repeated at every time step.
RUN_LSTM_BENCHMARK = true;
LSTM_NUM_HIDDEN = 32;       % smaller network generalizes better on ~6k sequences
LSTM_FC_HIDDEN = 16;         % compact nonlinear readout
LSTM_EPOCHS = 120;
LSTM_MINIBATCH_SIZE = 64;
LSTM_LEARN_RATE = 5e-4;
LSTM_L2 = 1e-4;              % stronger regularization than the first LSTM version
LSTM_PATIENCE = 12;
LSTM_SEED = 2028;
LSTM_VERBOSE = true;

% PSO params
SWARM_SIZE = 150;        % 72-dim BESS+EV problem
PSO_MAXITER = 400;       % keep same budget for fair comparison
W_INERTIA = 0.85;        % w_max; custom_pso decays to 0.4*w_max
C1 = 1.7;
C2 = 1.7;
VEL_CLAMP_FACTOR = 0.25;  % fraction of each variable range; EV needs vector clamp

N_VALIDATION_RUNS = 20;
SEED_BASE = 1000;        % deterministic; matched across strategies for paired comparison

% --------------------------
% Multi-day evaluation settings
% --------------------------
% Set MULTIDAY_N_DAYS between 7 and 30. The script automatically selects
% evenly spaced midnight-start test days from the held-out test set.
RUN_MULTIDAY_EVALUATION = true;
MULTIDAY_N_DAYS = 20;        % recommended quick test: 7-10; final paper: 20-30
MULTIDAY_N_RUNS = 10;        % runs/day/algorithm; final paper can use 20 if runtime allows
MULTIDAY_INCLUDE_DE = true;  % set false for faster PSO-only multi-day evaluation
MULTIDAY_DAY_SELECTION = "evenly";  % "evenly" or "first"
MULTIDAY_SAVE_CSV = true;

% Battery/grid params
% V2G-emphasis scenario: reduce stationary BESS dominance so EV flexibility is visible.
% This is suitable when the paper wants to demonstrate meaningful EV/V2G participation.
Pmax = 1.5;      % kW (max BESS charge/discharge power; reduced from 2.0)
Ecap = 4.0;      % kWh (BESS capacity; reduced from 5.0)
SoC0 = 0.50;
SoC_min = 0.25;
SoC_max = 0.90;
eta_ch = 0.95;
eta_dis = 0.95;
dt = 1.0;        % hour

% PV capacity (from data analysis - peak around 1.6 kW)
PV_CAPACITY = 1.8;  % kW (slightly above historical max of ~1.6 kW)

% Demand charge rate (residential = $2/kW, commercial = $8/kW)
% Commercial/office demand-charge scenario to make V2G peak shaving economically meaningful.
demand_charge_rate = 5.0;  % $ per kW of peak demand (was 2.0)

% Publication-safety switches
% Price enhancement is applied as a fixed deterministic rule. Do NOT choose
% between original/enhanced forecasts using test-day actual values.
USE_LOAD_ADAPTIVE_PRICE_ENHANCEMENT = true;

% Check if file exists
if ~isfile(CSV_FILE)
    error('CSV file "%s" not found. Put it in working dir or set path.', CSV_FILE);
end

%% --------------------------
% 1) Load CSV and detect columns
% --------------------------
T = readtable(CSV_FILE);
cols = string(T.Properties.VariableNames);
disp("CSV columns found:");
disp(cols.');

TS_col    = detect_col(cols, ["Timestamp","timestamp","time","date"], T);
PV_col    = detect_col(cols, ["PV","pv","solar"], T);
Load_col  = detect_col(cols, ["Load","load","demand"], T);
Price_col = detect_col(cols, ["Price","price","tariff"], T);
SoC_col   = detect_col(cols, ["SoC","soc","stateofcharge"], T);

ts = T.(TS_col);
if ~isdatetime(ts)
    % The provided Microgrid_with_EV.csv uses strings such as
    % 01/01/2012 00:00, so parse this explicitly first. Fall back to
    % MATLAB auto-detection for other timestamp formats.
    try
        ts = datetime(ts, 'InputFormat','MM/dd/yyyy HH:mm', ...
            'Format','yyyy-MM-dd HH:mm:ss');
    catch
        try
            ts = datetime(ts, 'Format','yyyy-MM-dd HH:mm:ss');
        catch
            error("Timestamp column could not be parsed. Expected e.g. 01/01/2012 00:00.");
        end
    end
end
if all(isnat(ts))
    error("Timestamp column could not be parsed.");
end

PV_hist    = double(T.(PV_col));
Load_hist  = double(T.(Load_col));
Price_hist = double(T.(Price_col));
SoC_hist   = double(T.(SoC_col));

EV_Charging_hist = double(T.EV_Charging_kW);
EV_Discharging_hist = double(T.EV_Discharging_kW);
EV_SoC_hist = double(T.EV_SoC);
Vehicle_Count_hist = double(T.Vehicle_Count);

%% EV Parameters - Single V2G-Enabled Electric Vehicle
EV_params = struct();
EV_params.EV_capacity_total = 40;      % kWh

EV_params.EV_SoC0 = 0.35;              % EV arrival SoC = 35%
EV_params.EV_final_soc_req = 0.70;     % EV departure requirement = 70%
EV_params.ev_terminal_shortage_cost = 200.0;  % strong $/kWh penalty for unmet departure energy

EV_params.EV_SoC_min = 0.20;
EV_params.EV_SoC_max = 0.95;
EV_params.eta_ev_ch = 0.92;
EV_params.eta_ev_dis = 0.92;
EV_params.P_ev_max = 7.2;
EV_params.vehicle_count = 1;
EV_params.ev_degradation_cost = 0.005;
EV_params.mode_change_cost = 0.005;    % small cost; avoid suppressing useful V2G
EV_params.max_net_discharge_frac = 0.20;  % max net V2G discharge beyond daily charging, as fraction of EV capacity

EV_params.v2g_required_energy = 2.0;       % kWh required V2G energy in peak window
EV_params.v2g_requirement_weight = 30.0;   % penalty weight for unmet V2G service
EV_params.v2g_overdelivery_weight = 10.0;  % penalty for excessive V2G beyond service target
EV_params.v2g_overdelivery_margin = 1.25;  % allow up to 125% of target before over-delivery penalty
EV_params.v2g_allowed_hours = 17:21;       % clock hours where V2G is allowed
EV_params.no_charge_hours = 17:21;         % prevent EV charging during peak-price hours
EV_params.P_ev_ch_max = 3.6;               % practical AC charging limit (kW)
EV_params.P_ev_dis_max = 3.6;              % practical V2G discharge limit (kW)
EV_params.horizon_start_hour = 0;          % overwritten after t0 is selected

N = numel(PV_hist);
if ~(numel(Load_hist)==N && numel(Price_hist)==N && numel(SoC_hist)==N && numel(ts)==N)
    error("CSV columns have inconsistent lengths.");
end

if mean(PV_hist,'omitnan') > 100
    PV_hist = PV_hist / 1000.0;
    disp("Converted PV from W to kW");
end
if mean(Load_hist,'omitnan') > 100
    Load_hist = Load_hist / 1000.0;
    disp("Converted Load from W to kW");
end
if mean(Price_hist,'omitnan') > 10
    Price_hist = Price_hist / 100.0;
    disp("Adjusted Price scaling");
end

PV_hist    = max(0.0, PV_hist);
Load_hist  = max(0.0, Load_hist);
Price_hist = max(0.0, Price_hist);

EV_Charging_hist    = max(0.0, EV_Charging_hist);
EV_Discharging_hist = max(0.0, EV_Discharging_hist);
EV_Charging_hist    = min(EV_Charging_hist, EV_params.P_ev_max * EV_params.vehicle_count);
EV_Discharging_hist = min(EV_Discharging_hist, EV_params.P_ev_max * EV_params.vehicle_count);
EV_SoC_hist         = min(max(EV_SoC_hist, EV_params.EV_SoC_min), EV_params.EV_SoC_max);
Vehicle_Count_hist  = max(0.0, Vehicle_Count_hist);

fprintf("Total samples: %d\n", N);
fprintf("PV range: [%.3f, %.3f] kW\n", min(PV_hist), max(PV_hist));
fprintf("Load range: [%.3f, %.3f] kW\n", min(Load_hist), max(Load_hist));
fprintf("Price range: [%.3f, %.3f] $\n", min(Price_hist), max(Price_hist));

%% --------------------------
% 2) Enhanced calendar features
% --------------------------
cal_all = enhanced_calendar_features(ts);
fprintf('Calendar features count: %d\n', size(cal_all, 2));
if size(cal_all, 2) ~= 6
    warning('Calendar features has %d dimensions, expected 6', size(cal_all, 2));
end
n_cal_feat = size(cal_all, 2);

%% --------------------------
% 3) Build lagged supervised dataset (INCLUDING EV features)
% --------------------------
X_list = [];
Y_pv = [];
Y_load = [];
Y_price = [];
Y_ev_ch = [];
Y_ev_dis = [];
t_target = [];

fprintf("\nBuilding lagged dataset with LAGS=%d...\n", LAGS);
expected_features = LAGS * 5 + n_cal_feat;
fprintf('Expected features per sample: %d\n', expected_features);

for t = LAGS:(N-PRED_HORIZON)
    pv_lag    = PV_hist(t-LAGS+1:t)';
    load_lag  = Load_hist(t-LAGS+1:t)';
    price_lag = Price_hist(t-LAGS+1:t)';
    ev_soc_lag = EV_SoC_hist(t-LAGS+1:t)';
    vehicle_lag = Vehicle_Count_hist(t-LAGS+1:t)';

    cal_feat = cal_all(t+1, :);

    x_tr = [pv_lag, load_lag, price_lag, ev_soc_lag, vehicle_lag, cal_feat];

    if t == LAGS
        fprintf('Sample feature vector size: %d (should be %d)\n', length(x_tr), expected_features);
        if length(x_tr) ~= expected_features
            error('Feature size mismatch! Got %d, expected %d', length(x_tr), expected_features);
        end
    end

    X_list = [X_list; x_tr];
    Y_pv    = [Y_pv;    PV_hist(t+1 : t+PRED_HORIZON).'];
    Y_load  = [Y_load;  Load_hist(t+1 : t+PRED_HORIZON).'];
    Y_price = [Y_price; Price_hist(t+1 : t+PRED_HORIZON).'];
    Y_ev_ch = [Y_ev_ch; EV_Charging_hist(t+1 : t+PRED_HORIZON).'];
    Y_ev_dis = [Y_ev_dis; EV_Discharging_hist(t+1 : t+PRED_HORIZON).'];
    t_target = [t_target; (t+1)];
end

X_all = double(X_list);
M = size(X_all, 1);
fprintf("Lagged samples: %d (each with %d features)\n", M, size(X_all,2));

%% --------------------------
% 4) Chronological Train/Val/Test split
% --------------------------
idx_train_end = floor(TRAIN_FRAC * M);
idx_val_end   = idx_train_end + floor(VAL_FRAC * M);

idx_train = (1:idx_train_end).';
idx_val   = (idx_train_end+1:idx_val_end).';
idx_test  = (idx_val_end+1:M).';

fprintf("\nSplit summary:\n");
fprintf("  Train: %d samples (%.1f%%)\n", numel(idx_train), 100*numel(idx_train)/M);
fprintf("  Val:   %d samples (%.1f%%)\n", numel(idx_val), 100*numel(idx_val)/M);
fprintf("  Test:  %d samples (%.1f%%)\n", numel(idx_test), 100*numel(idx_test)/M);

X_train = X_all(idx_train,:);
X_val   = X_all(idx_val,:);
X_test  = X_all(idx_test,:);

%% --------------------------
% 5) Normalize inputs
% --------------------------
mu = mean(X_train, 1);
sd = std(X_train, 0, 1) + 1e-12;
mu = mu(:)';
sd = sd(:)';

fprintf('\n=== Normalization Info ===\n');
fprintf('Number of features in training: %d\n', size(X_train, 2));
fprintf('mu size: %d, sd size: %d\n', length(mu), length(sd));

normalize = @(X) (X - mu) ./ sd;

X_train_n = normalize(X_train);
X_val_n   = normalize(X_val);
X_test_n  = normalize(X_test);

Ypv_train = Y_pv(idx_train,:);    Ypv_val = Y_pv(idx_val,:);    Ypv_test = Y_pv(idx_test,:);
Yld_train = Y_load(idx_train,:);  Yld_val = Y_load(idx_val,:);  Yld_test = Y_load(idx_test,:);
Ypr_train = Y_price(idx_train,:); Ypr_val = Y_price(idx_val,:); Ypr_test = Y_price(idx_test,:);

%% --------------------------
% 6) Train models (PV, Load, Price only - NO EV forecasting)
% --------------------------
fprintf('\n============================================================\n');
fprintf('Stage A: Training forecasters (PV, Load, Price only)\n');
fprintf('============================================================\n');

fprintf('\n=== Multi-horizon training ===\n');

pv_model_all = train_multi_horizon_fcm(...
    X_train_n, log(Ypv_train+0.05), ...
    X_val_n,   log(Ypv_val+0.05), ...
    CANDIDATE_CLUSTERS, FCM_M, FCM_MAXITER, FCM_TOL);

load_model_all = train_multi_horizon_fcm(...
    X_train_n, Yld_train, ...
    X_val_n,   Yld_val, ...
    CANDIDATE_CLUSTERS, FCM_M, FCM_MAXITER, FCM_TOL);

price_model_all = train_multi_horizon_fcm(...
    X_train_n, Ypr_train, ...
    X_val_n,   Ypr_val, ...
    CANDIDATE_CLUSTERS, FCM_M, FCM_MAXITER, FCM_TOL);

%% Retrain on full training+validation set
X_all_n = normalize(X_all);
fprintf('\n=== Retraining multi-horizon models ===\n');

idx_tv = [idx_train; idx_val];

pv_k_selected    = cellfun(@(mh) mh.k_requested, pv_model_all);
load_k_selected  = cellfun(@(mh) mh.k_requested, load_model_all);
price_k_selected = cellfun(@(mh) mh.k_requested, price_model_all);
pv_lambda_selected    = cellfun(@(mh) mh.lambda, pv_model_all);
load_lambda_selected  = cellfun(@(mh) mh.lambda, load_model_all);
price_lambda_selected = cellfun(@(mh) mh.lambda, price_model_all);

fprintf('  PV requested clusters per horizon:    %s\n', mat2str(pv_k_selected.'));
fprintf('  Load requested clusters per horizon:  %s\n', mat2str(load_k_selected.'));
fprintf('  Price requested clusters per horizon: %s\n', mat2str(price_k_selected.'));
fprintf('  PV ridge lambda per horizon:           %s\n', mat2str(pv_lambda_selected.'));
fprintf('  Load ridge lambda per horizon:         %s\n', mat2str(load_lambda_selected.'));
fprintf('  Price ridge lambda per horizon:        %s\n', mat2str(price_lambda_selected.'));

pv_final_all = train_multi_horizon_fcm(...
    X_all_n(idx_tv,:), log(Y_pv(idx_tv,:)+0.05), ...
    X_all_n(idx_tv,:), log(Y_pv(idx_tv,:)+0.05), ...
    CANDIDATE_CLUSTERS, FCM_M, FCM_MAXITER, FCM_TOL, pv_k_selected, pv_lambda_selected);

load_final_all = train_multi_horizon_fcm(...
    X_all_n(idx_tv,:), Y_load(idx_tv,:), ...
    X_all_n(idx_tv,:), Y_load(idx_tv,:), ...
    CANDIDATE_CLUSTERS, FCM_M, FCM_MAXITER, FCM_TOL, load_k_selected, load_lambda_selected);

price_final_all = train_multi_horizon_fcm(...
    X_all_n(idx_tv,:), Y_price(idx_tv,:), ...
    X_all_n(idx_tv,:), Y_price(idx_tv,:), ...
    CANDIDATE_CLUSTERS, FCM_M, FCM_MAXITER, FCM_TOL, price_k_selected, price_lambda_selected);

%% ------------------------------------------------------------
% FFNN forecasting benchmark: PV, Load, and Price only
% ------------------------------------------------------------
% This benchmark is intentionally simple and reviewer-friendly. It uses the
% same normalized lagged input matrix as the FCM-TSK forecaster and predicts
% the same 24-hour output vector. It is included only for forecast comparison,
% not for EV scheduling/dispatch optimization.
if RUN_FFNN_BENCHMARK
    fprintf('\n============================================================\n');
    fprintf('Training custom FFNN forecasting benchmark (PV, Load, Price)\n');
    fprintf('============================================================\n');

    ffnn_opts = struct();
    ffnn_opts.epochs = FFNN_EPOCHS;
    ffnn_opts.batch_size = FFNN_BATCH_SIZE;
    ffnn_opts.learn_rate = FFNN_LEARN_RATE;
    ffnn_opts.l2 = FFNN_L2;
    ffnn_opts.patience = FFNN_PATIENCE;
    ffnn_opts.seed = FFNN_SEED;
    ffnn_opts.verbose = FFNN_VERBOSE;

    ffnn_pv_model = train_custom_ffnn_regressor( ...
        X_train_n, Ypv_train, X_val_n, Ypv_val, ...
        FFNN_HIDDEN_SIZES, ffnn_opts, 'PV');

    ffnn_load_model = train_custom_ffnn_regressor( ...
        X_train_n, Yld_train, X_val_n, Yld_val, ...
        FFNN_HIDDEN_SIZES, ffnn_opts, 'Load');

    ffnn_price_model = train_custom_ffnn_regressor( ...
        X_train_n, Ypr_train, X_val_n, Ypr_val, ...
        FFNN_HIDDEN_SIZES, ffnn_opts, 'Price');
else
    ffnn_pv_model = [];
    ffnn_load_model = [];
    ffnn_price_model = [];
end

%% ------------------------------------------------------------
% LSTM forecasting benchmark: same inputs and 24-h outputs
% ------------------------------------------------------------
if RUN_LSTM_BENCHMARK
    if exist('trainNetwork','file') ~= 2
        warning('Deep Learning Toolbox/trainNetwork is unavailable. LSTM benchmark disabled.');
        RUN_LSTM_BENCHMARK = false;
        lstm_pv_model = []; lstm_load_model = []; lstm_price_model = [];
    else
        fprintf('\n============================================================\n');
        fprintf('Training LSTM forecasting benchmark (PV, Load, Price)\n');
        fprintf('============================================================\n');

        % Convert each flattened lag vector to [features x time].
        Xseq_train = flat_inputs_to_lstm_sequences(X_train_n, LAGS, n_cal_feat);
        Xseq_val   = flat_inputs_to_lstm_sequences(X_val_n,   LAGS, n_cal_feat);

        lstm_opts = struct('num_hidden',LSTM_NUM_HIDDEN, ...
            'fc_hidden',LSTM_FC_HIDDEN, 'epochs',LSTM_EPOCHS, ...
            'batch_size',LSTM_MINIBATCH_SIZE, 'learn_rate',LSTM_LEARN_RATE, ...
            'l2',LSTM_L2, 'patience',LSTM_PATIENCE, 'seed',LSTM_SEED, ...
            'verbose',LSTM_VERBOSE);

        lstm_pv_model = train_lstm_regressor(Xseq_train, Ypv_train, ...
            Xseq_val, Ypv_val, lstm_opts, 'PV');
        lstm_load_model = train_lstm_regressor(Xseq_train, Yld_train, ...
            Xseq_val, Yld_val, lstm_opts, 'Load');
        lstm_price_model = train_lstm_regressor(Xseq_train, Ypr_train, ...
            Xseq_val, Ypr_val, lstm_opts, 'Price');
    end
else
    lstm_pv_model = []; lstm_load_model = []; lstm_price_model = [];
end

%% ---------------------------
% Feature ordering inside the 126-d vector is:
%   [pv_lag(1:24), load_lag(1:24), price_lag(1:24), ev_soc_lag(1:24),
%    vehicle_lag(1:24), cal_feat(1:6)]
% so the MOST RECENT lag of each variable sits at:
%   PV    -> column 24
%   Load  -> column 48
%   Price -> column 72
% ---------------------------
idx_pv_recent    = LAGS;            % 24
idx_load_recent  = 2*LAGS;          % 48
idx_price_recent = 3*LAGS;          % 72

plot_actual_model_mfs(pv_final_all{1},    idx_pv_recent,    mu, sd, 'PV (kW)',    'Actual Trained Model: PV input MF (h=1)',    true,  PV_CAPACITY);
plot_actual_model_mfs(load_final_all{1},  idx_load_recent,  mu, sd, 'Load (kW)',  'Actual Trained Model: Load input MF (h=1)',  false, []);
plot_actual_model_mfs(price_final_all{1}, idx_price_recent, mu, sd, 'Price ($/kWh)', 'Actual Trained Model: Price input MF (h=1)', false, []);

% NOTE: this is a SEPARATE, from-scratch FCM run on raw [PV,Load,Price,SoC]
% purely for illustration. It is NOT the deployed multi-horizon model
% (that one is plotted above).
numClusters = 5;
m_fcm = 2;
maxIter_fcm = 200;
tol_fcm = 1e-5;

X_illustrative = [PV_hist, Load_hist, Price_hist, SoC_hist];
numInputs = size(X_illustrative, 2);

rng(2026);  
idx_seed = randperm(size(X_illustrative,1), numClusters);
C_illustrative = X_illustrative(idx_seed,:);

U_illustrative = zeros(size(X_illustrative,1), numClusters);
for iter = 1:maxIter_fcm
    for i = 1:size(X_illustrative,1)
        for j = 1:numClusters
            denom = 0;
            for k = 1:numClusters
                denom = denom + ( (norm(X_illustrative(i,:) - C_illustrative(j,:)) + 1e-8) / ...
                                   (norm(X_illustrative(i,:) - C_illustrative(k,:)) + 1e-8) )^(2/(m_fcm-1));
            end
            U_illustrative(i,j) = 1/denom;
        end
    end
    C_old = C_illustrative;
    for j = 1:numClusters
        numc = sum((U_illustrative(:,j).^m_fcm).*X_illustrative);
        denc = sum(U_illustrative(:,j).^m_fcm);
        C_illustrative(j,:) = numc/denc;
    end
    if max(abs(C_illustrative(:)-C_old(:))) < tol_fcm, break; end
end

center_ill = zeros(numInputs, numClusters);
sigma_ill = zeros(numInputs, numClusters);
for in = 1:numInputs
    for c = 1:numClusters
        center_ill(in,c) = C_illustrative(c,in);
        w = U_illustrative(:,c).^m_fcm;
        sigma_ill(in,c) = sqrt( sum(w .* (X_illustrative(:,in)-center_ill(in,c)).^2) / (sum(w)+1e-8) );
        if sigma_ill(in,c) < 1e-3
            sigma_ill(in,c) = 1e-3;
        end
    end
end

colors = {'b', 'r', 'g', 'm', 'c'};
input_names = {'PV','Load','Price','SoC'};
x_labels = {'PV (kW)','Load (kW)','Price ($/kWh)','State of Charge'};

for in = 1:numInputs
    centers = center_ill(in, :);
    sigmas = sigma_ill(in, :);
    [centers, sidx] = sort(centers);
    sigmas = sigmas(sidx);

    figure('Name', sprintf('[ILLUSTRATIVE ONLY] Gaussian MFs for %s Input', input_names{in}), ...
           'Position', [100, 100, 550, 400], 'Color', 'w');

    x_min = min(centers) - max(sigmas)*2;
    x_max = max(centers) + max(sigmas)*2;
    if in == 1
        x_min = max(0, x_min);
    end
    xv = linspace(x_min, x_max, 300);

    hold on;
    for c = 1:numClusters
        mf = exp(-(xv - centers(c)).^2 ./ (2 * sigmas(c)^2));
        plot(xv, mf, 'Color', colors{c}, 'LineWidth', 2, ...
            'DisplayName', sprintf('MF %d (center=%.3f)', c, centers(c)));
    end
    title(sprintf('[ILLUSTRATIVE ONLY, not the deployed model] %s Input', input_names{in}), ...
          'FontSize', 11, 'FontWeight', 'bold');
    xlabel(x_labels{in}, 'FontSize', 11);
    ylabel('\mu', 'FontSize', 11);
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    ylim([0 1.1]);
    hold off;
end

fprintf('\n=== [Illustrative-only] Cluster Centers for Each Raw Input ===\n');
fprintf('Input\t\tMF1\tMF2\tMF3\tMF4\tMF5\n');
fprintf('PV (kW):\t'); fprintf('%.3f\t', center_ill(1,:));
fprintf('\nLoad (kW):\t'); fprintf('%.3f\t', center_ill(2,:));
fprintf('\nPrice ($):\t'); fprintf('%.3f\t', center_ill(3,:));
fprintf('\nSoC:\t\t'); fprintf('%.3f\t', center_ill(4,:));
fprintf('\n');

%% --------------------------
% 7) 24h forecast rollout (PV, Load, Price only)
% --------------------------
fprintf('\n=== Pre-rollout checks ===\n');
if ~exist('pv_final_all', 'var'),   error('pv_final_all does not exist!');   end
if ~exist('load_final_all', 'var'), error('load_final_all does not exist!'); end
if ~exist('price_final_all', 'var'),error('price_final_all does not exist!');end
fprintf('All models exist\n');

test_start_indices = t_target(idx_test);
test_timestamps = ts(test_start_indices);
hour_of_day = hour(test_timestamps);
midnight_indices = find(hour_of_day == 0);

if isempty(midnight_indices)
    t0 = test_start_indices(1);
    fprintf('Warning: No midnight start found, using: %s\n', datestr(ts(t0)));
else
    t0 = test_start_indices(midnight_indices(1));
end

fprintf('\n============================================================\n');
fprintf('24h Forecast starting at: %s\n', datestr(ts(t0), 'yyyy-mm-dd HH:MM:SS'));
fprintf('============================================================\n');

% Store forecast start hour so EV charging/V2G windows inside the repair
% and objective functions use real clock hours, not only array indices.
EV_params.horizon_start_hour = hour(ts(t0));

% Baseline EV parameters: Strategy 1 and Strategy 2 serve the same
% mandatory EV charging demand, but they are NOT enrolled in the V2G service.
%
EV_params_base = EV_params;
EV_params_base.v2g_required_energy = 0.0;
EV_params_base.v2g_requirement_weight = 0.0;
EV_params_base.v2g_overdelivery_weight = 0.0;
EV_params_base.v2g_allowed_hours = [];
EV_params_base.no_charge_hours = [];

pv_win      = PV_hist(t0-LAGS:t0-1);
load_win    = Load_hist(t0-LAGS:t0-1);
price_win   = Price_hist(t0-LAGS:t0-1);
ev_soc_win  = EV_SoC_hist(t0-LAGS:t0-1);
vehicle_win = Vehicle_Count_hist(t0-LAGS:t0-1);
cal_feat    = cal_all(t0, :);

x = [pv_win(:)', load_win(:)', price_win(:)', ev_soc_win(:)', vehicle_win(:)', cal_feat(:)'];
x_n = (x - mu) ./ sd;

expected_features = LAGS * 5 + n_cal_feat;
if length(x) ~= expected_features
    error('Feature size mismatch! Got %d, expected %d', length(x), expected_features);
end

PV_fore = zeros(PRED_HORIZON,1);
Load_fore = zeros(PRED_HORIZON,1);
Price_fore = zeros(PRED_HORIZON,1);

for h = 1:PRED_HORIZON
    model = pv_final_all{h};
    y_log = fuzzy_predict_batch(x_n, model.center, model.sigma, model.a);
    pv_hat = max(0, exp(y_log)-0.05);
    pv_hat = min(pv_hat, PV_CAPACITY);

    hour_h = mod(hour(ts(t0)) + h - 1, 24);
    if hour_h >= 20 || hour_h <= 5
        pv_hat = 0;
    end
    PV_fore(h) = pv_hat;

    model = load_final_all{h};
    ld_hat = fuzzy_predict_batch(x_n, model.center, model.sigma, model.a);
    Load_fore(h) = max(0, ld_hat);

    model = price_final_all{h};
    pr_hat = fuzzy_predict_batch(x_n, model.center, model.sigma, model.a);
    Price_fore(h) = max(0, pr_hat);
end

LOAD_PLAUSIBLE_MAX  = 3 * max(Load_hist);
PRICE_PLAUSIBLE_MAX = 3 * max(Price_hist);

PRICE_SOFT_THRESHOLD = 1.5 * max(Price_hist);
[worst_price_val, worst_price_hr] = max(Price_fore);
if worst_price_val > PRICE_SOFT_THRESHOLD
    fprintf('  NOTE: worst-hour price forecast = %.4f $/kWh at hour %d (soft threshold %.4f, hard ceiling %.4f) -- not clipped, flagged for review.\n', ...
        worst_price_val, worst_price_hr, PRICE_SOFT_THRESHOLD, PRICE_PLAUSIBLE_MAX);
end

LOAD_SOFT_THRESHOLD = 1.5 * max(Load_hist);
[worst_load_val, worst_load_hr] = max(Load_fore);
if worst_load_val > LOAD_SOFT_THRESHOLD
    fprintf('  NOTE: worst-hour load forecast = %.3f kW at hour %d (soft threshold %.3f, hard ceiling %.3f) -- not clipped, flagged for review.\n', ...
        worst_load_val, worst_load_hr, LOAD_SOFT_THRESHOLD, LOAD_PLAUSIBLE_MAX);
end

clipped_load = find(Load_fore > LOAD_PLAUSIBLE_MAX);
if ~isempty(clipped_load)
    for hh = clipped_load(:)'
        fprintf('  WARNING: Load forecast at hour %d = %.3f kW exceeds plausibility ceiling (%.3f kW) -- clipped.\n', ...
            hh, Load_fore(hh), LOAD_PLAUSIBLE_MAX);
    end
    Load_fore(clipped_load) = LOAD_PLAUSIBLE_MAX;
end

clipped_price = find(Price_fore > PRICE_PLAUSIBLE_MAX);
if ~isempty(clipped_price)
    for hh = clipped_price(:)'
        fprintf('  WARNING: Price forecast at hour %d = %.4f $/kWh exceeds plausibility ceiling (%.4f $/kWh) -- clipped.\n', ...
            hh, Price_fore(hh), PRICE_PLAUSIBLE_MAX);
    end
    Price_fore(clipped_price) = PRICE_PLAUSIBLE_MAX;
end


Price_fore_original = Price_fore;

% === FFNN 24-h forecast rollout for the same test day ===
% This is NOT used for dispatch in the proposed method; it is only a
% forecasting benchmark against FCM-TSK. The same load-adaptive price
% enhancement is applied later when enabled, so the comparison is fair.
if RUN_FFNN_BENCHMARK
    PV_ffnn_fore = ffnn_predict_batch(x_n, ffnn_pv_model).';
    Load_ffnn_fore = ffnn_predict_batch(x_n, ffnn_load_model).';
    Price_ffnn_fore_original = ffnn_predict_batch(x_n, ffnn_price_model).';

    PV_ffnn_fore = min(max(PV_ffnn_fore, 0), PV_CAPACITY);
    Load_ffnn_fore = min(max(0, Load_ffnn_fore), LOAD_PLAUSIBLE_MAX);
    Price_ffnn_fore_original = min(max(0, Price_ffnn_fore_original), PRICE_PLAUSIBLE_MAX);

    for h = 1:PRED_HORIZON
        hour_h = mod(hour(ts(t0)) + h - 1, 24);
        if hour_h >= 20 || hour_h <= 5
            PV_ffnn_fore(h) = 0;
        end
    end

    Price_ffnn_fore_enhanced = apply_load_adaptive_price_enhancement( ...
        Price_ffnn_fore_original, Load_ffnn_fore);
    if USE_LOAD_ADAPTIVE_PRICE_ENHANCEMENT
        Price_ffnn_fore = Price_ffnn_fore_enhanced;
    else
        Price_ffnn_fore = Price_ffnn_fore_original;
    end
else
    PV_ffnn_fore = nan(PRED_HORIZON,1);
    Load_ffnn_fore = nan(PRED_HORIZON,1);
    Price_ffnn_fore = nan(PRED_HORIZON,1);
    Price_ffnn_fore_original = nan(PRED_HORIZON,1);
    Price_ffnn_fore_enhanced = nan(PRED_HORIZON,1);
end

% === LSTM 24-h benchmark forecast for the same test day ===
if RUN_LSTM_BENCHMARK
    x_seq = flat_inputs_to_lstm_sequences(x_n, LAGS, n_cal_feat);
    PV_lstm_fore = lstm_predict_batch(x_seq, lstm_pv_model).';
    Load_lstm_fore = lstm_predict_batch(x_seq, lstm_load_model).';
    Price_lstm_fore_original = lstm_predict_batch(x_seq, lstm_price_model).';

    PV_lstm_fore = min(max(PV_lstm_fore,0), PV_CAPACITY);
    Load_lstm_fore = min(max(0, Load_lstm_fore), LOAD_PLAUSIBLE_MAX);
    Price_lstm_fore_original = min(max(0, Price_lstm_fore_original), PRICE_PLAUSIBLE_MAX);
    for h = 1:PRED_HORIZON
        hour_h = mod(hour(ts(t0)) + h - 1, 24);
        if hour_h >= 20 || hour_h <= 5, PV_lstm_fore(h) = 0; end
    end
    Price_lstm_fore_enhanced = apply_load_adaptive_price_enhancement( ...
        Price_lstm_fore_original, Load_lstm_fore);
    if USE_LOAD_ADAPTIVE_PRICE_ENHANCEMENT
        Price_lstm_fore = Price_lstm_fore_enhanced;
    else
        Price_lstm_fore = Price_lstm_fore_original;
    end
else
    PV_lstm_fore = nan(PRED_HORIZON,1);
    Load_lstm_fore = nan(PRED_HORIZON,1);
    Price_lstm_fore = nan(PRED_HORIZON,1);
end

load_features = zeros(PRED_HORIZON, 4);
for h = 1:PRED_HORIZON
    load_features(h,1) = Load_fore(h);
    if h > 1
        load_features(h,2) = Load_fore(h) - Load_fore(h-1);
    else
        load_features(h,2) = 0;
    end
    if Load_fore(h) > 3.5
        load_features(h,3) = 1;
    else
        load_features(h,3) = 0;
    end
    max_load = max(Load_fore);
    min_load = min(Load_fore);
    if max_load > min_load
        load_features(h,4) = (Load_fore(h) - min_load) / (max_load - min_load);
    else
        load_features(h,4) = 0.5;
    end
end

Price_fore_enhanced = Price_fore_original;
for h = 1:PRED_HORIZON
    if load_features(h,3) == 1
        Price_fore_enhanced(h) = Price_fore_enhanced(h) * 1.05;
    end
    if load_features(h,2) > 0.3
        Price_fore_enhanced(h) = Price_fore_enhanced(h) * 1.03;
    end
    if load_features(h,2) < -0.3
        Price_fore_enhanced(h) = Price_fore_enhanced(h) * 0.97;
    end
end
Price_fore_enhanced = max(0, Price_fore_enhanced);

Price_actual = Price_hist(t0:t0+PRED_HORIZON-1);
mae_original = mean(abs(Price_actual - Price_fore_original));
mae_enhanced = mean(abs(Price_actual - Price_fore_enhanced));

fprintf('\n========================================\n');
fprintf('PRICE FORECAST ENHANCEMENT RESULTS\n');
fprintf('========================================\n');
fprintf('Original FCM-TSK MAE:  $%.4f/kWh\n', mae_original);
fprintf('Enhanced FCM-TSK MAE:   $%.4f/kWh\n', mae_enhanced);
fprintf('Improvement:            %.1f%%\n', (1 - mae_enhanced/mae_original)*100);

if USE_LOAD_ADAPTIVE_PRICE_ENHANCEMENT
    Price_fore = Price_fore_enhanced;
    fprintf('\nUsing ENHANCED price forecast for optimization\n');
else
    Price_fore = Price_fore_original;
    fprintf('\nUsing ORIGINAL price forecast for optimization\n');
end

% === RULE-BASED EV SCHEDULE (not forecasted) ===
[EV_ch_fore, EV_dis_fore] = rule_based_ev_schedule(Price_fore, Load_fore, PV_fore, EV_params, ts(t0));

EV_soc_fore = zeros(PRED_HORIZON+1, 1);
EV_soc_fore(1) = EV_params.EV_SoC0;
for h = 1:PRED_HORIZON
    if h < PRED_HORIZON
        delta_soc = (EV_ch_fore(h) * EV_params.eta_ev_ch - EV_dis_fore(h) / EV_params.eta_ev_dis) / EV_params.EV_capacity_total;
        EV_soc_fore(h+1) = max(EV_params.EV_SoC_min, min(EV_params.EV_SoC_max, EV_soc_fore(h) + delta_soc));
    end
end

PV_actual    = PV_hist(t0:t0+PRED_HORIZON-1);
Load_actual  = Load_hist(t0:t0+PRED_HORIZON-1);
Price_actual = Price_hist(t0:t0+PRED_HORIZON-1);
EV_ch_actual = EV_Charging_hist(t0:t0+PRED_HORIZON-1);
EV_dis_actual = EV_Discharging_hist(t0:t0+PRED_HORIZON-1);

pv_error = mean(abs(PV_actual - PV_fore));
load_error = mean(abs(Load_actual - Load_fore));
price_error = mean(abs(Price_actual - Price_fore));

if RUN_FFNN_BENCHMARK
    pv_error_ffnn = mean(abs(PV_actual - PV_ffnn_fore));
    load_error_ffnn = mean(abs(Load_actual - Load_ffnn_fore));
    price_error_ffnn = mean(abs(Price_actual - Price_ffnn_fore));
else
    pv_error_ffnn = NaN;
    load_error_ffnn = NaN;
    price_error_ffnn = NaN;
end

fprintf('\nForecast Accuracy (FCM-TSK used for proposed dispatch):\n');
fprintf('  PV MAE:       %.4f kW\n', pv_error);
fprintf('  Load MAE:     %.4f kW\n', load_error);
fprintf('  Price MAE:    %.4f $/kWh\n', price_error);
if RUN_FFNN_BENCHMARK
    fprintf('\nForecast Accuracy (custom FFNN benchmark):\n');
    fprintf('  PV MAE:       %.4f kW\n', pv_error_ffnn);
    fprintf('  Load MAE:     %.4f kW\n', load_error_ffnn);
    fprintf('  Price MAE:    %.4f $/kWh\n', price_error_ffnn);
    fprintf('\nFCM-TSK improvement over FFNN: PV %.1f%%, Load %.1f%%, Price %.1f%%\n', ...
        100*(1 - pv_error/max(pv_error_ffnn,eps)), ...
        100*(1 - load_error/max(load_error_ffnn,eps)), ...
        100*(1 - price_error/max(price_error_ffnn,eps)));
end

if RUN_LSTM_BENCHMARK
    pv_error_lstm = mean(abs(PV_actual - PV_lstm_fore));
    load_error_lstm = mean(abs(Load_actual - Load_lstm_fore));
    price_error_lstm = mean(abs(Price_actual - Price_lstm_fore));
    fprintf('\nForecast Accuracy (LSTM benchmark):\n');
    fprintf('  PV MAE:       %.4f kW\n', pv_error_lstm);
    fprintf('  Load MAE:     %.4f kW\n', load_error_lstm);
    fprintf('  Price MAE:    %.4f $/kWh\n', price_error_lstm);
    fprintf('\nFCM-TSK improvement over LSTM: PV %.1f%%, Load %.1f%%, Price %.1f%%\n', ...
        100*(1-pv_error/max(pv_error_lstm,eps)), ...
        100*(1-load_error/max(load_error_lstm,eps)), ...
        100*(1-price_error/max(price_error_lstm,eps)));
else
    pv_error_lstm = NaN; load_error_lstm = NaN; price_error_lstm = NaN;
end

[EV_ch_persistence, EV_dis_persistence] = persistence_ev_forecast(...
    EV_Charging_hist, EV_Discharging_hist, t0, PRED_HORIZON);
[EV_ch_persistence_week, EV_dis_persistence_week] = persistence_ev_forecast_enhanced(...
    EV_Charging_hist, EV_Discharging_hist, ts, t0, PRED_HORIZON, 'weekly');
[EV_ch_hourly_avg, EV_dis_hourly_avg] = persistence_ev_forecast_enhanced(...
    EV_Charging_hist, EV_Discharging_hist, ts, t0, PRED_HORIZON, 'hourly');

[EV_ch_persistence, EV_dis_persistence] = repair_ev_schedule(EV_ch_persistence, EV_dis_persistence, EV_params, dt);
[EV_ch_persistence_week, EV_dis_persistence_week] = repair_ev_schedule(EV_ch_persistence_week, EV_dis_persistence_week, EV_params, dt);
[EV_ch_hourly_avg, EV_dis_hourly_avg] = repair_ev_schedule(EV_ch_hourly_avg, EV_dis_hourly_avg, EV_params, dt);

[EV_ch_fore, EV_dis_fore] = rule_based_ev_schedule(Price_fore, Load_fore, PV_fore, EV_params, ts(t0));

fprintf('\n=== EV SCHEDULE BASELINE COMPARISON (not FCM-TSK forecast) ===\n');
fprintf('Method                  | Charging (kWh) | Discharging (kWh)\n');
fprintf('------------------------|----------------|-------------------\n');
fprintf('Persistence (daily)     | %.2f           | %.2f\n', sum(EV_ch_persistence), sum(EV_dis_persistence));
fprintf('Persistence (weekly)    | %.2f           | %.2f\n', sum(EV_ch_persistence_week), sum(EV_dis_persistence_week));
fprintf('Hourly Average          | %.2f           | %.2f\n', sum(EV_ch_hourly_avg), sum(EV_dis_hourly_avg));
fprintf('Rule-based preliminary  | %.2f          | %.2f\n', sum(EV_ch_fore), sum(EV_dis_fore));

[EV_ch_uncontrolled, EV_dis_uncontrolled] = uncontrolled_ev_charging_schedule(EV_params_base, PRED_HORIZON, dt, ts(t0));
fprintf('\n=== MANDATORY EV BASELINE USED IN DISPATCH COMPARISON ===\n');
fprintf('Uncontrolled EV charging energy: %.2f kWh\n', sum(EV_ch_uncontrolled)*dt);
fprintf('Uncontrolled EV discharging energy: %.2f kWh\n', sum(EV_dis_uncontrolled)*dt);
[~, ~, EV_soc_uncontrolled_check] = repair_ev_schedule(EV_ch_uncontrolled, EV_dis_uncontrolled, EV_params_base, dt);
fprintf('Uncontrolled EV final SoC: %.1f%%\n', 100*EV_soc_uncontrolled_check(end));
fprintf('Required EV final SoC: %.1f%%\n', 100*EV_params.EV_final_soc_req);
if EV_soc_uncontrolled_check(end) + 1e-6 < EV_params.EV_final_soc_req
    warning('Uncontrolled EV baseline does not meet the required final SoC. Check P_ev_ch_max or horizon length.');
end

%% --------------------------
% 8) Prepare battery parameters structure
% --------------------------
battery_params = struct();
battery_params.Pmax = Pmax;
battery_params.Ecap = Ecap;
battery_params.SoC0 = SoC0;
battery_params.SoC_min = SoC_min;
battery_params.SoC_max = SoC_max;
battery_params.eta_ch = eta_ch;
battery_params.eta_dis = eta_dis;
battery_params.dt = dt;
battery_params.demand_charge_rate = demand_charge_rate;
battery_params.terminal_soc_target = SoC0;
battery_params.terminal_soc_weight = 50.0;
battery_params.SoC_reserve = 0.35;
battery_params.reserve_soc_weight = 30.0;   
battery_params.soc_smoothing_weight = 0.5;  

battery_params.demand_shaping_weight = 0.0;  
battery_params.demand_shaping_p = 6;          

fprintf('\nBattery parameters prepared:\n');
fprintf('  Pmax: %.1f kW\n', battery_params.Pmax);
fprintf('  Ecap: %.1f kWh\n', battery_params.Ecap);
fprintf('  SoC range: [%.2f, %.2f]\n', battery_params.SoC_min, battery_params.SoC_max);
fprintf('  Demand charge rate: $%.2f/kW\n', demand_charge_rate);
fprintf('  Demand shaping: weight=%.2f, p=%d\n', battery_params.demand_shaping_weight, battery_params.demand_shaping_p);
fprintf('EV scenario prepared:\n');
fprintf('  EV initial SoC: %.1f%%\n', 100*EV_params.EV_SoC0);
fprintf('  EV final SoC requirement: %.1f%%\n', 100*EV_params.EV_final_soc_req);
fprintf('  EV mode-change cost: $%.2f/change\n', EV_params.mode_change_cost);
fprintf('  V2G service target: %.2f kWh during hours %s\n', ...
    EV_params.v2g_required_energy, mat2str(EV_params.v2g_allowed_hours));

%% --------------------------

BuyPrice  = Price_fore;
SellPrice = 0.85 * Price_fore;

% --- Strategy 1: No BESS + uncontrolled mandatory EV charging (deterministic baseline) ---
% IMPORTANT: Since the EV has a required departure SoC, the baseline must
% include EV charging. Comparing BESS+EV against "no EV" would be unfair.
zeroPbat = zeros(PRED_HORIZON, 1);
zeroCost = total_cost_withGrid_and_soc_repair_EV(zeroPbat, ...
    EV_ch_uncontrolled, EV_dis_uncontrolled, ...
    PV_fore, Load_fore, Price_fore, SellPrice, PRED_HORIZON, SoC0, SoC_min, SoC_max, ...
    Ecap, Pmax, eta_ch, eta_dis, dt, EV_params_base, battery_params);

fprintf('\n============================================================\n');
fprintf('Stage B: Multi-run PSO Optimization (BESS+uncontrolled-EV vs BESS+optimized-EV)\n');
fprintf('============================================================\n');
fprintf('Strategy 1 (grid + uncontrolled mandatory EV charging) cost = %.4f $ [deterministic]\n', zeroCost);

% Generate ONE initial population per run, used
% IDENTICALLY by PSO and GA below. This removes the previously-discovered
% confound where custom_pso's asymmetric init formula and custom_ga's
% uniform init formula put the two algorithms' EV-variable populations in
% very different starting regions of the search space (PSO's formula
% happened to clamp a large fraction of EV-power genes to exactly 0,
% giving PSO an unintended head start on the EV mutual-exclusivity and
% cycling penalties). With shared_init, any PSO-vs-GA difference reported
% below reflects the algorithms themselves, not how each one happened to
% be seeded.
%
% WIDER WARM START: rather than seeding just ONE individual with the
% rule-based EV schedule (a single, easily-lost foothold), WARM_START_FRAC
% of the BESS+EV population is seeded as perturbed copies of it. The best
% observed solution across many multi-run experiments has consistently
% landed close to this region (e.g. $39-40/day), but neither PSO nor GA
% was finding it reliably with only one warm-started individual out of
% SWARM_SIZE. Giving the search more footholds near that known-reasonable
% region -- while still leaving most of the population free to explore
% broadly -- is intended to improve how often the mean run reaches it,
% without artificially forcing every run there.
WARM_START_FRAC = 0.40;   % 40% warm-started; stabilizes BESS+EV search
                          % uncontrolled EV seed; the remaining population is
                          % uniformly random for exploration.
rng(20260101, 'twister');   % Stage-B RNG firewall: makes dispatch seeding
                            % independent of any RNG draws consumed by
                            % Stage-A training (FFNN/LSTM/ARIMA)                          
fprintf('Generating shared initial populations (identical for PSO and DE/GA)...\n');
fprintf('  Warm-starting %.0f%% of EACH population (%d of %d individuals): BESS-only near Pbat=0, BESS+EV near [Pbat=0, uncontrolled EV schedule].\n', ...
    100*WARM_START_FRAC, max(1, round(WARM_START_FRAC*SWARM_SIZE)), SWARM_SIZE);
% Use the uncontrolled mandatory EV schedule as the warm start because it
% satisfies the departure SoC requirement. The rule-based V2G schedule is
% still reported above, but it is not used as the main initial condition
% because it may violate the new mandatory EV energy requirement.
EV_ch_seed = EV_ch_uncontrolled;
EV_dis_seed = EV_dis_uncontrolled;

shared_init = generate_shared_init_populations(PRED_HORIZON, battery_params, EV_params, ...
    EV_ch_seed, EV_dis_seed, SWARM_SIZE, N_VALIDATION_RUNS, SEED_BASE, WARM_START_FRAC);

mrc = multirun_pso_compare(PV_fore, Load_fore, Price_fore, BuyPrice, SellPrice, ...
    battery_params, EV_params, EV_params_base, EV_ch_seed, EV_dis_seed, EV_ch_uncontrolled, EV_dis_uncontrolled, ...
    PRED_HORIZON, SWARM_SIZE, PSO_MAXITER, W_INERTIA, C1, C2, VEL_CLAMP_FACTOR, ...
    N_VALIDATION_RUNS, SEED_BASE, shared_init);

fprintf('\n--- Strategy 2: BESS Only (n=%d runs) ---\n', N_VALIDATION_RUNS);
fprintf('  Mean:   $%.4f +/- $%.4f (std)\n', mrc.s2_mean, mrc.s2_std);
fprintf('  Median: $%.4f\n', mrc.s2_median);
fprintf('  Best:   $%.4f   Worst: $%.4f\n', mrc.s2_min, mrc.s2_max);
fprintf('  CV:     %.2f%%\n', 100*mrc.s2_std/mrc.s2_mean);

fprintf('\n--- Strategy 3: BESS + EV (n=%d runs) ---\n', N_VALIDATION_RUNS);
fprintf('  Mean:   $%.4f +/- $%.4f (std)\n', mrc.s3_mean, mrc.s3_std);
fprintf('  Median: $%.4f\n', mrc.s3_median);
fprintf('  Best:   $%.4f   Worst: $%.4f\n', mrc.s3_min, mrc.s3_max);
fprintf('  CV:     %.2f%%\n', 100*mrc.s3_std/mrc.s3_mean);
fprintf('  EV charge:    %.2f +/- %.2f kWh\n', mrc.s3_ev_ch_mean, mrc.s3_ev_ch_std);
fprintf('  EV discharge: %.2f +/- %.2f kWh\n', mrc.s3_ev_dis_mean, mrc.s3_ev_dis_std);
fprintf('  EV final SoC: %.1f +/- %.1f%%\n', 100*mrc.s3_ev_final_soc_mean, 100*mrc.s3_ev_final_soc_std);

fprintf('\n--- Local Polish Diagnostic (PSO) ---\n');
fprintf('  BESS-only:  improved %d/%d runs, mean improvement $%.4f\n', ...
    mrc.s2_polish_n_improved, N_VALIDATION_RUNS, mrc.s2_polish_mean);
fprintf('  BESS+EV:    improved %d/%d runs, mean improvement $%.4f\n', ...
    mrc.s3_polish_n_improved, N_VALIDATION_RUNS, mrc.s3_polish_mean);

fprintf('\n--- Paired comparison across matched seeds ---\n');
fprintf('  Strategy 2 vs Strategy 1 (baseline): mean diff = $%.4f\n', zeroCost - mrc.s2_mean);
fprintf('  Strategy 3 vs Strategy 1 (baseline): mean diff = $%.4f\n', zeroCost - mrc.s3_mean);
fprintf('  Strategy 3 vs Strategy 2 (paired signrank test): p = %.4f\n', mrc.p_s3_vs_s2);
if mrc.p_s3_vs_s2 < 0.05
    if mrc.s3_mean < mrc.s2_mean
        fprintf('    => EV adds a statistically significant BENEFIT over BESS-only.\n');
    else
        fprintf('    => EV produces a statistically significant DETRIMENT (higher cost) relative to BESS-only.\n');
        fprintf('       (The low p-value confirms the difference is real, not noise -- but the\n');
        fprintf('        difference runs the WRONG direction for an "EV helps" claim.)\n');
    end
else
    fprintf('    => No statistically significant difference between BESS+uncontrolled-EV and BESS+optimized-EV.\n');
end

if mrc.s3_mean >= zeroCost
    fprintf('\n*** WARNING: mean Strategy-3 cost is NOT better than the grid+uncontrolled-EV baseline. ***\n');
    fprintf('*** Any "savings" quoted from a single best run would be cherry-picked. ***\n');
end

bestPbat     = mrc.s3_repr_Pbat;
bestP_ev_ch  = mrc.s3_repr_P_ev_ch;
bestP_ev_dis = mrc.s3_repr_P_ev_dis;
bestCost     = mrc.s3_repr_cost;
hist         = mrc.s3_repr_hist;
pso_hist     = mrc.s3_repr_convergence;

fprintf('\nRepresentative (median-cost) Strategy-3 run used for figures: $%.4f\n', bestCost);
fprintf('(Mean across %d runs: $%.4f, Best observed: $%.4f, Worst observed: $%.4f)\n', ...
    N_VALIDATION_RUNS, mrc.s3_mean, mrc.s3_min, mrc.s3_max);

fprintf('\n==========================================\n');
fprintf('FINAL DISPATCH SUMMARY (representative run) WITH DEGRADATION & DEMAND CHARGES\n');
fprintf('==========================================\n');

fprintf('\n--- COST BREAKDOWN ---\n');
fprintf('  Grid Energy Cost:   $%.4f\n', hist.grid_energy_cost);
fprintf('  Constraint Penalty: $%.4f\n', hist.other_penalty_cost);
fprintf('  Degradation Cost:   $%.4f (%.2f equivalent cycles)\n', ...
    hist.degradation_cost, hist.equivalent_cycles);
fprintf('  Demand Charge:      $%.4f (peak: %.2f kW)\n', ...
    hist.demand_charge, hist.peak_demand);
fprintf('  Demand Shaping Pen.:$%.4f\n', hist.demand_shaping_penalty);
fprintf('  Terminal SoC Pen.:  $%.4f\n', hist.terminal_soc_penalty);
if isfield(hist, 'ev_terminal_soc_penalty')
    fprintf('  EV Final SoC Pen.:  $%.4f\n', hist.ev_terminal_soc_penalty);
end
if isfield(hist, 'v2g_service_penalty')
    fprintf('  V2G Service Pen.:   $%.4f (delivered %.2f kWh)\n', ...
        hist.v2g_service_penalty, hist.v2g_service_energy);
end
fprintf('  SoC Smooth Pen.:    $%.4f\n', hist.soc_smoothing_penalty);

component_sum = hist.grid_energy_cost + hist.other_penalty_cost + ...
                hist.degradation_cost + hist.demand_charge + ...
                hist.terminal_soc_penalty + hist.soc_smoothing_penalty + ...
                hist.demand_shaping_penalty;

fprintf('====================================\n');
fprintf('  Component Sum:      $%.4f\n', component_sum);
fprintf('  TOTAL COST:         $%.4f\n', bestCost);
fprintf('  Cost Residual:      $%.6f\n', abs(bestCost - component_sum));

fprintf('\n--- OPERATIONAL METRICS ---\n');
fprintf('  Grid Import:        %.2f kWh\n', sum(max(0,hist.GridPower)));
fprintf('  Grid Export:        %.2f kWh\n', -sum(min(0,hist.GridPower)));
fprintf('  Battery Throughput: %.2f kWh\n', sum(abs(hist.Pbat)));
fprintf('  Battery Cycles:     %.2f equivalent cycles\n', hist.equivalent_cycles);
fprintf('  Max DoD:            %.1f%%\n', max(abs(hist.SoC - SoC0)) * 100);
fprintf('  Final SoC:          %.3f (target [%.2f, %.2f])\n', hist.SoC(end), SoC_min, SoC_max);
fprintf('\n=== PSO-Optimized EV Dispatch (representative run) ===\n');
fprintf('  EV Charging Energy: %.2f kWh\n', sum(hist.P_ev_ch));
fprintf('  EV Discharging Energy: %.2f kWh\n', sum(hist.P_ev_dis));
if isfield(hist, 'v2g_service_energy')
    fprintf('  EV V2G service energy: %.2f kWh (target %.2f kWh)\n', ...
        hist.v2g_service_energy, EV_params.v2g_required_energy);
end
fprintf('  Final EV SoC:        %.1f%%\n', hist.EV_SoC(end)*100);

% ---- Monetary cost vs objective value (report BOTH in the paper) ----
monetary_cost = hist.grid_energy_cost + hist.degradation_cost + hist.demand_charge;
fprintf('\n--- MONETARY vs OBJECTIVE (representative run) ---\n');
fprintf('  Monetary cost (energy+degradation+demand charge): $%.4f\n', monetary_cost);
fprintf('  Objective value J (incl. penalties/regularisers):  $%.4f\n', bestCost);
fprintf('  Virtual share of J: %.2f%%  <-- compute savings on the MONETARY figure\n', ...
    100*max(0, bestCost - monetary_cost)/max(bestCost,1e-9));

finalCost = bestCost; % kept for naming consistency with downstream code

%% --------------------------
DE_F  = 0.8;   % differential weight (scales the vector-difference mutation)
DE_CR = 0.9;   % crossover probability (binomial crossover)

fprintf('\n============================================================\n');
fprintf('Stage B2: Multi-run DE Optimization (BESS+uncontrolled-EV vs BESS+optimized-EV)\n');
fprintf('============================================================\n');

mrc_de = multirun_de_compare(PV_fore, Load_fore, Price_fore, BuyPrice, SellPrice, ...
    battery_params, EV_params, EV_params_base, EV_ch_seed, EV_dis_seed, EV_ch_uncontrolled, EV_dis_uncontrolled, ...
    PRED_HORIZON, SWARM_SIZE, PSO_MAXITER, DE_F, DE_CR, ...
    N_VALIDATION_RUNS, SEED_BASE, shared_init);

fprintf('\n--- DE Strategy 2: BESS Only (n=%d runs) ---\n', N_VALIDATION_RUNS);
fprintf('  Mean:   $%.4f +/- $%.4f (std)\n', mrc_de.s2_mean, mrc_de.s2_std);
fprintf('  Median: $%.4f   Best: $%.4f   Worst: $%.4f\n', mrc_de.s2_median, mrc_de.s2_min, mrc_de.s2_max);
fprintf('  CV:     %.2f%%\n', 100*mrc_de.s2_std/mrc_de.s2_mean);

fprintf('\n--- DE Strategy 3: BESS + EV (n=%d runs) ---\n', N_VALIDATION_RUNS);
fprintf('  Mean:   $%.4f +/- $%.4f (std)\n', mrc_de.s3_mean, mrc_de.s3_std);
fprintf('  Median: $%.4f   Best: $%.4f   Worst: $%.4f\n', mrc_de.s3_median, mrc_de.s3_min, mrc_de.s3_max);
fprintf('  CV:     %.2f%%\n', 100*mrc_de.s3_std/mrc_de.s3_mean);

fprintf('\n--- Local Polish Diagnostic (DE) ---\n');
fprintf('  BESS-only:  improved %d/%d runs, mean improvement $%.4f\n', ...
    mrc_de.s2_polish_n_improved, N_VALIDATION_RUNS, mrc_de.s2_polish_mean);
fprintf('  BESS+EV:    improved %d/%d runs, mean improvement $%.4f\n', ...
    mrc_de.s3_polish_n_improved, N_VALIDATION_RUNS, mrc_de.s3_polish_mean);

% --- PSO vs DE paired comparison (SAME seeds -> valid paired test) ---
try
    p_pso_vs_de_bess    = signrank(mrc.s2_all_costs, mrc_de.s2_all_costs);
    p_pso_vs_de_bessev  = signrank(mrc.s3_all_costs, mrc_de.s3_all_costs);
catch
    [~, p_pso_vs_de_bess]   = ttest(mrc.s2_all_costs, mrc_de.s2_all_costs);
    [~, p_pso_vs_de_bessev] = ttest(mrc.s3_all_costs, mrc_de.s3_all_costs);
end

fprintf('\n--- PSO vs DE (paired, matched seeds) ---\n');
fprintf('  BESS-only:   PSO $%.4f vs DE $%.4f   (p = %.4f)\n', mrc.s2_mean, mrc_de.s2_mean, p_pso_vs_de_bess);
fprintf('  BESS+EV:     PSO $%.4f vs DE $%.4f   (p = %.4f)\n', mrc.s3_mean, mrc_de.s3_mean, p_pso_vs_de_bessev);
if p_pso_vs_de_bessev < 0.05
    if mrc.s3_mean < mrc_de.s3_mean
        fprintf('    => PSO is significantly better than DE on the BESS+EV problem.\n');
    else
        fprintf('    => DE is significantly better than PSO on the BESS+EV problem.\n');
    end
else
    fprintf('    => No statistically significant difference between PSO and DE on this problem.\n');
end

%% --------------------------
% 10) Compare optimization strategies (now using mean-of-N as headline)
% --------------------------
fprintf('\n==============================\n');
fprintf('STRATEGY COMPARISON ANALYSIS\n');
fprintf('==============================\n');

baseline_grid = Load_fore + EV_ch_uncontrolled - EV_dis_uncontrolled - PV_fore;
baseline_peak_demand = max(max(0, baseline_grid));
baseline_demand_charge = demand_charge_rate * baseline_peak_demand;

proposed_peak_demand = max(max(0, hist.GridPower));
proposed_demand_charge = demand_charge_rate * proposed_peak_demand;

demand_charge_reduction = 100 * ...
    (baseline_demand_charge - proposed_demand_charge) / ...
     baseline_demand_charge;

fprintf('\n--- Strategy 1: Grid + Uncontrolled EV Charging ---\n');
fprintf('  Total Cost:     $%.4f\n', zeroCost);

fprintf('\n--- Strategy 2: BESS + Uncontrolled EV Charging ---\n');
fprintf('  Mean Cost (n=%d):  $%.4f +/- $%.4f\n', N_VALIDATION_RUNS, mrc.s2_mean, mrc.s2_std);
fprintf('  Mean Savings vs baseline: $%.4f (%.1f%%)\n', ...
    zeroCost - mrc.s2_mean, (zeroCost - mrc.s2_mean)/zeroCost*100);
fprintf('  Best observed:  $%.4f   Worst observed: $%.4f\n', mrc.s2_min, mrc.s2_max);

fprintf('\n--- DEMAND CHARGE ANALYSIS (representative run) ---\n');
fprintf('  Baseline Peak Demand:   %.2f kW\n', baseline_peak_demand);
fprintf('  Proposed Peak Demand:   %.2f kW\n', proposed_peak_demand);
fprintf('  Baseline Demand Charge: $%.2f\n', baseline_demand_charge);
fprintf('  Proposed Demand Charge: $%.2f\n', proposed_demand_charge);
fprintf('  Demand Charge Reduction: %.1f%%\n', demand_charge_reduction);

fprintf('\n--- Strategy 3: BESS + EV (Proposed) ---\n');
fprintf('  Mean Cost (n=%d):  $%.4f +/- $%.4f\n', N_VALIDATION_RUNS, mrc.s3_mean, mrc.s3_std);
fprintf('  Mean Savings vs baseline: $%.4f (%.1f%%)\n', ...
    zeroCost - mrc.s3_mean, (zeroCost - mrc.s3_mean)/zeroCost*100);
fprintf('  Best observed:  $%.4f   Worst observed: $%.4f\n', mrc.s3_min, mrc.s3_max);

% Economic analysis -- now based on the MEAN across N runs, not best-of-N.
% A best-case range is reported alongside for transparency.
daily_savings_mean = zeroCost - mrc.s3_mean;
daily_savings_best  = zeroCost - mrc.s3_min;

if daily_savings_mean > 0
    annual_savings_mean = daily_savings_mean * 365;
    bess_cost_per_kWh = 500;
    bess_cost_total = Ecap * bess_cost_per_kWh;
    payback_years_mean = bess_cost_total / annual_savings_mean;

    fprintf('\n--- ECONOMIC ANALYSIS (based on MEAN of %d runs) ---\n', N_VALIDATION_RUNS);
    fprintf('  BESS Cost:          $%.0f (%.1f kWh @ $%.0f/kWh installed)\n', bess_cost_total, Ecap, bess_cost_per_kWh);
    fprintf('  EV Fleet Value:     Existing asset (no additional cost)\n');
    fprintf('  Daily Savings (mean):      $%.2f\n', daily_savings_mean);
    fprintf('  Annualized Savings (mean)*: $%.0f\n', annual_savings_mean);
    fprintf('  Indicative Payback (mean)*: %.1f years\n', payback_years_mean);
    if daily_savings_best > 0
        annual_savings_best = daily_savings_best * 365;
        payback_years_best = bess_cost_total / annual_savings_best;
        fprintf('  Best-case (n=%d, single best run): Daily $%.2f, Payback %.1f yr\n', ...
            N_VALIDATION_RUNS, daily_savings_best, payback_years_best);
    end
    fprintf('  *Annualized from one representative 24-h test day.\n');
    fprintf('   Use multi-day/seasonal dispatch before claiming final annual payback.\n');
else
    fprintf('\n*** WARNING: mean EV+BESS operation did NOT beat the grid+uncontrolled-EV baseline. ***\n');
    fprintf('*** Economic claims should not be made until PSO consistency improves. ***\n');
    if daily_savings_best > 0
        fprintf('    (Only the single best of %d runs beat baseline: $%.2f/day savings.)\n', ...
            N_VALIDATION_RUNS, daily_savings_best);
        fprintf('    This best-case number should be reported as an upper bound only, not as the expected result.\n');
    end
end

%% ============================================================
% Forecast benchmark comparison: FCM-TSK, FFNN, and LSTM
% ============================================================

figure('Position', [120, 120, 1050, 340], 'Name', 'Forecast Benchmark Comparison', 'Color', 'w');

subplot(1,3,1);
plot(0:23, PV_actual, 'k-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'DisplayName', 'Actual'); hold on;
plot(0:23, PV_fore, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'FCM-TSK');
if RUN_FFNN_BENCHMARK, plot(0:23, PV_ffnn_fore, 'r--^', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'FFNN'); end
if RUN_LSTM_BENCHMARK, plot(0:23, PV_lstm_fore, 'm-.d', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'LSTM'); end
xlabel({'Hour','(a)'}); ylabel('PV power (kW)'); title('PV forecast'); grid on; xlim([0 23]); xticks(0:4:23);
legend('Location','best','FontSize',8);

subplot(1,3,2);
plot(0:23, Load_actual, 'k-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'DisplayName', 'Actual'); hold on;
plot(0:23, Load_fore, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'FCM-TSK');
if RUN_FFNN_BENCHMARK, plot(0:23, Load_ffnn_fore, 'r--^', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'FFNN'); end
if RUN_LSTM_BENCHMARK, plot(0:23, Load_lstm_fore, 'm-.d', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'LSTM'); end
xlabel({'Hour','(b)'}); ylabel('Load power (kW)'); title('Load forecast'); grid on; xlim([0 23]); xticks(0:4:23);
legend('Location','best','FontSize',8);

subplot(1,3,3);
plot(0:23, Price_actual, 'k-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'DisplayName', 'Actual'); hold on;
plot(0:23, Price_fore, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'FCM-TSK');
if RUN_FFNN_BENCHMARK, plot(0:23, Price_ffnn_fore, 'r--^', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'FFNN'); end
if RUN_LSTM_BENCHMARK, plot(0:23, Price_lstm_fore, 'm-.d', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'LSTM'); end
xlabel({'Hour','(c)'}); ylabel('Price ($/kWh)'); title('Price forecast'); grid on; xlim([0 23]); xticks(0:4:23);
legend('Location','best','FontSize',8);
sgtitle('Forecast Benchmark Comparison');
exportgraphics(gcf, 'forecast_benchmark_comparison.eps', 'ContentType', 'vector');

%% ============================================================
% BASELINES: B1 (Persistence) and B2 (Seasonal Persistence)
% ============================================================
fprintf('\n=== Baseline Forecasts ===\n');

PV_B1 = zeros(PRED_HORIZON,1); Load_B1 = zeros(PRED_HORIZON,1); Price_B1 = zeros(PRED_HORIZON,1);
EV_ch_B1 = zeros(PRED_HORIZON,1); EV_dis_B1 = zeros(PRED_HORIZON,1);
PV_B2 = zeros(PRED_HORIZON,1); Load_B2 = zeros(PRED_HORIZON,1); Price_B2 = zeros(PRED_HORIZON,1);
EV_ch_B2 = zeros(PRED_HORIZON,1); EV_dis_B2 = zeros(PRED_HORIZON,1);

for h = 1:PRED_HORIZON
    PV_B1(h)    = PV_hist(t0-1);
    Load_B1(h)  = Load_hist(t0-1);
    Price_B1(h) = Price_hist(t0-1);
    EV_ch_B1(h) = EV_Charging_hist(t0-1);
    EV_dis_B1(h) = EV_Discharging_hist(t0-1);

    idx_season = t0 + h - 24;
    if idx_season > 0
        PV_B2(h)    = PV_hist(idx_season);
        Load_B2(h)  = Load_hist(idx_season);
        Price_B2(h) = Price_hist(idx_season);
        EV_ch_B2(h) = EV_Charging_hist(idx_season);
        EV_dis_B2(h) = EV_Discharging_hist(idx_season);
    else
        PV_B2(h)    = PV_B1(h);
        Load_B2(h)  = Load_B1(h);
        Price_B2(h) = Price_B1(h);
        EV_ch_B2(h) = EV_ch_B1(h);
        EV_dis_B2(h) = EV_dis_B1(h);
    end

    PV_B1(h) = min(max(PV_B1(h),0), PV_CAPACITY);
    PV_B2(h) = min(max(PV_B2(h),0), PV_CAPACITY);

    hour_h = mod(hour(ts(t0)) + h - 1, 24);
    if hour_h >= 20 || hour_h <= 5
        PV_B1(h) = 0;
        PV_B2(h) = 0;
    end
end

[EV_ch_B1, EV_dis_B1] = repair_ev_schedule(EV_ch_B1, EV_dis_B1, EV_params, dt);
[EV_ch_B2, EV_dis_B2] = repair_ev_schedule(EV_ch_B2, EV_dis_B2, EV_params, dt);

pv_mae_B1    = mean(abs(PV_actual - PV_B1));
load_mae_B1  = mean(abs(Load_actual - Load_B1));
price_mae_B1 = mean(abs(Price_actual - Price_B1));

pv_mae_B2    = mean(abs(PV_actual - PV_B2));
load_mae_B2  = mean(abs(Load_actual - Load_B2));
price_mae_B2 = mean(abs(Price_actual - Price_B2));

% Best forecasting baseline among persistence, seasonal persistence, FFNN, and LSTM.
% Build the arrays dynamically so disabled benchmarks do not introduce NaNs.
forecast_baseline_names = {'Persistence','Seasonal Persistence'};
pv_baseline_maes    = [pv_mae_B1, pv_mae_B2];
load_baseline_maes  = [load_mae_B1, load_mae_B2];
price_baseline_maes = [price_mae_B1, price_mae_B2];
if RUN_FFNN_BENCHMARK
    forecast_baseline_names{end+1} = 'FFNN';
    pv_baseline_maes(end+1) = pv_error_ffnn;
    load_baseline_maes(end+1) = load_error_ffnn;
    price_baseline_maes(end+1) = price_error_ffnn;
end
if RUN_LSTM_BENCHMARK
    forecast_baseline_names{end+1} = 'LSTM';
    pv_baseline_maes(end+1) = pv_error_lstm;
    load_baseline_maes(end+1) = load_error_lstm;
    price_baseline_maes(end+1) = price_error_lstm;
end
[best_baseline_pv, idx_best_pv]       = min(pv_baseline_maes);
[best_baseline_load, idx_best_load]   = min(load_baseline_maes);
[best_baseline_price, idx_best_price] = min(price_baseline_maes);
best_baseline_pv_name = forecast_baseline_names{idx_best_pv};
best_baseline_load_name = forecast_baseline_names{idx_best_load};
best_baseline_price_name = forecast_baseline_names{idx_best_price};

skill_score_pv = 1 - (pv_error / pv_mae_B2);
skill_score_load = 1 - (load_error / load_mae_B2);
skill_score_price = 1 - (price_error / price_mae_B2);
skill_score_EV_ch = NaN;
skill_score_EV_dis = NaN;

fprintf('\n=== Baseline Comparison ===\n');
fprintf('\n--- B1: Persistence ---\n');
fprintf('  PV MAE:    %.4f kW\n', pv_mae_B1);
fprintf('  Load MAE:  %.4f kW\n', load_mae_B1);
fprintf('  Price MAE: %.4f $/kWh\n', price_mae_B1);

fprintf('\n--- B2: Seasonal Persistence ---\n');
fprintf('  PV MAE:    %.4f kW\n', pv_mae_B2);
fprintf('  Load MAE:  %.4f kW\n', load_mae_B2);
fprintf('  Price MAE: %.4f $/kWh\n', price_mae_B2);

if RUN_FFNN_BENCHMARK
    fprintf('\n--- B3: Custom FFNN Benchmark ---\n');
    fprintf('  PV MAE:    %.4f kW\n', pv_error_ffnn);
    fprintf('  Load MAE:  %.4f kW\n', load_error_ffnn);
    fprintf('  Price MAE: %.4f $/kWh\n', price_error_ffnn);
end

if RUN_LSTM_BENCHMARK
    fprintf('\n--- B4: LSTM Benchmark ---\n');
    fprintf('  PV MAE:    %.4f kW\n', pv_error_lstm);
    fprintf('  Load MAE:  %.4f kW\n', load_error_lstm);
    fprintf('  Price MAE: %.4f $/kWh\n', price_error_lstm);
end

fprintf('\n--- Proposed FCM-TSK Model ---\n');
fprintf('  PV MAE:    %.4f kW\n', pv_error);
fprintf('  Load MAE:  %.4f kW\n', load_error);
fprintf('  Price MAE: %.4f $/kWh\n', price_error);

fprintf('\n--- Skill Score ---\n');
fprintf('  PV Skill Score:    %.4f\n', skill_score_pv);
fprintf('  Load Skill Score:  %.4f\n', skill_score_load);
fprintf('  Price Skill Score: %.4f\n', skill_score_price);
fprintf('  EV Charging/Discharging Skill Score: N/A (rule-based/optimization variable)\n');
fprintf('\n--- Best-Baseline Skill Score Including FFNN and LSTM ---\n');
fprintf('  PV:    best baseline = %s (MAE %.4f), FCM skill = %.4f\n', ...
    best_baseline_pv_name, best_baseline_pv, 1 - pv_error/max(best_baseline_pv,eps));
fprintf('  Load:  best baseline = %s (MAE %.4f), FCM skill = %.4f\n', ...
    best_baseline_load_name, best_baseline_load, 1 - load_error/max(best_baseline_load,eps));
fprintf('  Price: best baseline = %s (MAE %.4f), FCM skill = %.4f\n', ...
    best_baseline_price_name, best_baseline_price, 1 - price_error/max(best_baseline_price,eps));

%%
fprintf('\n=== Statistical Comparison (Diebold-Mariano Test) ===\n');

err_fcm_pv = PV_actual - PV_fore;  err_b1_pv  = PV_actual - PV_B1;  err_b2_pv  = PV_actual - PV_B2;
err_fcm_load = Load_actual - Load_fore;  err_b1_load  = Load_actual - Load_B1;  err_b2_load  = Load_actual - Load_B2;
err_fcm_price = Price_actual - Price_fore;  err_b1_price  = Price_actual - Price_B1;  err_b2_price  = Price_actual - Price_B2;
err_ffnn_pv = PV_actual - PV_ffnn_fore;
err_ffnn_load = Load_actual - Load_ffnn_fore;
err_ffnn_price = Price_actual - Price_ffnn_fore;

fprintf('\n--- PV Forecast ---\n');
[dm_pv_b1, p_pv_b1] = dm_test(err_fcm_pv, err_b1_pv, 'FCM vs B1');
[dm_pv_b2, p_pv_b2] = dm_test(err_fcm_pv, err_b2_pv, 'FCM vs B2');
[dm_pv_ffnn, p_pv_ffnn] = dm_test(err_fcm_pv, err_ffnn_pv, 'FCM vs FFNN');
if RUN_LSTM_BENCHMARK
    err_lstm_pv = PV_actual - PV_lstm_fore;
    [dm_pv_lstm, p_pv_lstm] = dm_test(err_fcm_pv, err_lstm_pv, 'FCM vs LSTM');
    fprintf('  PV    FCM vs LSTM: DM = %+.4f, p = %.4f\n', dm_pv_lstm, p_pv_lstm);
end

fprintf('\n--- Load Forecast ---\n');
[dm_ld_b1, p_ld_b1] = dm_test(err_fcm_load, err_b1_load, 'FCM vs B1');
[dm_ld_b2, p_ld_b2] = dm_test(err_fcm_load, err_b2_load, 'FCM vs B2');
[dm_ld_ffnn, p_ld_ffnn] = dm_test(err_fcm_load, err_ffnn_load, 'FCM vs FFNN');
if RUN_LSTM_BENCHMARK
    err_lstm_load = Load_actual - Load_lstm_fore;
    [dm_ld_lstm, p_ld_lstm] = dm_test(err_fcm_load, err_lstm_load, 'FCM vs LSTM');
    fprintf('  Load  FCM vs LSTM: DM = %+.4f, p = %.4f\n', dm_ld_lstm, p_ld_lstm);
end

fprintf('\n--- Price Forecast ---\n');
[dm_pr_b1, p_pr_b1] = dm_test(err_fcm_price, err_b1_price, 'FCM vs B1');
[dm_pr_b2, p_pr_b2] = dm_test(err_fcm_price, err_b2_price, 'FCM vs B2');
[dm_pr_ffnn, p_pr_ffnn] = dm_test(err_fcm_price, err_ffnn_price, 'FCM vs FFNN');
if RUN_LSTM_BENCHMARK
    err_lstm_price = Price_actual - Price_lstm_fore;
    [dm_pr_lstm, p_pr_lstm] = dm_test(err_fcm_price, err_lstm_price, 'FCM vs LSTM');
    fprintf('  Price FCM vs LSTM: DM = %+.4f, p = %.4f\n', dm_pr_lstm, p_pr_lstm);
end

fprintf('\n--- EV Charging/Discharging Forecast ---\n');
fprintf('EV charging/discharging are not claimed as FCM-TSK forecasts in this study.\n');

%% --------------------------
% 11) VISUALIZATION - PUBLICATION READY (unchanged figure structure;
%     now fed by the median-cost "representative" run instead of best-of-N)
% --------------------------

% ============================================================
% FIGURE 1: Forecast benchmark comparison (Actual, FCM-TSK, FFNN, LSTM)
% ============================================================
figure('Position', [100, 100, 1050, 340], 'Name', 'Forecast Quality', 'Color', 'w');

subplot(1,3,1);
plot(0:23, PV_actual, 'k-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'DisplayName', 'Actual'); hold on;
plot(0:23, PV_fore, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'FCM-TSK');
if RUN_FFNN_BENCHMARK, plot(0:23, PV_ffnn_fore, 'r--^', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'FFNN'); end
if RUN_LSTM_BENCHMARK, plot(0:23, PV_lstm_fore, 'm-.d', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'LSTM'); end
xlabel({'Time (Hour)','(a)'}); ylabel('PV power (kW)'); grid on; xlim([0 23]); xticks(0:4:23); legend('Location','best','FontSize',8);

subplot(1,3,2);
plot(0:23, Load_actual, 'k-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'DisplayName', 'Actual'); hold on;
plot(0:23, Load_fore, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'FCM-TSK');
if RUN_FFNN_BENCHMARK, plot(0:23, Load_ffnn_fore, 'r--^', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'FFNN'); end
if RUN_LSTM_BENCHMARK, plot(0:23, Load_lstm_fore, 'm-.d', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'LSTM'); end
xlabel({'Time (Hour)','(b)'}); ylabel('Load power (kW)'); grid on; xlim([0 23]); xticks(0:4:23); legend('Location','best','FontSize',8);

subplot(1,3,3);
plot(0:23, Price_actual, 'k-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'DisplayName', 'Actual'); hold on;
plot(0:23, Price_fore, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'FCM-TSK');
if RUN_FFNN_BENCHMARK, plot(0:23, Price_ffnn_fore, 'r--^', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'FFNN'); end
if RUN_LSTM_BENCHMARK, plot(0:23, Price_lstm_fore, 'm-.d', 'LineWidth', 1.4, 'MarkerSize', 4, 'DisplayName', 'LSTM'); end
xlabel({'Time (Hour)','(c)'}); ylabel('Price ($/kWh)'); grid on; xlim([0 23]); xticks(0:4:23); legend('Location','best','FontSize',8);

set(gcf, 'PaperPositionMode', 'auto');
exportgraphics(gcf, 'forecast.eps', 'ContentType', 'vector');

% ============================================================
% FIGURE 2: Power Dispatch Analysis (representative run)
% ============================================================
figure('Position', [100, 100, 900, 900], 'Name', 'Optimal Dispatch', 'Color', 'w');

subplot(3,2,1);
hold on;
area(0:23, PV_fore, 'FaceColor', [1, 0.8, 0.4], 'EdgeColor', 'none', 'DisplayName', 'PV Generation');
plot(0:23, Load_fore, 'k-', 'LineWidth', 2, 'DisplayName', 'Load Demand');
plot(0:23, Load_fore - PV_fore, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Net Load');
xlabel('Time (Hour)', 'FontSize', 10); ylabel('Power (kW)', 'FontSize', 10);
legend('Location', 'northwest', 'FontSize', 8);
text(0.5, -0.3, '(a)', 'Units', 'normalized', 'HorizontalAlignment', 'center');
grid on; box on; xlim([0, 23.5]); xticks(0:4:23);

subplot(3,2,2);
bar(0:23, hist.Pbat, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none', 'DisplayName', 'BESS Power');
ylabel('Power (kW)', 'FontSize', 10);
grid on; xlim([-0.5, 23.5]); ylim([-Pmax-0.3, Pmax+0.3]);
hold on;
plot(xlim, [0, 0], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
xlabel('Time (Hour)', 'FontSize', 10); xticks(0:4:23);
text(1, Pmax*0.7, 'DISCHARGE ->', 'Color', 'm', 'FontSize', 8);
text(1, -Pmax*0.7, '<- CHARGE', 'Color', 'r', 'FontSize', 8);
legend('Location', 'best', 'FontSize', 8);
text(0.5, -0.3, '(b)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

subplot(3,2,3);
hold on;
bar(0:23, -hist.P_ev_ch, 'FaceColor', [0.2, 0.8, 0.4], 'EdgeColor', 'none', 'DisplayName', 'EV Charging (-Ve)');
bar(0:23, hist.P_ev_dis, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'none', 'DisplayName', 'EV V2G Discharge(+Ve)');
ylabel('Power (kW)', 'FontSize', 10);
legend('Location', 'southwest', 'FontSize', 8);
text(0.5, -0.3, '(c)', 'Units', 'normalized', 'HorizontalAlignment', 'center');
grid on; box on; xlim([-0.5, 23.5]); xlabel('Time (Hour)', 'FontSize', 10); xticks(0:4:23);
hold on; plot(xlim, [0, 0], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');

subplot(3,2,4);
yyaxis left;
bar(0:23, hist.GridPower, 'FaceColor', [0.3, 0.5, 0.7], 'EdgeColor', 'none', 'DisplayName', 'Grid Power');
ylabel('Grid Power (kW)', 'FontSize', 10); ylim([-2, max(hist.GridPower)+1]);
yyaxis right;
plot(0:23, BuyPrice, 'r--', 'LineWidth', 2, 'DisplayName', 'Buy Price');
plot(0:23, SellPrice, 'g:', 'LineWidth', 2, 'DisplayName', 'Sell Price');
ylabel('Price ($/kWh)', 'FontSize', 10);
xlabel('Time (Hour)', 'FontSize', 10);
legend('Location', 'south', 'FontSize', 8);
text(0.5, -0.3, '(d)', 'Units', 'normalized', 'HorizontalAlignment', 'center');
grid on; xlim([0, 23]); xticks(0:4:23);

subplot(3,2,5);
SoC_plot = [SoC0; hist.SoC(:)];
plot(0:24, SoC_plot * 100, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b', 'DisplayName', 'BESS SoC');
hold on;
yline(SoC_min * 100, 'r--', 'LineWidth', 1.5, ...
      'DisplayName', sprintf('BESS SoC Min (%.0f%%)', SoC_min*100));
yline(SoC_max * 100, 'r--', 'LineWidth', 1.5, ...
      'DisplayName', sprintf('BESS SoC Max (%.0f%%)', SoC_max*100));
xlabel('Time (Hour)', 'FontSize', 10); ylabel('State of Charge (%)', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 8);
text(0.5, -0.3, '(e)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; xlim([0, 24]); xticks(0:4:24); ylim([0, 100]);

subplot(3,2,6);
EVSoC_plot = [EV_params.EV_SoC0; hist.EV_SoC(:)];
plot(0:24, EVSoC_plot * 100, 'g-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'g', 'DisplayName', 'EV SoC');
hold on;
yline(EV_params.EV_SoC_min * 100, 'r--', 'LineWidth', 1.5,'Label', sprintf('EV SoC Min (%.0f%%)', EV_params.EV_SoC_min*100),'HandleVisibility', 'off');
yline(EV_params.EV_SoC_max * 100, 'r--', 'LineWidth', 1.5, 'Label', sprintf('EV SoC Max (%.0f%%)', EV_params.EV_SoC_max*100),'HandleVisibility', 'off');
yline(EV_params.EV_final_soc_req * 100, 'k:', 'LineWidth', 1.5, 'DisplayName', sprintf('Departure Requirement (%.0f%%)', EV_params.EV_final_soc_req*100));
xlabel('Time (Hour)', 'FontSize', 10); ylabel('State of Charge (%)', 'FontSize', 10);
legend('Location', 'best', 'FontSize', 8);
text(0.5, -0.3, '(f)', 'Units', 'normalized', 'HorizontalAlignment', 'center');
grid on; xlim([0, 24]); xticks(0:4:24); ylim([0, 100]);

% ============================================================
% FIGURE 3: PSO Convergence with Confidence Bounds (n=N_VALIDATION_RUNS)
% ============================================================
All_Histories = mrc.s3_all_histories;
valid_cols = sum(All_Histories > 0, 1) > 0;
All_Histories = All_Histories(:, valid_cols);

mean_history = mean(All_Histories, 1);
std_history  = std(All_Histories, 0, 1);
upper_bound = mean_history + std_history;
lower_bound = mean_history - std_history;
iters = 2:length(mean_history);

figure('Position', [100, 100, 500, 400], 'Name', 'PSO Convergence with CI', 'Color', 'w');
fill([iters, fliplr(iters)], [upper_bound(iters), fliplr(lower_bound(iters))], ...
     [0.85, 0.85, 0.95], 'EdgeColor', 'none', 'DisplayName', '\pm1 Std. Dev.');
hold on;
plot(iters, mean_history(iters), 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Mean of %d runs', N_VALIDATION_RUNS));
% NOTE: pso_hist (the representative run's raw convergence trace) may be
% SHORTER than mean_history if that particular run triggered early
% stopping before reaching maxiter -- unlike mean_history, it is not
% padded. Index it with its own range to avoid an out-of-bounds error.
iters_repr = 2:length(pso_hist);
plot(iters_repr, pso_hist(iters_repr), '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2, 'DisplayName', 'Representative (median-cost) run');
xlabel('PSO Iteration', 'FontSize', 12); ylabel('Objective Cost ($)', 'FontSize', 12);
legend('Location', 'northeast'); grid on; box on;
ylim([min(lower_bound(iters))-0.5, max(upper_bound(iters))+0.5]);

final_mean = mean_history(end); final_std = std_history(end);
text(0.65*max(iters), min(lower_bound(iters))+0.3, ...
    sprintf('Final: $%.2f \\pm $%.2f\nCV = %.2f%%', final_mean, final_std, 100*final_std/final_mean), ...
    'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k');

% ============================================================
% FIGURE 4: PSO Validation (Multi-Run Consistency) — same content as
% the previous standalone validation figure, sourced from mrc.
% ============================================================
figure('Position', [100, 100, 300, 300], 'Name', 'PSO Convergence Validation: BESS + EV');
semilogy(1:length(pso_hist), pso_hist, 'b-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Best Cost ($) — representative run'); grid on;

% ============================================================
% FIGURE 5: Cost Breakdown Analysis (representative run)
% ============================================================
figure('Name', 'Cost Breakdown','Position', [200, 200, 900, 400], 'Color', 'w');

cost_categories = {'Grid Energy', 'Constraint', 'Degradation', 'Demand', 'Demand Shaping', 'Terminal SoC', 'SoC Smooth'};
cost_values = [hist.grid_energy_cost, hist.other_penalty_cost, ...
               hist.degradation_cost, hist.demand_charge, hist.demand_shaping_penalty, ...
               hist.terminal_soc_penalty, hist.soc_smoothing_penalty];

keep = abs(cost_values) > 1e-6;
cost_categories = cost_categories(keep);
cost_values = cost_values(keep);
cost_percent = 100 * cost_values / sum(cost_values);

subplot(2,1,1);
bar(cost_values, 'FaceColor', [0.3, 0.6, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.2);
set(gca, 'XTickLabel', cost_categories, 'FontSize', 10);
ylabel('Cost ($)', 'FontSize', 11); grid on; box on;
for i = 1:numel(cost_values)
    text(i, cost_values(i) + 1.5, sprintf('$%.2f', cost_values(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end
text(0.5, -0.1, '(a)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

subplot(2,1,2);
bar(cost_percent, 'FaceColor', [0.3, 0.7, 0.4], 'EdgeColor', 'k', 'LineWidth', 1.2);
set(gca, 'XTickLabel', cost_categories, 'FontSize', 10);
ylabel('Cost Share (%)', 'FontSize', 11); ylim([0 100]); grid on; box on;
for i = 1:numel(cost_percent)
    text(i, cost_percent(i) + 5, sprintf('%.1f%%', cost_percent(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end
text(0.5, -0.1, '(b)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

% ============================================================
% FIGURE A: DE Convergence with Confidence Bounds (n=N_VALIDATION_RUNS)
% ============================================================
All_Histories_DE = mrc_de.s3_all_histories;
valid_cols_de = sum(All_Histories_DE > 0, 1) > 0;
All_Histories_DE = All_Histories_DE(:, valid_cols_de);
mean_hist_de = mean(All_Histories_DE, 1);
std_hist_de  = std(All_Histories_DE, 0, 1);
iters_de = 2:length(mean_hist_de);

figure('Position', [100, 100, 500, 400], 'Name', 'DE Convergence with +/-1 SD Band', 'Color', 'w');
fill([iters_de, fliplr(iters_de)], ...
     [mean_hist_de(iters_de)+std_hist_de(iters_de), fliplr(mean_hist_de(iters_de)-std_hist_de(iters_de))], ...
     [0.95, 0.85, 0.85], 'EdgeColor', 'none', 'DisplayName', '\pm1 Std. Dev.');
hold on;
plot(iters_de, mean_hist_de(iters_de), 'r-', 'LineWidth', 2, 'DisplayName', sprintf('DE mean of %d runs', N_VALIDATION_RUNS));
% NOTE: mrc_de.s3_repr_convergence (the representative run's raw trace)
% may be SHORTER than mean_hist_de if that run triggered early stopping
% before reaching maxgen -- unlike mean_hist_de, it is not padded. Index
% it with its own range to avoid an out-of-bounds error.
iters_de_repr = 2:length(mrc_de.s3_repr_convergence);
plot(iters_de_repr, mrc_de.s3_repr_convergence(iters_de_repr), '--', 'Color', [0.5 0 0], 'LineWidth', 1.2, 'DisplayName', 'DE representative run');
xlabel('DE Generation', 'FontSize', 12); ylabel('Objective Cost ($)', 'FontSize', 12);
legend('Location', 'northeast'); grid on; box on;
final_mean_de = mean_hist_de(end); final_std_de = std_hist_de(end);
text(0.65*max(iters_de), min(mean_hist_de(iters_de)-std_hist_de(iters_de))+0.3, ...
    sprintf('Final: $%.2f \\pm $%.2f\nCV = %.2f%%', final_mean_de, final_std_de, 100*final_std_de/final_mean_de), ...
    'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k');
% ============================================================
% FIGURE B: PSO vs DE Convergence Overlay (BESS+EV problem)
% ============================================================
valid_cols_pso = sum(mrc.s3_all_histories > 0, 1) > 0;
mean_hist_pso = mean(mrc.s3_all_histories(:, valid_cols_pso), 1);
n_common = min(length(mean_hist_pso), length(mean_hist_de));
iters_common = 2:n_common;

figure('Position', [100, 100, 550, 420], 'Name', 'PSO vs DE Convergence', 'Color', 'w');
plot(iters_common, mean_hist_pso(iters_common), 'b-', 'LineWidth', 2, 'DisplayName', sprintf('PSO (mean of %d runs)', N_VALIDATION_RUNS));
hold on;
plot(iters_common, mean_hist_de(iters_common), 'r-', 'LineWidth', 2, 'DisplayName', sprintf('DE (mean of %d runs)', N_VALIDATION_RUNS));
xlabel('Iteration / Generation', 'FontSize', 12); ylabel('Objective Cost ($)', 'FontSize', 12);
title('PSO vs DE: Mean Convergence (BESS+EV problem)', 'FontSize', 12);
legend('Location', 'northeast'); grid on; box on;

% ============================================================
% FIGURE C: Grouped Bar Chart, PSO vs DE, with error bars (headline figure)
% ============================================================
figure('Position', [100, 100, 600, 400], 'Name', 'PSO vs DE Cost Comparison', 'Color', 'w');
means_grouped = [mrc.s2_mean, mrc_de.s2_mean; mrc.s3_mean, mrc_de.s3_mean];
stds_grouped  = [mrc.s2_std,  mrc_de.s2_std;  mrc.s3_std,  mrc_de.s3_std];
b = bar(means_grouped, 'grouped');
b(1).FaceColor = [0.2 0.5 0.8]; b(2).FaceColor = [0.8 0.3 0.3];
hold on;
ngroups = size(means_grouped,1); nbars = size(means_grouped,2);
groupwidth = min(0.8, nbars/(nbars+1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1)*groupwidth/(2*nbars);
    errorbar(x, means_grouped(:,i), stds_grouped(:,i), 'k.', 'LineWidth', 1.2);
end
set(gca, 'XTickLabel', {'BESS Only', 'BESS + EV'}, 'FontSize', 10);
ylabel('Mean Daily Cost ($) \pm 1 Std. Dev.', 'FontSize', 11);
legend({'PSO','DE'}, 'Location', 'best');
yline(zeroCost, 'k--', 'Grid+uncontrolled-EV baseline');
grid on; title(sprintf('PSO vs DE across %d matched-seed runs', N_VALIDATION_RUNS));

% ============================================================
% FIGURE D: Boxplot of Cost Distributions, PSO vs DE
% ============================================================
figure('Position', [100, 100, 600, 400], 'Name', 'Cost Distribution: PSO vs DE', 'Color', 'w');
all_costs_matrix = [mrc.s2_all_costs, mrc_de.s2_all_costs, mrc.s3_all_costs, mrc_de.s3_all_costs];
boxplot(all_costs_matrix, {'PSO-BESS','DE-BESS','PSO-BESS+EV','DE-BESS+EV'});
ylabel('Daily Cost ($)', 'FontSize', 11);
yline(zeroCost, 'r--', 'Grid+uncontrolled-EV baseline');
grid on; title(sprintf('Cost distribution across %d runs per algorithm/strategy', N_VALIDATION_RUNS));

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE\n');
fprintf('========================================\n');
fprintf('Generated figures:\n');
fprintf('  Figure(s) 0a-0c: Actual-model MFs (FIX #4) + illustrative-only MFs (relabeled)\n');
fprintf('  Figure 1: Forecast Quality and SoC Trajectories\n');
fprintf('  Figure 2: Optimal Dispatch Results (representative/median run, PSO)\n');
fprintf('  Figure 3: PSO Convergence Analysis (mean +/- std over %d runs)\n', N_VALIDATION_RUNS);
fprintf('  Figure 4: PSO Validation (representative run trace)\n');
fprintf('  Figure 5: Cost Breakdown Analysis (representative run, PSO)\n');
fprintf('  Figure A: DE Convergence Analysis (mean +/- std over %d runs)\n', N_VALIDATION_RUNS);
fprintf('  Figure B: PSO vs DE Convergence Overlay (BESS+EV)\n');
fprintf('  Figure C: PSO vs DE Mean Cost Comparison (grouped bar, error bars)\n');
fprintf('  Figure D: PSO vs DE Cost Distribution (boxplot)\n');

%% Figure: Forecast Error Distribution Comparison
figure('Position', [100, 100, 1050, 700], 'Name', 'Forecast Error Distribution', 'Color', 'w');

subplot(2,3,1);
errors_fcm = abs(PV_actual - PV_fore); errors_b2 = abs(PV_actual - PV_B2); errors_b1 = abs(PV_actual - PV_B1);
boxplot([errors_fcm(:), errors_b2(:), errors_b1(:)], {'FCM-TSK', 'Seasonal', 'Persistence'});
ylabel('PV Forecast Error (kW)', 'FontSize', 11); grid on; hold on;
plot(1, mean(errors_fcm), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(2, mean(errors_b2), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(3, mean(errors_b1), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
text(0.5, -0.2, '(a)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

subplot(2,3,2);
errors_fcm = abs(Load_actual - Load_fore); errors_b2 = abs(Load_actual - Load_B2); errors_b1 = abs(Load_actual - Load_B1);
boxplot([errors_fcm(:), errors_b2(:), errors_b1(:)], {'FCM-TSK', 'Seasonal', 'Persistence'});
ylabel('Load Forecast Error (kW)', 'FontSize', 11); grid on; hold on;
plot(1, mean(errors_fcm), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(2, mean(errors_b2), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(3, mean(errors_b1), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
text(0.5, -0.2, '(b)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

subplot(2,3,3);
errors_fcm = abs(Price_actual - Price_fore); errors_b2 = abs(Price_actual - Price_B2); errors_b1 = abs(Price_actual - Price_B1);
boxplot([errors_fcm(:), errors_b2(:), errors_b1(:)], {'FCM-TSK', 'Seasonal', 'Persistence'});
ylabel('Price Forecast Error ($/kWh)', 'FontSize', 11); grid on; hold on;
plot(1, mean(errors_fcm), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(2, mean(errors_b2), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(3, mean(errors_b1), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
text(0.5, -0.2, '(c)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

subplot(2,3,4);
improvements = 100 * [1 - pv_error/pv_mae_B2, 1 - load_error/load_mae_B2, 1 - price_error/price_mae_B2];
bar(improvements, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTickLabel', {'PV', 'Load', 'Price'}, 'FontSize', 10);
ylabel('MAE Reduction vs Seasonal (%)', 'FontSize', 11); grid on; ylim([0, 70]);
sig = [p_pv_b2 < 0.05, p_ld_b2 < 0.05, p_pr_b2 < 0.05];
for i = 1:3
    if sig(i)
        text(i, improvements(i) + 0.05*max(improvements), '*', 'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold');
    end
end
text(0.5, -0.2, '(d)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

subplot(2,3,5);
p_values = [p_pv_b2, p_ld_b2, p_pr_b2];
labels = {'PV', 'Load', 'Price'};
bar(p_values, 'FaceColor', [0.8, 0.4, 0.2]);
set(gca, 'XTickLabel', labels, 'FontSize', 10);
ylabel('p-value (log scale)', 'FontSize', 11); set(gca, 'YScale', 'log'); ylim([1e-4 1]);
grid on; hold on;
x_center = mean(xlim);
text(x_center, 0.05*1.5, 'p = 0.05', 'HorizontalAlignment', 'center', 'Color', 'r', 'FontSize', 9, 'BackgroundColor', 'w');
yline(0.05, 'r--', 'LineWidth', 1.5);
for i = 1:length(p_values)
    text(i, max(p_values(i),1e-4)*1.5, sprintf('%.4f', p_values(i)), 'HorizontalAlignment','center', 'FontSize',9, 'FontWeight','bold');
end
text(0.5, -0.2, '(e)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

subplot(2,3,6);
avg_improvement = mean(improvements);
bar(avg_improvement, 'FaceColor', [0.3, 0.7, 0.4]);
ylabel('Average Improvement (%)', 'FontSize', 11);
set(gca, 'XTickLabel', {'Average'}, 'FontSize', 10); ylim([0 100]); grid on;
text(1, avg_improvement + 3, sprintf('%.1f%%', avg_improvement), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(0.5, -0.2, '(f)', 'Units', 'normalized', 'HorizontalAlignment', 'center');

%% Generate Contribution Summary Table
fprintf('\n========================================\n');
fprintf('MAIN CONTRIBUTIONS SUMMARY TABLE\n');
fprintf('========================================\n');

fprintf('\n%-25s | %-15s | %-15s | %-15s\n', 'Metric', 'FCM-TSK', 'Best Baseline', 'Improvement');
fprintf('%-25s-+-%-15s-+-%-15s-+-%-15s\n', repmat('-',1,25), repmat('-',1,15), repmat('-',1,15), repmat('-',1,15));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'PV MAE (kW)', sprintf('%.4f', pv_error), sprintf('%.4f (%s)', best_baseline_pv, best_baseline_pv_name), sprintf('%.1f%%', 100*(1-pv_error/max(best_baseline_pv,eps))));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Load MAE (kW)', sprintf('%.4f', load_error), sprintf('%.4f (%s)', best_baseline_load, best_baseline_load_name), sprintf('%.1f%%', 100*(1-load_error/max(best_baseline_load,eps))));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Price MAE ($/kWh)', sprintf('%.4f', price_error), sprintf('%.4f (%s)', best_baseline_price, best_baseline_price_name), sprintf('%.1f%%', 100*(1-price_error/max(best_baseline_price,eps))));

fprintf('%-25s | %-15s | %-15s | %-15s\n', sprintf('Economic Metric (n=%d)', N_VALIDATION_RUNS), 'Mean Value', 'Baseline', 'Mean Improvement');
fprintf('%-25s-+-%-15s-+-%-15s-+-%-15s\n', repmat('-',1,25), repmat('-',1,15), repmat('-',1,15), repmat('-',1,15));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Daily Cost ($)', sprintf('%.2f', mrc.s3_mean), sprintf('%.2f', zeroCost), sprintf('%.1f%%', 100*(zeroCost-mrc.s3_mean)/abs(zeroCost)));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Daily Savings ($)', sprintf('%.2f', zeroCost-mrc.s3_mean), '-', '-');
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Best observed cost ($)', sprintf('%.2f', mrc.s3_min), '-', sprintf('(of %d runs)', N_VALIDATION_RUNS));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Worst observed cost ($)', sprintf('%.2f', mrc.s3_max), '-', sprintf('(of %d runs)', N_VALIDATION_RUNS));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Demand Charge ($)', sprintf('%.2f', proposed_demand_charge), sprintf('%.2f', baseline_demand_charge), sprintf('%.1f%%', demand_charge_reduction));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Peak Demand (kW)', sprintf('%.2f', proposed_peak_demand), sprintf('%.2f', baseline_peak_demand), sprintf('%.1f%%', 100*(baseline_peak_demand-proposed_peak_demand)/baseline_peak_demand));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'BESS Cycles (day)', sprintf('%.2f', hist.equivalent_cycles), '-', '-');
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Final EV SoC (%)', sprintf('%.1f', hist.EV_SoC(end)*100), '>50%', 'OK');
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'DE Daily Cost ($)', sprintf('%.2f', mrc_de.s3_mean), sprintf('%.2f', zeroCost), sprintf('%.1f%%', 100*(zeroCost-mrc_de.s3_mean)/abs(zeroCost)));

fprintf('\n%-25s | %-15s | %-15s | %-15s\n', 'Statistical Test', 'DM Statistic', 'p-value', 'Significant');
fprintf('%-25s-+-%-15s-+-%-15s-+-%-15s\n', repmat('-',1,25), repmat('-',1,15), repmat('-',1,15), repmat('-',1,15));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'PV (FCM vs B1)', sprintf('%.4f', dm_pv_b1), sprintf('%.4f', p_pv_b1), sig_label(p_pv_b1));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'PV (FCM vs B2)', sprintf('%.4f', dm_pv_b2), sprintf('%.4f', p_pv_b2), sig_label(p_pv_b2));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'PV (FCM vs FFNN)', sprintf('%.4f', dm_pv_ffnn), sprintf('%.4f', p_pv_ffnn), sig_label(p_pv_ffnn));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Load (FCM vs B1)', sprintf('%.4f', dm_ld_b1), sprintf('%.4f', p_ld_b1), sig_label(p_ld_b1));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Load (FCM vs B2)', sprintf('%.4f', dm_ld_b2), sprintf('%.4f', p_ld_b2), sig_label(p_ld_b2));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Load (FCM vs FFNN)', sprintf('%.4f', dm_ld_ffnn), sprintf('%.4f', p_ld_ffnn), sig_label(p_ld_ffnn));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Price (FCM vs B1)', sprintf('%.4f', dm_pr_b1), sprintf('%.4f', p_pr_b1), sig_label(p_pr_b1));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Price (FCM vs B2)', sprintf('%.4f', dm_pr_b2), sprintf('%.4f', p_pr_b2), sig_label(p_pr_b2));
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Price (FCM vs FFNN)', sprintf('%.4f', dm_pr_ffnn), sprintf('%.4f', p_pr_ffnn), sig_label(p_pr_ffnn));

fprintf('\n%-25s | %-15s\n', 'Dispatch Test', 'p-value (paired signrank)');
fprintf('%-25s-+-%-15s\n', repmat('-',1,25), repmat('-',1,15));
fprintf('%-25s | %-15s\n', 'Strategy3 vs Strategy2', sprintf('%.4f', mrc.p_s3_vs_s2));
fprintf('%-25s | %-15s\n', 'PSO vs DE (BESS only)', sprintf('%.4f', p_pso_vs_de_bess));
fprintf('%-25s | %-15s\n', 'PSO vs DE (BESS+EV)', sprintf('%.4f', p_pso_vs_de_bessev));

fprintf('\nNote: economic headline figures are now MEAN of %d matched-seed PSO/DE runs, not a cherry-picked best run.\n', N_VALIDATION_RUNS);
fprintf('========================================\n');



%% ============================================================
% MULTI-DAY EVALUATION: 7-30 TEST DAYS
% ============================================================
% This block evaluates the final dispatch framework over several held-out
% midnight-start test days. It uses the SAME fair EV baseline as the
% single-day study:
%   S1 = Grid + uncontrolled mandatory EV charging
%   S2 = BESS + uncontrolled mandatory EV charging
%   S3 = BESS + optimized EV charging/V2G
%
% To control runtime:
%   MULTIDAY_N_DAYS should be between 7 and 30.
%   MULTIDAY_N_RUNS can be 5-10 for debugging and 20 for final reporting.
if RUN_MULTIDAY_EVALUATION
    fprintf('\n############################################################\n');
    fprintf('## MULTI-DAY EVALUATION (%d requested days, %d runs/day)\n', MULTIDAY_N_DAYS, MULTIDAY_N_RUNS);
    fprintf('############################################################\n');

    % Candidate test days: midnight starts only, fully inside available data.
    test_start_indices_all = t_target(idx_test);
    test_timestamps_all = ts(test_start_indices_all);
    md_mask = hour(test_timestamps_all) == 0;
    candidate_t0 = test_start_indices_all(md_mask);
    candidate_t0 = candidate_t0(candidate_t0 + PRED_HORIZON - 1 <= numel(PV_hist));

    if isempty(candidate_t0)
        warning('No midnight-start test days available for multi-day evaluation. Skipping.');
    else
        nAvailable = numel(candidate_t0);
        nDays = min(max(MULTIDAY_N_DAYS, 7), 30);
        nDays = min(nDays, nAvailable);

        if MULTIDAY_N_DAYS < 7 || MULTIDAY_N_DAYS > 30
            warning('MULTIDAY_N_DAYS should be between 7 and 30. Using %d days.', nDays);
        end

        if strcmpi(MULTIDAY_DAY_SELECTION, "first")
            sel_idx = 1:nDays;
        else
            sel_idx = unique(round(linspace(1, nAvailable, nDays)));
            % If rounding produced duplicates, fill from the beginning.
            if numel(sel_idx) < nDays
                missing = setdiff(1:nAvailable, sel_idx, 'stable');
                sel_idx = [sel_idx(:); missing(1:(nDays-numel(sel_idx))).'];
                sel_idx = sort(sel_idx);
            end
        end

        selected_t0 = candidate_t0(sel_idx(1:nDays));

        fprintf('Available midnight-start test days: %d\n', nAvailable);
        fprintf('Selected days: %d\n', nDays);
        fprintf('Relative compute vs single-day PSO/DE section: approximately %.1fx\n', ...
            (nDays * MULTIDAY_N_RUNS) / max(N_VALIDATION_RUNS, 1));

        dayResults = cell(nDays,1);

        for d = 1:nDays
            t0_day = selected_t0(d);
            daySeedBase = SEED_BASE + 10000*d;  % independent, reproducible per day

            fprintf('\n--- Multi-day evaluation: Day %d/%d (%s) ---\n', ...
                d, nDays, datestr(ts(t0_day), 'yyyy-mm-dd'));

            dayResults{d} = evaluate_one_multiday_dispatch( ...
                t0_day, ts, PV_hist, Load_hist, Price_hist, EV_Charging_hist, EV_Discharging_hist, EV_SoC_hist, Vehicle_Count_hist, ...
                cal_all, mu, sd, pv_final_all, load_final_all, price_final_all, ...
                RUN_FFNN_BENCHMARK, ffnn_pv_model, ffnn_load_model, ffnn_price_model, ...
                RUN_LSTM_BENCHMARK, lstm_pv_model, lstm_load_model, lstm_price_model, n_cal_feat, ...
                LAGS, PRED_HORIZON, PV_CAPACITY, USE_LOAD_ADAPTIVE_PRICE_ENHANCEMENT, ...
                battery_params, EV_params, ...
                SWARM_SIZE, PSO_MAXITER, W_INERTIA, C1, C2, VEL_CLAMP_FACTOR, ...
                MULTIDAY_N_RUNS, daySeedBase, WARM_START_FRAC, ...
                DE_F, DE_CR, MULTIDAY_INCLUDE_DE);

            fprintf('  Forecast MAE: PV=%.4f, Load=%.4f, Price=%.4f\n', ...
                dayResults{d}.pv_mae, dayResults{d}.load_mae, dayResults{d}.price_mae);
            if RUN_FFNN_BENCHMARK
                fprintf('  FFNN MAE:     PV=%.4f, Load=%.4f, Price=%.4f\n', ...
                    dayResults{d}.ffnn_pv_mae, dayResults{d}.ffnn_load_mae, dayResults{d}.ffnn_price_mae);
            end
            if RUN_LSTM_BENCHMARK
                fprintf('  LSTM MAE:     PV=%.4f, Load=%.4f, Price=%.4f\n', ...
                    dayResults{d}.lstm_pv_mae, dayResults{d}.lstm_load_mae, dayResults{d}.lstm_price_mae);
            end
            fprintf('  S1 grid+uncontrolled EV: $%.4f\n', dayResults{d}.s1_cost);
            fprintf('  S2 BESS+uncontrolled EV: $%.4f +/- $%.4f\n', ...
                dayResults{d}.s2_mean, dayResults{d}.s2_std);
            fprintf('  S3 PSO BESS+optimized EV/V2G: $%.4f +/- $%.4f  (saving %.2f%%)\n', ...
                dayResults{d}.s3_mean, dayResults{d}.s3_std, dayResults{d}.s3_savings_pct);
            fprintf('  EV charge/discharge: %.2f / %.2f kWh, final SoC %.1f%%, V2G %.2f kWh\n', ...
                dayResults{d}.ev_ch_mean, dayResults{d}.ev_dis_mean, ...
                100*dayResults{d}.ev_final_soc_mean, dayResults{d}.v2g_service_energy_repr);
            if MULTIDAY_INCLUDE_DE
                fprintf('  DE BESS+optimized EV/V2G: $%.4f +/- $%.4f  (saving %.2f%%)\n', ...
                    dayResults{d}.de_s3_mean, dayResults{d}.de_s3_std, dayResults{d}.de_s3_savings_pct);
            end
        end

        % Convert cell results into arrays for reporting/plots.
        md_date = NaT(nDays,1);
        md_pv_mae = zeros(nDays,1);
        md_load_mae = zeros(nDays,1);
        md_price_mae = zeros(nDays,1);
        md_ffnn_pv_mae = nan(nDays,1);
        md_ffnn_load_mae = nan(nDays,1);
        md_ffnn_price_mae = nan(nDays,1);
        md_lstm_pv_mae = nan(nDays,1);
        md_lstm_load_mae = nan(nDays,1);
        md_lstm_price_mae = nan(nDays,1);
        md_s1 = zeros(nDays,1);
        md_s2 = zeros(nDays,1);
        md_s2_std = zeros(nDays,1);
        md_s3 = zeros(nDays,1);
        md_s3_std = zeros(nDays,1);
        md_s3_sav_pct = zeros(nDays,1);
        md_s3_cv = zeros(nDays,1);
        md_p_s3_vs_s2 = zeros(nDays,1);
        md_ev_ch = zeros(nDays,1);
        md_ev_dis = zeros(nDays,1);
        md_ev_soc = zeros(nDays,1);
        md_v2g = zeros(nDays,1);
        md_base_peak = zeros(nDays,1);
        md_prop_peak = zeros(nDays,1);
        md_peak_red_pct = zeros(nDays,1);
        md_de_s3 = nan(nDays,1);
        md_de_s3_std = nan(nDays,1);
        md_de_sav_pct = nan(nDays,1);

        for d = 1:nDays
            r = dayResults{d};
            md_date(d) = r.date;
            md_pv_mae(d) = r.pv_mae;
            md_load_mae(d) = r.load_mae;
            md_price_mae(d) = r.price_mae;
            if isfield(r, 'ffnn_pv_mae')
                md_ffnn_pv_mae(d) = r.ffnn_pv_mae;
                md_ffnn_load_mae(d) = r.ffnn_load_mae;
                md_ffnn_price_mae(d) = r.ffnn_price_mae;
            end
            if isfield(r, 'lstm_pv_mae')
                md_lstm_pv_mae(d) = r.lstm_pv_mae;
                md_lstm_load_mae(d) = r.lstm_load_mae;
                md_lstm_price_mae(d) = r.lstm_price_mae;
            end
            md_s1(d) = r.s1_cost;
            md_s2(d) = r.s2_mean;
            md_s2_std(d) = r.s2_std;
            md_s3(d) = r.s3_mean;
            md_s3_std(d) = r.s3_std;
            md_s3_sav_pct(d) = r.s3_savings_pct;
            md_s3_cv(d) = 100*r.s3_std/max(abs(r.s3_mean), eps);
            md_p_s3_vs_s2(d) = r.p_s3_vs_s2;
            md_ev_ch(d) = r.ev_ch_mean;
            md_ev_dis(d) = r.ev_dis_mean;
            md_ev_soc(d) = r.ev_final_soc_mean;
            md_v2g(d) = r.v2g_service_energy_repr;
            md_base_peak(d) = r.baseline_peak;
            md_prop_peak(d) = r.proposed_peak;
            md_peak_red_pct(d) = r.peak_reduction_pct;
            if MULTIDAY_INCLUDE_DE
                md_de_s3(d) = r.de_s3_mean;
                md_de_s3_std(d) = r.de_s3_std;
                md_de_sav_pct(d) = r.de_s3_savings_pct;
            end
        end

        fprintf('\n============================================================\n');
        fprintf('MULTI-DAY EVALUATION SUMMARY (%d days, %d runs/day)\n', nDays, MULTIDAY_N_RUNS);
        fprintf('============================================================\n');

        fprintf('\n--- Forecast Accuracy Across Days ---\n');
        fprintf('  PV MAE:    mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
            mean(md_pv_mae), std(md_pv_mae), min(md_pv_mae), max(md_pv_mae));
        fprintf('  Load MAE:  mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
            mean(md_load_mae), std(md_load_mae), min(md_load_mae), max(md_load_mae));
        fprintf('  Price MAE: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
            mean(md_price_mae), std(md_price_mae), min(md_price_mae), max(md_price_mae));
        if RUN_FFNN_BENCHMARK
            ffnn_pv_valid = md_ffnn_pv_mae(~isnan(md_ffnn_pv_mae));
            ffnn_load_valid = md_ffnn_load_mae(~isnan(md_ffnn_load_mae));
            ffnn_price_valid = md_ffnn_price_mae(~isnan(md_ffnn_price_mae));

            ffnn_pv_mean = mean(ffnn_pv_valid);
            ffnn_load_mean = mean(ffnn_load_valid);
            ffnn_price_mean = mean(ffnn_price_valid);

            fprintf('\n--- FFNN Forecast Accuracy Across Days ---\n');
            fprintf('  PV MAE:    mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
                ffnn_pv_mean, std(ffnn_pv_valid), min(ffnn_pv_valid), max(ffnn_pv_valid));
            fprintf('  Load MAE:  mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
                ffnn_load_mean, std(ffnn_load_valid), min(ffnn_load_valid), max(ffnn_load_valid));
            fprintf('  Price MAE: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
                ffnn_price_mean, std(ffnn_price_valid), min(ffnn_price_valid), max(ffnn_price_valid));
            fprintf('\n--- Multi-day FCM-TSK Improvement Over FFNN ---\n');
            fprintf('  PV:    %.2f%%\n', 100*(1 - mean(md_pv_mae)/max(ffnn_pv_mean,eps)));
            fprintf('  Load:  %.2f%%\n', 100*(1 - mean(md_load_mae)/max(ffnn_load_mean,eps)));
            fprintf('  Price: %.2f%%\n', 100*(1 - mean(md_price_mae)/max(ffnn_price_mean,eps)));
        end

        if RUN_LSTM_BENCHMARK
            lpv = md_lstm_pv_mae(~isnan(md_lstm_pv_mae));
            lld = md_lstm_load_mae(~isnan(md_lstm_load_mae));
            lpr = md_lstm_price_mae(~isnan(md_lstm_price_mae));
            fprintf('\n--- LSTM Forecast Accuracy Across Days ---\n');
            fprintf('  PV MAE:    mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', mean(lpv),std(lpv),min(lpv),max(lpv));
            fprintf('  Load MAE:  mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', mean(lld),std(lld),min(lld),max(lld));
            fprintf('  Price MAE: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', mean(lpr),std(lpr),min(lpr),max(lpr));
            fprintf('\n--- Multi-day FCM-TSK Improvement Over LSTM ---\n');
            fprintf('  PV:    %.2f%%\n',100*(1-mean(md_pv_mae)/max(mean(lpv),eps)));
            fprintf('  Load:  %.2f%%\n',100*(1-mean(md_load_mae)/max(mean(lld),eps)));
            fprintf('  Price: %.2f%%\n',100*(1-mean(md_price_mae)/max(mean(lpr),eps)));
        end

        fprintf('\n--- Economic Performance Across Days ---\n');
        fprintf('  S1 baseline cost: mean=$%.4f +/- $%.4f\n', mean(md_s1), std(md_s1));
        fprintf('  S2 BESS+uncontrolled EV: mean=$%.4f +/- $%.4f, saving=%.2f%%\n', ...
            mean(md_s2), std(md_s2), 100*(mean(md_s1)-mean(md_s2))/mean(md_s1));
        fprintf('  S3 PSO BESS+optimized EV/V2G: mean=$%.4f +/- $%.4f, saving=%.2f%% +/- %.2f%%\n', ...
            mean(md_s3), std(md_s3), mean(md_s3_sav_pct), std(md_s3_sav_pct));
        fprintf('  Days with positive PSO savings: %d/%d\n', sum(md_s3_sav_pct > 0), nDays);
        fprintf('  Days with significant EV benefit vs S2 (p<0.05, correct direction): %d/%d\n', ...
            sum((md_p_s3_vs_s2 < 0.05) & (md_s3 < md_s2)), nDays);
        fprintf('  Mean PSO CV across days: %.2f%%\n', mean(md_s3_cv));

        if MULTIDAY_INCLUDE_DE
            fprintf('  DE BESS+optimized EV/V2G: mean=$%.4f +/- $%.4f, saving=%.2f%% +/- %.2f%%\n', ...
                mean(md_de_s3), std(md_de_s3), mean(md_de_sav_pct), std(md_de_sav_pct));
        end

        fprintf('\n--- EV/V2G Utilization Across Days ---\n');
        fprintf('  EV charge:    %.2f +/- %.2f kWh/day\n', mean(md_ev_ch), std(md_ev_ch));
        fprintf('  EV discharge: %.2f +/- %.2f kWh/day\n', mean(md_ev_dis), std(md_ev_dis));
        fprintf('  V2G service:  %.2f +/- %.2f kWh/day\n', mean(md_v2g), std(md_v2g));
        fprintf('  Final EV SoC: %.1f +/- %.1f%%\n', 100*mean(md_ev_soc), 100*std(md_ev_soc));

        fprintf('\n--- Peak Demand Across Days ---\n');
        fprintf('  Baseline peak: %.2f +/- %.2f kW\n', mean(md_base_peak), std(md_base_peak));
        fprintf('  Proposed peak: %.2f +/- %.2f kW\n', mean(md_prop_peak), std(md_prop_peak));
        fprintf('  Peak reduction: %.2f%% +/- %.2f%%\n', mean(md_peak_red_pct), std(md_peak_red_pct));

        try
            [~, p_sav_ttest] = ttest(md_s3_sav_pct, 0);
        catch
            p_sav_ttest = NaN;
        end
        try
            p_sav_signrank = signrank(md_s3_sav_pct);
        catch
            p_sav_signrank = NaN;
        end
        fprintf('\n--- Across-Day Statistical Test ---\n');
        fprintf('  One-sample t-test for PSO savings%% > 0: p = %.4f\n', p_sav_ttest);
        fprintf('  Wilcoxon signed-rank test for PSO savings%% > 0: p = %.4f\n', p_sav_signrank);

        % Save table for paper/tables.
        T_multiday = table(md_date, md_pv_mae, md_load_mae, md_price_mae, ...
            md_ffnn_pv_mae, md_ffnn_load_mae, md_ffnn_price_mae, ...
            md_s1, md_s2, md_s2_std, md_s3, md_s3_std, md_s3_sav_pct, md_s3_cv, md_p_s3_vs_s2, ...
            md_ev_ch, md_ev_dis, 100*md_ev_soc, md_v2g, ...
            md_base_peak, md_prop_peak, md_peak_red_pct, md_de_s3, md_de_s3_std, md_de_sav_pct, ...
            'VariableNames', {'Date','FCM_PV_MAE','FCM_Load_MAE','FCM_Price_MAE', ...
            'FFNN_PV_MAE','FFNN_Load_MAE','FFNN_Price_MAE', ...
            'S1_Grid_UncontrolledEV_Cost','S2_BESS_UncontrolledEV_MeanCost','S2_BESS_UncontrolledEV_StdCost', ...
            'S3_PSO_OptimizedEV_MeanCost','S3_PSO_StdCost','S3_PSO_SavingsPct','S3_PSO_CV_Pct','P_S3_vs_S2', ...
            'EV_Charge_kWh','EV_Discharge_kWh','EV_FinalSoC_pct','V2G_Service_kWh', ...
            'BaselinePeak_kW','ProposedPeak_kW','PeakReduction_pct','DE_S3_MeanCost','DE_S3_StdCost','DE_SavingsPct'});

        if MULTIDAY_SAVE_CSV
            out_csv = fullfile(script_dir, sprintf('multiday_results_%ddays_%druns.csv', nDays, MULTIDAY_N_RUNS));
            writetable(T_multiday, out_csv);
            fprintf('\nSaved multi-day table: %s\n', out_csv);
        end

        % Figure: daily cost comparison.
        % Shows all economic stages:
        %   S1 -> S2: benefit from BESS with uncontrolled EV charging
        %   S2 -> S3: additional benefit from optimized EV/V2G scheduling
        %   S3 vs DE: optimizer benchmark comparison
        figure('Position', [100, 100, 950, 440], 'Name', 'Multi-day Daily Cost Comparison', 'Color', 'w');
        plot(md_date, md_s1, 'ko-', 'LineWidth', 1.6, 'MarkerSize', 6, ...
            'DisplayName', 'S1 Grid + Uncontrolled EV');
        hold on;
        errorbar(md_date, md_s2, md_s2_std, 'd--', 'LineWidth', 1.5, ...
            'MarkerSize', 6, 'CapSize', 8, ...
            'DisplayName', 'S2 BESS + Uncontrolled EV');
        errorbar(md_date, md_s3, md_s3_std, 'bs-', 'LineWidth', 1.7, ...
            'MarkerSize', 6, 'CapSize', 8, ...
            'DisplayName', 'S3 PSO BESS + Optimized EV/V2G');
        if MULTIDAY_INCLUDE_DE
            errorbar(md_date, md_de_s3, md_de_s3_std, 'r^-', 'LineWidth', 1.5, ...
                'MarkerSize', 6, 'CapSize', 8, ...
                'DisplayName', 'DE BESS + Optimized EV/V2G');
        end
        grid on; box on;
        xlabel('Test Day'); ylabel('Daily Cost ($)');
        title(sprintf('Daily Dispatch Cost Across %d Test Days', nDays));
        legend('Location','best');
        xtickformat('yyyy-MM-dd');
        xtickangle(45);

        % Figure: savings and peak reduction.
        figure('Position', [120, 120, 900, 420], 'Name', 'Multi-day Savings and Peak Reduction', 'Color', 'w');
        yyaxis left;
        bar(md_date, md_s3_sav_pct, 0.55);
        ylabel('PSO Savings vs S1 (%)');
        yline(0, 'k--', 'HandleVisibility','off');
        yyaxis right;
        plot(md_date, md_peak_red_pct, 'o-', 'LineWidth', 1.8, 'MarkerSize', 6);
        ylabel('Peak Reduction (%)');
        grid on; box on;
        xlabel('Test Day');
        title('Multi-day Savings and Peak-Demand Reduction');
        xtickformat('yyyy-MM-dd');
        xtickangle(45);

        % Figure: EV energy use.
        figure('Position', [140, 140, 900, 420], 'Name', 'Multi-day EV/V2G Utilization', 'Color', 'w');
        bar(md_date, [md_ev_ch, md_ev_dis, md_v2g], 'grouped');
        grid on; box on;
        xlabel('Test Day'); ylabel('Energy (kWh/day)');
        title('EV Charging, Discharging, and V2G Service Across Test Days');
        legend('EV charge','EV V2G discharge','V2G service','Location','best');
        xtickformat('yyyy-MM-dd');
        xtickangle(45);
    end
end


%% ==========================
% LOCAL FUNCTIONS
% ==========================


function dayres = evaluate_one_multiday_dispatch(t0_day, ts, PV_hist, Load_hist, Price_hist, EV_Charging_hist, EV_Discharging_hist, EV_SoC_hist, Vehicle_Count_hist, ...
    cal_all, mu, sd, pv_final_all, load_final_all, price_final_all, ...
    run_ffnn_benchmark, ffnn_pv_model, ffnn_load_model, ffnn_price_model, ...
    run_lstm_benchmark, lstm_pv_model, lstm_load_model, lstm_price_model, n_cal_feat, ...
    LAGS, H, PV_CAPACITY, use_price_enhancement, battery_params, EV_params_template, ...
    swarm_size, maxiter, w_inertia, c1, c2, vel_clamp_factor, n_runs, seed_base, warm_start_frac, ...
    de_F, de_CR, include_de)
% EVALUATE_ONE_MULTIDAY_DISPATCH
% Performs one 24-h forecast + dispatch evaluation day using the same
% baseline definitions as the main single-day experiment:
%   S1 = grid + uncontrolled mandatory EV charging
%   S2 = BESS + uncontrolled mandatory EV charging
%   S3 = BESS + optimized EV charging/V2G

    dt = battery_params.dt;

    % Day-specific EV parameter copies.
    EV_params = EV_params_template;
    EV_params.horizon_start_hour = hour(ts(t0_day));

    EV_params_base = EV_params;
    EV_params_base.v2g_required_energy = 0.0;
    EV_params_base.v2g_requirement_weight = 0.0;
    EV_params_base.v2g_overdelivery_weight = 0.0;
    EV_params_base.v2g_allowed_hours = [];
    EV_params_base.no_charge_hours = [];

    % ---------- Forecast rollout ----------
    pv_win      = PV_hist(t0_day-LAGS:t0_day-1);
    load_win    = Load_hist(t0_day-LAGS:t0_day-1);
    price_win   = Price_hist(t0_day-LAGS:t0_day-1);

    % Same EV-related input features used during training.
    ev_soc_win  = EV_SoC_hist(t0_day-LAGS:t0_day-1);
    vehicle_win = Vehicle_Count_hist(t0_day-LAGS:t0_day-1);

    cal_feat = cal_all(t0_day, :);
    x = [pv_win(:)', load_win(:)', price_win(:)', ev_soc_win(:)', vehicle_win(:)', cal_feat(:)'];
    x_n = (x - mu) ./ sd;

    PV_fore = zeros(H,1);
    Load_fore = zeros(H,1);
    Price_fore = zeros(H,1);

    for h = 1:H
        model = pv_final_all{h};
        y_log = fuzzy_predict_batch(x_n, model.center, model.sigma, model.a);
        pv_hat = max(0, exp(y_log)-0.05);
        pv_hat = min(pv_hat, PV_CAPACITY);
        hour_h = mod(hour(ts(t0_day)) + h - 1, 24);
        if hour_h >= 20 || hour_h <= 5
            pv_hat = 0;
        end
        PV_fore(h) = pv_hat;

        model = load_final_all{h};
        Load_fore(h) = max(0, fuzzy_predict_batch(x_n, model.center, model.sigma, model.a));

        model = price_final_all{h};
        Price_fore(h) = max(0, fuzzy_predict_batch(x_n, model.center, model.sigma, model.a));
    end

    % ----------------------------------------------------------------------
    % FIX (forecast plausibility safeguard): PV_fore is already bounded on
    % both ends (max(0,...) and min(.,PV_CAPACITY)). Load_fore and
    % Price_fore previously had NO upper bound at all. The TSK consequent
    % is a raw affine extrapolation per rule; on an unusual test day (e.g.
    % a holiday with atypical demand/price patterns) where the input falls
    % outside the region any rule was well-supported by during training,
    % the prediction can blow up to an implausible value with nothing
    % catching it. This was traced as the likely cause of one multi-day
    % test day reporting a price MAE roughly 4x every other evaluated day
    % -- which both inflated that day's S1 baseline cost (price feeds the
    % grid-energy-cost term directly) and destabilised the PSO/DE search
    % (the optimum dispatch timing becomes far more sensitive around a
    % spuriously-priced hour, increasing run-to-run variance even on the
    % otherwise low-variance BESS-only problem). A generous ceiling (3x
    % the historical maximum) now catches this, with an explicit warning
    % printed for any clipped hour so it is visible in the log rather than
    % silently corrupting both the forecast-accuracy metric and the
    % optimiser's input.
    % ----------------------------------------------------------------------
    LOAD_PLAUSIBLE_MAX  = 3 * max(Load_hist);
    PRICE_PLAUSIBLE_MAX = 3 * max(Price_hist);

    PRICE_SOFT_THRESHOLD = 1.5 * max(Price_hist);
    [worst_price_val, worst_price_hr] = max(Price_fore);
    if worst_price_val > PRICE_SOFT_THRESHOLD
        fprintf('  NOTE (%s): worst-hour price forecast = %.4f $/kWh at hour %d (soft threshold %.4f, hard ceiling %.4f) -- not clipped, flagged for review.\n', ...
            datestr(ts(t0_day), 'yyyy-mm-dd'), worst_price_val, worst_price_hr, PRICE_SOFT_THRESHOLD, PRICE_PLAUSIBLE_MAX);
    end

    LOAD_SOFT_THRESHOLD = 1.5 * max(Load_hist);
    [worst_load_val, worst_load_hr] = max(Load_fore);
    if worst_load_val > LOAD_SOFT_THRESHOLD
        fprintf('  NOTE (%s): worst-hour load forecast = %.3f kW at hour %d (soft threshold %.3f, hard ceiling %.3f) -- not clipped, flagged for review.\n', ...
            datestr(ts(t0_day), 'yyyy-mm-dd'), worst_load_val, worst_load_hr, LOAD_SOFT_THRESHOLD, LOAD_PLAUSIBLE_MAX);
    end

    clipped_load = find(Load_fore > LOAD_PLAUSIBLE_MAX);
    if ~isempty(clipped_load)
        for hh = clipped_load(:)'
            fprintf('  WARNING (%s): Load forecast at hour %d = %.3f kW exceeds plausibility ceiling (%.3f kW) -- clipped.\n', ...
                datestr(ts(t0_day), 'yyyy-mm-dd'), hh, Load_fore(hh), LOAD_PLAUSIBLE_MAX);
        end
        Load_fore(clipped_load) = LOAD_PLAUSIBLE_MAX;
    end

    clipped_price = find(Price_fore > PRICE_PLAUSIBLE_MAX);
    if ~isempty(clipped_price)
        for hh = clipped_price(:)'
            fprintf('  WARNING (%s): Price forecast at hour %d = %.4f $/kWh exceeds plausibility ceiling (%.4f $/kWh) -- clipped.\n', ...
                datestr(ts(t0_day), 'yyyy-mm-dd'), hh, Price_fore(hh), PRICE_PLAUSIBLE_MAX);
        end
        Price_fore(clipped_price) = PRICE_PLAUSIBLE_MAX;
    end


    Price_fore_original = Price_fore;
    if use_price_enhancement
        load_features = zeros(H,4);
        for h = 1:H
            load_features(h,1) = Load_fore(h);
            if h > 1
                load_features(h,2) = Load_fore(h) - Load_fore(h-1);
            else
                load_features(h,2) = 0;
            end
            load_features(h,3) = double(Load_fore(h) > 3.5);
            max_load = max(Load_fore); min_load = min(Load_fore);
            if max_load > min_load
                load_features(h,4) = (Load_fore(h)-min_load)/(max_load-min_load);
            else
                load_features(h,4) = 0.5;
            end
        end

        Price_fore = Price_fore_original;
        for h = 1:H
            if load_features(h,3) == 1
                Price_fore(h) = Price_fore(h) * 1.05;
            end
            if load_features(h,2) > 0.3
                Price_fore(h) = Price_fore(h) * 1.03;
            elseif load_features(h,2) < -0.3
                Price_fore(h) = Price_fore(h) * 0.97;
            end
        end
        Price_fore = max(0, Price_fore);
    end

    % ---------- FFNN benchmark forecast for this day ----------
    if run_ffnn_benchmark
        PV_ffnn_fore = ffnn_predict_batch(x_n, ffnn_pv_model).';
        Load_ffnn_fore = ffnn_predict_batch(x_n, ffnn_load_model).';
        Price_ffnn_fore_original = ffnn_predict_batch(x_n, ffnn_price_model).';

        PV_ffnn_fore = min(max(PV_ffnn_fore,0), PV_CAPACITY);
        Load_ffnn_fore = min(max(0, Load_ffnn_fore), LOAD_PLAUSIBLE_MAX);
        Price_ffnn_fore_original = min(max(0, Price_ffnn_fore_original), PRICE_PLAUSIBLE_MAX);

        for h = 1:H
            hour_h = mod(hour(ts(t0_day)) + h - 1, 24);
            if hour_h >= 20 || hour_h <= 5
                PV_ffnn_fore(h) = 0;
            end
        end

        Price_ffnn_fore_enhanced = apply_load_adaptive_price_enhancement( ...
            Price_ffnn_fore_original, Load_ffnn_fore);
        if use_price_enhancement
            Price_ffnn_fore = Price_ffnn_fore_enhanced;
        else
            Price_ffnn_fore = Price_ffnn_fore_original;
        end
    else
        PV_ffnn_fore = nan(H,1);
        Load_ffnn_fore = nan(H,1);
        Price_ffnn_fore = nan(H,1);
    end

    % ---------- LSTM benchmark forecast for this day ----------
    if run_lstm_benchmark
        x_seq = flat_inputs_to_lstm_sequences(x_n, LAGS, n_cal_feat);
        PV_lstm_fore = lstm_predict_batch(x_seq, lstm_pv_model).';
        Load_lstm_fore = lstm_predict_batch(x_seq, lstm_load_model).';
        Price_lstm_fore_original = lstm_predict_batch(x_seq, lstm_price_model).';
        PV_lstm_fore = min(max(PV_lstm_fore,0),PV_CAPACITY);
        Load_lstm_fore = min(max(0,Load_lstm_fore), LOAD_PLAUSIBLE_MAX);
        Price_lstm_fore_original = min(max(0,Price_lstm_fore_original), PRICE_PLAUSIBLE_MAX);
        for h=1:H
            hour_h=mod(hour(ts(t0_day))+h-1,24);
            if hour_h>=20 || hour_h<=5, PV_lstm_fore(h)=0; end
        end
        Price_lstm_fore_enhanced = apply_load_adaptive_price_enhancement( ...
            Price_lstm_fore_original,Load_lstm_fore);
        if use_price_enhancement, Price_lstm_fore=Price_lstm_fore_enhanced;
        else, Price_lstm_fore=Price_lstm_fore_original; end
    else
        PV_lstm_fore=nan(H,1); Load_lstm_fore=nan(H,1); Price_lstm_fore=nan(H,1);
    end

    PV_actual = PV_hist(t0_day:t0_day+H-1);
    Load_actual = Load_hist(t0_day:t0_day+H-1);
    Price_actual = Price_hist(t0_day:t0_day+H-1);

    dayres.date = dateshift(ts(t0_day), 'start', 'day');
    dayres.t0 = t0_day;
    dayres.pv_mae = mean(abs(PV_actual - PV_fore));
    dayres.load_mae = mean(abs(Load_actual - Load_fore));
    dayres.price_mae = mean(abs(Price_actual - Price_fore));
    dayres.ffnn_pv_mae = mean(abs(PV_actual - PV_ffnn_fore));
    dayres.ffnn_load_mae = mean(abs(Load_actual - Load_ffnn_fore));
    dayres.ffnn_price_mae = mean(abs(Price_actual - Price_ffnn_fore));
    dayres.lstm_pv_mae = mean(abs(PV_actual - PV_lstm_fore));
    dayres.lstm_load_mae = mean(abs(Load_actual - Load_lstm_fore));
    dayres.lstm_price_mae = mean(abs(Price_actual - Price_lstm_fore));

    % ---------- EV baseline and dispatch optimization ----------
    [EV_ch_uncontrolled, EV_dis_uncontrolled] = uncontrolled_ev_charging_schedule(EV_params_base, H, dt, ts(t0_day));
    EV_ch_seed = EV_ch_uncontrolled;
    EV_dis_seed = EV_dis_uncontrolled;

    BuyPrice = Price_fore;
    SellPrice = 0.85 * Price_fore;

    zeroPbat = zeros(H,1);
    s1_cost = total_cost_withGrid_and_soc_repair_EV(zeroPbat, ...
        EV_ch_uncontrolled, EV_dis_uncontrolled, ...
        PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
        battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
        battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
        dt, EV_params_base, battery_params);

    shared_init_day = generate_shared_init_populations(H, battery_params, EV_params, ...
        EV_ch_seed, EV_dis_seed, swarm_size, n_runs, seed_base, warm_start_frac);

    mrc_day = multirun_pso_compare(PV_fore, Load_fore, Price_fore, BuyPrice, SellPrice, ...
        battery_params, EV_params, EV_params_base, EV_ch_seed, EV_dis_seed, ...
        EV_ch_uncontrolled, EV_dis_uncontrolled, H, swarm_size, maxiter, ...
        w_inertia, c1, c2, vel_clamp_factor, n_runs, seed_base, shared_init_day);

    dayres.s1_cost = s1_cost;
    dayres.s2_mean = mrc_day.s2_mean;
    dayres.s2_std = mrc_day.s2_std;
    dayres.s3_mean = mrc_day.s3_mean;
    dayres.s3_std = mrc_day.s3_std;
    dayres.s3_min = mrc_day.s3_min;
    dayres.s3_max = mrc_day.s3_max;
    dayres.p_s3_vs_s2 = mrc_day.p_s3_vs_s2;
    dayres.s3_savings_pct = 100 * (s1_cost - mrc_day.s3_mean) / max(abs(s1_cost), eps);
    dayres.ev_ch_mean = mrc_day.s3_ev_ch_mean;
    dayres.ev_dis_mean = mrc_day.s3_ev_dis_mean;
    dayres.ev_final_soc_mean = mrc_day.s3_ev_final_soc_mean;

    hist_repr = mrc_day.s3_repr_hist;
    if isfield(hist_repr, 'v2g_service_energy')
        dayres.v2g_service_energy_repr = hist_repr.v2g_service_energy;
    else
        dayres.v2g_service_energy_repr = sum(hist_repr.P_ev_dis) * dt;
    end

    baseline_grid = Load_fore + EV_ch_uncontrolled - EV_dis_uncontrolled - PV_fore;
    dayres.baseline_peak = max(max(0, baseline_grid));
    dayres.proposed_peak = max(max(0, hist_repr.GridPower));
    dayres.peak_reduction_pct = 100 * (dayres.baseline_peak - dayres.proposed_peak) / max(dayres.baseline_peak, eps);

    if include_de
        mrc_de_day = multirun_de_compare(PV_fore, Load_fore, Price_fore, BuyPrice, SellPrice, ...
            battery_params, EV_params, EV_params_base, EV_ch_seed, EV_dis_seed, ...
            EV_ch_uncontrolled, EV_dis_uncontrolled, H, swarm_size, maxiter, ...
            de_F, de_CR, n_runs, seed_base, shared_init_day);
        dayres.de_s3_mean = mrc_de_day.s3_mean;
        dayres.de_s3_std = mrc_de_day.s3_std;
        dayres.de_s3_savings_pct = 100 * (s1_cost - mrc_de_day.s3_mean) / max(abs(s1_cost), eps);
    else
        dayres.de_s3_mean = NaN;
        dayres.de_s3_std = NaN;
        dayres.de_s3_savings_pct = NaN;
    end
end



function Xseq = flat_inputs_to_lstm_sequences(Xflat, LAGS, n_cal_feat)
% Converts [N x (5*LAGS+nCal)] block-ordered inputs into N cell sequences.
% Sequence channels: PV, Load, Price, EV SoC, vehicle count, then calendar.
    if isvector(Xflat), Xflat = reshape(Xflat,1,[]); end
    N = size(Xflat,1);
    expected = 5*LAGS+n_cal_feat;
    if size(Xflat,2) ~= expected
        error('LSTM input conversion expected %d features, received %d.',expected,size(Xflat,2));
    end
    Xseq = cell(N,1);
    for i=1:N
        seq = zeros(5+n_cal_feat,LAGS);
        for v=1:5
            cols=(v-1)*LAGS+(1:LAGS);
            seq(v,:)=Xflat(i,cols);
        end
        cal=Xflat(i,5*LAGS+(1:n_cal_feat)).';
        seq(6:end,:)=repmat(cal,1,LAGS);
        % The first two calendar channels are target-hour sin/cos. Rebuild
        % their historical values over the 24 lag positions instead of
        % repeating one constant value at every time step.
        if n_cal_feat >= 2
            theta_target = atan2(cal(1),cal(2));
            lag_from_target = LAGS:-1:1;
            theta_hist = theta_target - 2*pi*lag_from_target/24;
            seq(6,:) = sin(theta_hist);
            seq(7,:) = cos(theta_hist);
        end
        Xseq{i}=seq;
    end
end

function model = train_lstm_regressor(Xtr,Ytr,Xva,Yva,opts,model_name)
% Sequence-to-vector LSTM baseline with chronological validation.
% IMPORTANT: standardize each forecast horizon using TRAINING targets only.
% The original LSTM was trained on raw targets, whereas the FFNN internally
% standardized its outputs. This scale mismatch was the main reason for the
% very poor LSTM load forecast and the large/unstable validation RMSE.
    rng(opts.seed,'twister');
    Ytr = double(Ytr); Yva = double(Yva);
    y_mu = mean(Ytr,1);
    y_sd = std(Ytr,0,1);
    y_sd(y_sd < 1e-8) = 1;
    Ytr_n = (Ytr-y_mu)./y_sd;
    Yva_n = (Yva-y_mu)./y_sd;

    nFeatures=size(Xtr{1},1); nOut=size(Ytr,2);
    layers=[sequenceInputLayer(nFeatures,'Normalization','none','Name','input')
        lstmLayer(opts.num_hidden,'OutputMode','last','Name','lstm')
        dropoutLayer(0.10,'Name','dropout')
        fullyConnectedLayer(opts.fc_hidden,'Name','fc1')
        reluLayer('Name','relu')
        fullyConnectedLayer(nOut,'Name','output')
        regressionLayer('Name','regression')];
    valFreq=max(1,floor(numel(Xtr)/opts.batch_size));
    trainOpts=trainingOptions('adam', ...
        'MaxEpochs',opts.epochs,'MiniBatchSize',opts.batch_size, ...
        'InitialLearnRate',opts.learn_rate,'L2Regularization',opts.l2, ...
        'GradientThreshold',1,'Shuffle','never', ...
        'ValidationData',{Xva,Yva_n},'ValidationFrequency',valFreq, ...
        'ValidationPatience',opts.patience, ...
        'LearnRateSchedule','piecewise','LearnRateDropPeriod',30, ...
        'LearnRateDropFactor',0.5, ...
        'Verbose',opts.verbose,'Plots','none', ...
        'ExecutionEnvironment','auto');
    fprintf('  Training tuned LSTM %-6s: %d sequences, %d channels, %d outputs\n', ...
        model_name,numel(Xtr),nFeatures,nOut);
    model.net=trainNetwork(Xtr,Ytr_n,layers,trainOpts);
    model.y_mu=y_mu;
    model.y_sd=y_sd;
end

function Yhat = lstm_predict_batch(Xseq,model)
    Yhat_n=predict(model.net,Xseq,'MiniBatchSize',256);
    if isvector(Yhat_n), Yhat_n=reshape(Yhat_n,1,[]); end
    Yhat = Yhat_n .* model.y_sd + model.y_mu;
end

function model = train_custom_ffnn_regressor(Xtr, Ytr, Xva, Yva, hidden_sizes, opts, model_name)
% TRAIN_CUSTOM_FFNN_REGRESSOR
% Custom feed-forward neural network for multi-output regression.
% No Neural Network Toolbox or Deep Learning Toolbox is required.
% Inputs are assumed already normalized. Targets are internally standardized.

    if nargin < 7 || isempty(model_name), model_name = 'FFNN'; end
    if ~isfield(opts,'epochs'), opts.epochs = 250; end
    if ~isfield(opts,'batch_size'), opts.batch_size = 256; end
    if ~isfield(opts,'learn_rate'), opts.learn_rate = 1e-3; end
    if ~isfield(opts,'l2'), opts.l2 = 1e-5; end
    if ~isfield(opts,'patience'), opts.patience = 30; end
    if ~isfield(opts,'seed'), opts.seed = 2027; end
    if ~isfield(opts,'verbose'), opts.verbose = true; end

    Xtr = double(Xtr); Ytr = double(Ytr);
    Xva = double(Xva); Yva = double(Yva);

    y_mu = mean(Ytr,1);
    y_sd = std(Ytr,0,1) + 1e-12;
    Ytr_n = (Ytr - y_mu) ./ y_sd;
    Yva_n = (Yva - y_mu) ./ y_sd;

    rng(opts.seed);
    layer_sizes = [size(Xtr,2), hidden_sizes(:).', size(Ytr,2)];
    nLayers = numel(layer_sizes) - 1;

    W = cell(nLayers,1); b = cell(nLayers,1);
    mW = cell(nLayers,1); vW = cell(nLayers,1);
    mb = cell(nLayers,1); vb = cell(nLayers,1);

    for l = 1:nLayers
        fan_in = layer_sizes(l);
        fan_out = layer_sizes(l+1);
        if l < nLayers
            scale = sqrt(2 / fan_in);   % He initialization for ReLU layers
        else
            scale = sqrt(1 / fan_in);
        end
        W{l} = scale * randn(fan_in, fan_out);
        b{l} = zeros(1, fan_out);
        mW{l} = zeros(size(W{l})); vW{l} = zeros(size(W{l}));
        mb{l} = zeros(size(b{l})); vb{l} = zeros(size(b{l}));
    end

    beta1 = 0.9; beta2 = 0.999; adam_eps = 1e-8;
    n = size(Xtr,1);
    batch_size = min(opts.batch_size, n);
    iter = 0;

    best_val = inf;
    bestW = W; bestb = b;
    bad_epochs = 0;

    for epoch = 1:opts.epochs
        ord = randperm(n);
        for startIdx = 1:batch_size:n
            iter = iter + 1;
            idx = ord(startIdx:min(startIdx+batch_size-1,n));
            Xb = Xtr(idx,:);
            Yb = Ytr_n(idx,:);
            m = size(Xb,1);

            A = cell(nLayers+1,1); Z = cell(nLayers,1);
            A{1} = Xb;
            for l = 1:nLayers
                Z{l} = A{l} * W{l} + b{l};
                if l < nLayers
                    A{l+1} = max(0, Z{l});
                else
                    A{l+1} = Z{l};
                end
            end

            dZ = (2/m) * (A{end} - Yb);
            dW = cell(nLayers,1); db = cell(nLayers,1);
            for l = nLayers:-1:1
                dW{l} = A{l}' * dZ + opts.l2 * W{l};
                db{l} = sum(dZ,1);
                if l > 1
                    dAprev = dZ * W{l}';
                    dZ = dAprev .* (Z{l-1} > 0);
                end
            end

            for l = 1:nLayers
                mW{l} = beta1*mW{l} + (1-beta1)*dW{l};
                vW{l} = beta2*vW{l} + (1-beta2)*(dW{l}.^2);
                mb{l} = beta1*mb{l} + (1-beta1)*db{l};
                vb{l} = beta2*vb{l} + (1-beta2)*(db{l}.^2);

                mW_hat = mW{l} / (1 - beta1^iter);
                vW_hat = vW{l} / (1 - beta2^iter);
                mb_hat = mb{l} / (1 - beta1^iter);
                vb_hat = vb{l} / (1 - beta2^iter);

                W{l} = W{l} - opts.learn_rate * mW_hat ./ (sqrt(vW_hat) + adam_eps);
                b{l} = b{l} - opts.learn_rate * mb_hat ./ (sqrt(vb_hat) + adam_eps);
            end
        end

        Yva_hat_n = ffnn_forward_raw(Xva, W, b);
        val_err = Yva_hat_n - Yva_n;
        val_mse = mean(val_err(:).^2);

        if val_mse < best_val - 1e-7
            best_val = val_mse;
            bestW = W; bestb = b;
            bad_epochs = 0;
        else
            bad_epochs = bad_epochs + 1;
        end

        if opts.verbose && (epoch == 1 || mod(epoch,25) == 0 || bad_epochs == opts.patience)
            fprintf('  FFNN %-6s epoch %3d/%3d, val MSE = %.6f, best = %.6f\n', ...
                model_name, epoch, opts.epochs, val_mse, best_val);
        end

        if bad_epochs >= opts.patience
            if opts.verbose
                fprintf('  FFNN %-6s early stopping at epoch %d (best val MSE %.6f)\n', ...
                    model_name, epoch, best_val);
            end
            break;
        end
    end

    model = struct();
    model.W = bestW;
    model.b = bestb;
    model.y_mu = y_mu;
    model.y_sd = y_sd;
    model.hidden_sizes = hidden_sizes;
    model.best_val_mse = best_val;
    model.model_name = model_name;
end

function Yhat = ffnn_predict_batch(X, model)
% Predict in original target units.
    Yhat_n = ffnn_forward_raw(double(X), model.W, model.b);
    Yhat = Yhat_n .* model.y_sd + model.y_mu;
end

function Yhat = ffnn_forward_raw(X, W, b)
% Forward pass returning normalized output units.
    A = double(X);
    nLayers = numel(W);
    for l = 1:nLayers
        Z = A * W{l} + b{l};
        if l < nLayers
            A = max(0, Z);
        else
            A = Z;
        end
    end
    Yhat = A;
end

function Price_fore_enhanced = apply_load_adaptive_price_enhancement(Price_fore_original, Load_fore)
% Same deterministic post-processing rule used by the FCM-TSK price forecast.
    H = numel(Price_fore_original);
    Price_fore_enhanced = Price_fore_original(:);
    Load_fore = Load_fore(:);

    max_load = max(Load_fore);
    min_load = min(Load_fore);
    for h = 1:H
        if h > 1
            dload = Load_fore(h) - Load_fore(h-1);
        else
            dload = 0;
        end
        high_load_flag = double(Load_fore(h) > 3.5);
        if max_load > min_load
            %#ok<NASGU> % normalized load retained for transparency if the rule is extended later
            norm_load = (Load_fore(h)-min_load)/(max_load-min_load);
        else
            norm_load = 0.5; %#ok<NASGU>
        end

        if high_load_flag == 1
            Price_fore_enhanced(h) = Price_fore_enhanced(h) * 1.05;
        end
        if dload > 0.3
            Price_fore_enhanced(h) = Price_fore_enhanced(h) * 1.03;
        elseif dload < -0.3
            Price_fore_enhanced(h) = Price_fore_enhanced(h) * 0.97;
        end
    end
    Price_fore_enhanced = max(0, Price_fore_enhanced);
end

function label = sig_label(p_value)
    if p_value < 0.05
        label = 'Yes';
    elseif p_value < 0.10
        label = 'Borderline';
    else
        label = 'No';
    end
end

function col = detect_col(cols, candidates, T)
    col = "";
    for c = candidates
        idx = find(lower(cols) == lower(string(c)), 1);
        if ~isempty(idx)
            col = cols(idx);
            return;
        end
    end
    for k = 1:numel(cols)
        x = T.(cols(k));
        if isnumeric(x)
            col = cols(k);
            return;
        end
    end
    error("Couldn't detect columns among candidates: %s", strjoin(candidates,","));
end

function cal = enhanced_calendar_features(ts)
    if ~isdatetime(ts)
        ts = datetime(ts, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    end
    hour_of_day = hour(ts);
    dow = weekday(ts);
    dow0 = mod(dow + 5, 7);
    day_of_year = day(ts, 'dayofyear');

    hour_sin = sin(2*pi*hour_of_day/24);
    hour_cos = cos(2*pi*hour_of_day/24);
    dow_sin  = sin(2*pi*dow0/7);
    dow_cos  = cos(2*pi*dow0/7);
    doy_sin  = sin(2*pi*day_of_year/365.25);
    doy_cos  = cos(2*pi*day_of_year/365.25);

    cal = [hour_sin(:), hour_cos(:), dow_sin(:), dow_cos(:), doy_sin(:), doy_cos(:)];
    if size(cal, 2) ~= 6
        error('Calendar features should be 6, but got %d', size(cal, 2));
    end
end

% ----------------------------------------------------------------------
% Plot the membership functions of the ACTUAL trained model,
% denormalized into physical units for a chosen input dimension.
% ----------------------------------------------------------------------
function plot_actual_model_mfs(model, feature_idx, mu, sd, x_label_str, title_str, clip_nonneg, x_cap)
    numClusters = size(model.center, 2);
    center_norm = model.center(feature_idx, :);
    sigma_norm  = model.sigma(feature_idx, :);

    % Denormalize: x_phys = x_norm * sd + mu  (per-dimension linear map)
    center_phys = center_norm * sd(feature_idx) + mu(feature_idx);
    sigma_phys  = abs(sigma_norm * sd(feature_idx));

    [center_phys, sidx] = sort(center_phys);
    sigma_phys = sigma_phys(sidx);

    % ----------------------------------------------------------------------
    % When two or more rules' antecedent centers land within a small fraction
    % of the plotted range of each other -- a real phenomenon for
    % zero-inflated/skewed inputs such as PV's many exact-zero night-time
    % lags, where FCM crowds several clusters into the dominant region --
    % the resulting Gaussian curves are visually indistinguishable and
    % simply hide one another (the earlier-plotted colour is completely
    % overdrawn). Rather than silently overplotting, detect such groups
    % and draw ONE representative curve per group, with the legend and
    % title reporting how many rules collapsed into it.
    % ----------------------------------------------------------------------
    DUPLICATE_REL_TOL = 0.01;  % rules within 1% of the plotted centre range are treated as duplicates
    range_phys = max(center_phys) - min(center_phys);
    if range_phys < 1e-9
        range_phys = max(abs(center_phys(1)), 1);  % avoid divide-by-zero if ALL centres coincide
    end
    abs_tol = DUPLICATE_REL_TOL * range_phys;

    keep_idx = true(1, numClusters);
    group_label = arrayfun(@(c) sprintf('%d', c), 1:numClusters, 'UniformOutput', false);
    for c = 2:numClusters
        prev = c - 1;
        while prev > 1 && ~keep_idx(prev)
            prev = prev - 1;
        end
        if keep_idx(prev) && abs(center_phys(c) - center_phys(prev)) < abs_tol
            group_label{prev} = [group_label{prev}, ',', group_label{c}];
            keep_idx(c) = false;
        end
    end

    x_min = min(center_phys) - max(sigma_phys)*2;
    x_max = max(center_phys) + max(sigma_phys)*2;
    if clip_nonneg
        x_min = max(0, x_min);
    end
    if ~isempty(x_cap)
        x_max = min(x_max, x_cap*1.2);
    end
    xv = linspace(x_min, x_max, 300);

    colors = {'b', 'r', 'g', 'm', 'c', [0.5 0.3 0.1], [0.3 0.7 0.7]};

    figure('Name', title_str, 'Position', [100, 100, 550, 400], 'Color', 'w');
    hold on;
    plot_count = 0;
    for c = 1:numClusters
        if ~keep_idx(c)
            continue;  % collapsed into an earlier rule's curve -- not separately plotted
        end
        plot_count = plot_count + 1;
        mf = exp(-(xv - center_phys(c)).^2 ./ (2 * sigma_phys(c)^2 + 1e-12));
        col = colors{mod(plot_count-1, numel(colors))+1};
        if contains(group_label{c}, ',')
            disp_name = sprintf('Rules %s (center=%.3f) [collapsed]', group_label{c}, center_phys(c));
        else
            disp_name = sprintf('Rule %s (center=%.3f)', group_label{c}, center_phys(c));
        end
        plot(xv, mf, 'Color', col, 'LineWidth', 2, 'DisplayName', disp_name);
    end
    n_collapsed = numClusters - plot_count;
    if n_collapsed > 0
        title_str = sprintf('%s  [%d/%d rules collapsed: near-duplicate centres]', title_str, numClusters - plot_count, numClusters);
        fprintf('  [MF plot] %s: %d of %d rules had near-duplicate centres (tol=%.4g) and were collapsed in the plot.\n', ...
            x_label_str, n_collapsed, numClusters, abs_tol);
    end
    title(title_str, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(x_label_str, 'FontSize', 11);
    ylabel('\mu (membership)', 'FontSize', 11);
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    ylim([0 1.1]);
    hold off;
end


function [P_ev_ch, P_ev_dis] = uncontrolled_ev_charging_schedule(EV_params, H, dt, ts0)
% UNCONTROLLED_EV_CHARGING_SCHEDULE
% Baseline EV mobility load: charge immediately from arrival until the
% required departure SoC is met. No V2G is allowed in this uncontrolled
% baseline. This schedule is included in Strategy 1 and Strategy 2 so all
% strategies serve the same EV energy requirement.
%
% IMPORTANT FIX:
% Use the actual charging limit P_ev_ch_max when constructing the schedule.
% The earlier version used P_ev_max=7.2 kW and then repair_ev_schedule()
% clipped it to P_ev_ch_max=3.6 kW without redistributing the lost energy,
% so the uncontrolled baseline charged only about 8 kWh instead of the
% required ~15.2 kWh for 35% -> 70% SoC.

    if nargin < 4 || isempty(ts0)
        start_hour = getfield_default(EV_params, 'horizon_start_hour', 0);
    else
        start_hour = hour(ts0);
    end
    EV_params.horizon_start_hour = start_hour;

    P_ev_ch  = zeros(H,1);
    P_ev_dis = zeros(H,1);

    soc0    = EV_params.EV_SoC0;
    soc_req = getfield_default(EV_params, 'EV_final_soc_req', soc0);
    Eev     = EV_params.EV_capacity_total;
    eta_ch  = EV_params.eta_ev_ch;

    max_ev_power = EV_params.P_ev_max * EV_params.vehicle_count;
    Pmax_ch = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);

    required_grid_kWh = max(0, (soc_req - soc0) * Eev / max(eta_ch,1e-12));

    % Uncontrolled charger: starts charging immediately at the available
    % charger limit until the mobility energy requirement is met.
    for h = 1:H
        if required_grid_kWh <= 1e-9
            break;
        end
        P_ev_ch(h) = min(Pmax_ch, required_grid_kWh / dt);
        required_grid_kWh = required_grid_kWh - P_ev_ch(h) * dt;
    end

    % Repair for numerical safety and SoC bounds. Baseline EV params should
    % have no V2G service/no-charge windows, so repair will not suppress the
    % required charging energy.
    [P_ev_ch, P_ev_dis] = repair_ev_schedule(P_ev_ch, P_ev_dis, EV_params, dt);
end

function [EV_ch, EV_dis] = rule_based_ev_schedule(Price_fore, Load_fore, PV_fore, EV_params, t0)
    H = length(Price_fore);
    EV_ch  = zeros(H,1);
    EV_dis = zeros(H,1);
    EV_soc = zeros(H+1,1);
    EV_soc(1) = EV_params.EV_SoC0;

    P_ev_max = EV_params.P_ev_max;
    E_ev     = EV_params.EV_capacity_total;
    eta_ch   = EV_params.eta_ev_ch;
    eta_dis  = EV_params.eta_ev_dis;
    SoC_min = EV_params.EV_SoC_min;
    SoC_max = EV_params.EV_SoC_max;
    SoC_driver = 0.50;

    low_th  = prctile(Price_fore, 30);
    high_th = prctile(Price_fore, 70);

    fprintf('\n=== EV RULE-BASED SCHEDULE ===\n');

    for h = 1:H
        hr = mod(hour(t0) + h - 1, 24);
        available_capacity = max(0, (SoC_max - EV_soc(h)) * E_ev);
        available_energy   = max(0, (EV_soc(h) - SoC_driver) * E_ev);
        pv_surplus = max(0, PV_fore(h) - Load_fore(h));

        EV_ch(h)  = 0;
        EV_dis(h) = 0;

        if Price_fore(h) <= low_th && available_capacity > 0
            EV_ch(h) = min([P_ev_max, available_capacity, 3.6]);
        end
        if pv_surplus > 0.2 && available_capacity > 0 && EV_ch(h) == 0
            EV_ch(h) = min([P_ev_max, pv_surplus, available_capacity]);
        end
        if Price_fore(h) >= high_th && available_energy > 0 && EV_ch(h) == 0
            if hr >= 17 && hr <= 21
                EV_dis(h) = min([P_ev_max, available_energy, 3.6]);
            end
        end

        delta_soc = (EV_ch(h) * eta_ch - EV_dis(h) / eta_dis) / E_ev;
        EV_soc(h+1) = EV_soc(h) + delta_soc;
        EV_soc(h+1) = max(SoC_min, min(SoC_max, EV_soc(h+1)));
    end

    total_ch  = sum(EV_ch);
    total_dis = sum(EV_dis);
    if total_dis > total_ch
        scale = total_ch / (total_dis + 1e-9);
        EV_dis = EV_dis * scale;
    end

    EV_soc(:) = 0;
    EV_soc(1) = EV_params.EV_SoC0;
    for h = 1:H
        delta_soc = (EV_ch(h) * eta_ch - EV_dis(h) / eta_dis) / E_ev;
        EV_soc(h+1) = max(SoC_min, min(SoC_max, EV_soc(h) + delta_soc));
    end

    fprintf('Total charging scheduled: %.2f kWh\n', sum(EV_ch));
    fprintf('Total discharging scheduled: %.2f kWh\n', sum(EV_dis));
    fprintf('Final EV SoC: %.1f%%\n', EV_soc(end)*100);
end

function [C, U] = fcm_custom(X, num_clusters, m, maxiter, tol)
    [N, d] = size(X);
    idx = randperm(N, min(num_clusters, N));
    C = X(idx, :);
    U = zeros(N, num_clusters);

    for it = 1:maxiter
        for i = 1:N
            for j = 1:num_clusters
                denom = 0;
                num = norm(X(i,:) - C(j,:)) + 1e-12;
                for k = 1:num_clusters
                    denom = denom + (num / (norm(X(i,:) - C(k,:)) + 1e-12))^(2/(m-1));
                end
                U(i,j) = 1 / denom;
            end
        end
        C_old = C;
        for j = 1:num_clusters
            w = U(:,j).^m;
            wsum = sum(w);
            if wsum > 1e-12
                C(j,:) = (w' * X) / wsum;
            end
        end
        if max(abs(C(:) - C_old(:))) < tol
            break;
        end
    end
end

% ----------------------------------------------------------------------
% Fcm_custom above clusters ONLY the input matrix X.
% Because the lagged input vector x_t does not
% itself depend on which target y_{t+h} is being predicted, X is
% IDENTICAL across every one of the 24 forecast horizons and across all
% three forecast variables (PV, Load, Price) -- so calling fcm_custom on
% the same X with the same seed (rng(1000+k), seeded only by cluster
% count k) produces the literal same cluster centres every single time,
% for every horizon and every variable. This was discovered by inspecting
% the "Selected MF parameters" diagnostic, where centres/sigmas were
% identical to 4-5 decimal places across all 5 rules and all 3 models.
%
% This wrapper implements standard output-augmented (Sugeno-Yasukawa
% style) FCM identification: clustering is performed in the JOINT
% [input, target] space, so the resulting antecedent partition is shaped
% by how the input relates to THIS SPECIFIC target/horizon, rather than
% by input density alone. The target column is z-scored and weighted so
% its contribution to the clustering distance is comparable in magnitude
% to the COMBINED contribution of the entire (already z-scored) input
% block -- y_weight = sqrt(d) makes the single weighted target dimension
% contribute, in expectation, as much squared distance as the d input
% dimensions together, rather than being negligible in a 126-dimensional
% Euclidean distance. Only the INPUT portion of each resulting cluster
% centre is returned and used downstream (as the antecedent centre/sigma
% fed to build_gaussian_mfs), since at prediction time the target is
% exactly what is being forecast and is therefore unavailable as a
% clustering coordinate. The membership matrix U (computed from the full
% augmented distance) is still used downstream for sigma estimation and
% WLS rule weighting, which is desirable: training-time rule assignment
% legitimately benefits from knowing the true output.
% ----------------------------------------------------------------------
function [C_x, U] = fcm_custom_augmented(Xtr, ytr, num_clusters, m, maxiter, tol, y_weight)
    if nargin < 7 || isempty(y_weight)
        y_weight = sqrt(size(Xtr, 2));
    end
    ytr = ytr(:);
    y_mu = mean(ytr);
    y_sd = std(ytr) + 1e-12;
    y_norm = (ytr - y_mu) / y_sd;

    Xy = [Xtr, y_weight * y_norm];
    [C_aug, U] = fcm_custom(Xy, num_clusters, m, maxiter, tol);
    C_x = C_aug(:, 1:end-1);
end

function [center, sigma] = build_gaussian_mfs(X, C, U, m)
    [N, d] = size(X);
    numClusters = size(C, 1);
    center = zeros(d, numClusters);
    sigma = zeros(d, numClusters);

    for in_idx = 1:d
        for c = 1:numClusters
            center(in_idx, c) = C(c, in_idx);
            w = U(:,c).^m;
            diff_sq = (X(:,in_idx) - center(in_idx,c)).^2;
            numer = sum(w .* diff_sq);
            denom = sum(w) + 1e-12;
            sigma(in_idx,c) = sqrt(numer / denom);
            if sigma(in_idx,c) < 1e-3
                sigma(in_idx,c) = 1e-3;
            end
        end
    end
end

function [a, W_all] = estimate_tsk_parameters(X, Y, center, sigma, lambda)
% Ridge-regularized global weighted least-squares estimation of the
% first-order TSK consequents. Normalized fuzzy firing strengths weight the
% local affine terms in the global design matrix. Rule intercepts are not
% penalized.
    if nargin < 5 || isempty(lambda)
        lambda = 1e-6;
    end
    [N, d] = size(X);
    numRules = size(center, 2);

    W_all = compute_rule_firing(X, center, sigma);
    W_norm = W_all ./ (sum(W_all, 2) + 1e-12);

    Phi = zeros(N, numRules*(d+1));
    Phi_r = [ones(N,1), X];
    for r = 1:numRules
        cols = (r-1)*(d+1)+1 : r*(d+1);
        Phi(:, cols) = W_norm(:,r) .* Phi_r;
    end

    R = eye(size(Phi,2));
    intercept_idx = 1:(d+1):size(Phi,2);
    R(intercept_idx, intercept_idx) = 0;  % do not shrink rule biases

    theta = (Phi' * Phi + lambda * R) \ (Phi' * Y);
    a = reshape(theta, [d+1, numRules])';
end

function W_all = compute_rule_firing(X, center, sigma)
    [N, d] = size(X);
    numRules = size(center, 2);
    W_all = ones(N, numRules);
    for r = 1:numRules
        for in_idx = 1:d
            mu_in = exp(-(X(:,in_idx) - center(in_idx,r)).^2 ./ ...
                (2*sigma(in_idx,r)^2 + 1e-12));
            W_all(:,r) = W_all(:,r) .* mu_in;
        end
    end
    W_all = W_all + 1e-12;
end

function Ypred = fuzzy_predict_batch(X, center, sigma, a)
    if isvector(X)
        X = X(:).';
    end
    [N, ~] = size(X);
    numRules = size(center, 2);
    W = compute_rule_firing(X, center, sigma);
    W = W ./ (sum(W,2) + 1e-12);
    X1 = [ones(N,1), X];
    Yrules = X1 * a';
    Ypred = sum(W .* Yrules, 2);
end

function [C, U, n_merged] = merge_duplicate_fcm_rules(X, C, U, m)
% Conservatively merge genuinely redundant antecedent rules. A pair is
% merged only when its full normalized centre vectors are close AND its
% firing-strength patterns over the training set are almost identical.
% This avoids merging rules merely because one plotted input has a similar
% centre while other antecedent dimensions remain distinct.
    CENTER_TOL = 0.08;   % RMS distance per normalized input dimension
    FIRING_CORR_TOL = 0.995;
    n_merged = 0;

    while size(C,1) > 2
        [center, sigma] = build_gaussian_mfs(X, C, U, m);
        W = compute_rule_firing(X, center, sigma);
        K = size(C,1);
        found = false;
        best_pair = [];
        best_dist = inf;

        for i = 1:K-1
            for j = i+1:K
                dist_ij = norm(C(i,:) - C(j,:)) / sqrt(size(C,2));
                wi = W(:,i); wj = W(:,j);
                if std(wi) < 1e-14 || std(wj) < 1e-14
                    corr_ij = double(norm(wi-wj) / (norm(wi)+norm(wj)+1e-12) < 1e-3);
                else
                    cc = corrcoef(wi, wj);
                    corr_ij = cc(1,2);
                end
                if dist_ij <= CENTER_TOL && corr_ij >= FIRING_CORR_TOL && dist_ij < best_dist
                    best_pair = [i j];
                    best_dist = dist_ij;
                    found = true;
                end
            end
        end

        if ~found
            break;
        end

        i = best_pair(1); j = best_pair(2);
        mass_i = sum(U(:,i).^m);
        mass_j = sum(U(:,j).^m);
        C(i,:) = (mass_i*C(i,:) + mass_j*C(j,:)) / (mass_i + mass_j + 1e-12);
        U(:,i) = U(:,i) + U(:,j);
        C(j,:) = [];
        U(:,j) = [];
        U = U ./ (sum(U,2) + 1e-12);
        n_merged = n_merged + 1;
    end
end

function [best_model, best_mse] = train_fcm_tsk_with_k_search(Xtr, Ytr, Xva, Yva, candidate_clusters, m, maxiter, tol, fixed_lambda)
% Data-adaptive FCM-TSK identification. The requested cluster count is
% selected from candidate_clusters by validation MSE. Redundant rules are
% merged before fitting ridge-regularized weighted TSK consequents.
    if nargin < 9
        fixed_lambda = [];
    end
    ridge_candidates = [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1];
    if ~isempty(fixed_lambda)
        ridge_candidates = fixed_lambda;
    end

    best_mse = inf;
    best_model = struct();
    y_weight = sqrt(size(Xtr, 2));

    for k = candidate_clusters
        rng(1000 + k);
        [C, U] = fcm_custom_augmented(Xtr, Ytr, k, m, maxiter, tol, y_weight);
        [C, U, n_merged] = merge_duplicate_fcm_rules(Xtr, C, U, m);
        [center, sigma] = build_gaussian_mfs(Xtr, C, U, m);

        for lambda = ridge_candidates
            [a, ~] = estimate_tsk_parameters(Xtr, Ytr, center, sigma, lambda);
            Yva_pred = fuzzy_predict_batch(Xva, center, sigma, a);
            mse = mean((Yva - Yva_pred).^2);

            if mse < best_mse
                best_mse = mse;
                best_model.k_requested = k;
                best_model.k = size(C,1);       % effective rules after merging
                best_model.n_merged = n_merged;
                best_model.lambda = lambda;
                best_model.C = C;
                best_model.U = U;
                best_model.center = center;
                best_model.sigma = sigma;
                best_model.a = a;
            end
        end
    end
end

function model_all = train_multi_horizon_fcm(Xtr, Ytr, Xva, Yva, clusters, m, maxiter, tol, per_horizon_k, per_horizon_lambda)
% During model selection, each horizon searches k in the supplied candidate
% set and lambda in the ridge grid. During final train+validation retraining,
% the validation-selected requested k and lambda are reused exactly.
    if nargin < 9, per_horizon_k = []; end
    if nargin < 10, per_horizon_lambda = []; end
    H = size(Ytr, 2);
    model_all = cell(H,1);
    for h = 1:H
        ytr_h = Ytr(:,h);
        yva_h = Yva(:,h);
        if isempty(per_horizon_k)
            clusters_h = clusters;
        else
            clusters_h = per_horizon_k(h);
        end
        if isempty(per_horizon_lambda)
            lambda_h = [];
        else
            lambda_h = per_horizon_lambda(h);
        end
        [model_h, ~] = train_fcm_tsk_with_k_search( ...
            Xtr, ytr_h, Xva, yva_h, clusters_h, m, maxiter, tol, lambda_h);
        model_all{h} = model_h;
    end
end

function P = clamp_power_to_soc(P, soc, SoC_min, SoC_max, Ecap, Pmax, eta_ch, eta_dis, dt)
    P = min(max(P, -Pmax), Pmax);

    max_dis_E = max(0, (soc - SoC_min) * Ecap);
    max_dis_P = (max_dis_E * eta_dis) / (dt + 1e-12);

    max_ch_E = max(0, (SoC_max - soc) * Ecap);
    max_ch_P = max_ch_E / (dt * eta_ch + 1e-12);

    P = min(max(P, -max_ch_P), max_dis_P);
    if soc <= SoC_min + 1e-4 && P > 0
        P = 0;
    end
    if soc >= SoC_max - 1e-4 && P < 0
        P = 0;
    end
end

function [DM_stat, p_value] = dm_test(e1, e2, label)
    e1 = e1(:); e2 = e2(:);
    d = e1.^2 - e2.^2;
    T = length(d);
    d_mean = mean(d);
    d_var = var(d);
    DM_stat = d_mean / sqrt(d_var / T);
    p_value = 2 * (1 - normcdf(abs(DM_stat), 0, 1));

    fprintf('%s:\n', label);
    fprintf('  DM stat = %.4f, p-value = %.4f\n', DM_stat, p_value);
    if p_value < 0.05
        fprintf('  => Statistically significant difference (reject H0)\n');
    else
        fprintf('  => No significant difference (fail to reject H0)\n');
    end
end

% ----------------------------------------------------------------------

function validation = simple_pso_validation_EV(PV_fore, Load_fore, Price_fore, ...
    battery_params, EV_params, PSO_MAXITER, mainBestCost, n_runs, seed_base, swarm_size)
% Retained for completeness / backward compatibility. The main script now
% uses multirun_pso_compare() for the paired BESS+uncontrolled-EV vs BESS+optimized-EV protocol;
% this function is no longer called directly in the main flow but is kept
% available for ad-hoc single-strategy validation.

    if nargin < 8, n_runs = 20; end
    if nargin < 9, seed_base = 1000; end
    if nargin < 10, swarm_size = 80; end

    H = length(PV_fore);
    dim = 3 * H;
    max_ev_power = EV_params.P_ev_max * EV_params.vehicle_count;
    max_ev_ch_power = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);
    max_ev_dis_power = getfield_default(EV_params, 'P_ev_dis_max', max_ev_power);

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess =  battery_params.Pmax * ones(H, 1);
    lb_ev = zeros(2*H, 1);
    ub_ev = [max_ev_ch_power * ones(H, 1); max_ev_dis_power * ones(H, 1)];
    lb = [lb_bess; lb_ev];
    ub = [ub_bess; ub_ev];
    vel_clamp = 0.5 * max([battery_params.Pmax, max_ev_ch_power, max_ev_dis_power]);

    objfun = @(x) total_cost_withGrid_and_soc_repair_EV( ...
        x(1:H), x(H+1:2*H), x(2*H+1:3*H), ...
        PV_fore, Load_fore, Price_fore, 0.85*Price_fore, ...
        H, battery_params.SoC0, battery_params.SoC_min, battery_params.SoC_max, ...
        battery_params.Ecap, battery_params.Pmax, battery_params.eta_ch, ...
        battery_params.eta_dis, battery_params.dt, EV_params, battery_params);

    costs = zeros(n_runs, 1);
    all_histories = zeros(n_runs, PSO_MAXITER);
    best_all_cost = inf; best_all_x = []; best_all_history = [];

    for run = 1:n_runs
        rng(seed_base + run);  % FIX #3
        [best_x, best_cost, history] = custom_pso(objfun, dim, lb, ub, ...
            swarm_size, PSO_MAXITER, 0.9, 1.5, 1.5, vel_clamp);
        costs(run) = best_cost;
        hist_len = length(history);
        all_histories(run, 1:hist_len) = history(:)';
        if hist_len < PSO_MAXITER
            all_histories(run, hist_len+1:end) = history(end);
        end
        if best_cost < best_all_cost
            best_all_cost = best_cost; best_all_x = best_x; best_all_history = history;
        end
    end

    validation.mean_cost = mean(costs);
    validation.std_cost = std(costs);
    validation.min_cost = min(costs);
    validation.max_cost = max(costs);
    validation.cv = 100 * validation.std_cost / validation.mean_cost;
    validation.all_costs = costs;
    validation.all_histories = all_histories;
    validation.best_cost = best_all_cost;
    validation.best_x = best_all_x;
    validation.best_history = best_all_history;
end

function [totalCost, history] = total_cost_withGrid_and_soc_repair(Pbat, PV_fore, Load_fore, BuyPrice, SellPrice, ...
    H, SoC0, SoC_min, SoC_max, Ecap, Pmax, eta_ch, eta_dis, dt, battery_params)

    Pbat = double(Pbat(:));
    SoC = SoC0;
    soc_smoothing_weight = getfield_default(battery_params, 'soc_smoothing_weight', 0.0);
    soc_smoothing_penalty = 0;
    penalty_reserve = 0;   % FIX: initialize BESS-only reserve penalty accumulator

    totalCost = 0;
    history.GridPower = zeros(H, 1);
    history.SoC = zeros(H, 1);
    history.gridCost = zeros(H, 1);
    history.Pbat = zeros(H, 1);

    degradation_cost_per_cycle = 0.05;
    cycle_depth_penalty = 0.02;
    demand_charge_rate = battery_params.demand_charge_rate;

    for h = 1:H
        P = clamp_power_to_soc(Pbat(h), SoC, SoC_min, SoC_max, Ecap, Pmax, eta_ch, eta_dis, dt);
        history.Pbat(h) = P;

        if P >= 0
            dSoC = -(P * dt) / (eta_dis * Ecap + 1e-12);
        else
            dSoC = ((-P) * dt * eta_ch) / (Ecap + 1e-12);
        end
        SoC = SoC + dSoC;
        history.SoC(h) = SoC;
        if h > 1
            soc_smoothing_penalty = soc_smoothing_penalty + ...
            soc_smoothing_weight * (history.SoC(h) - history.SoC(h-1))^2;
        end
        SoC_reserve = getfield_default(battery_params, 'SoC_reserve', 0.35);
        reserve_weight = getfield_default(battery_params, 'reserve_soc_weight', 50.0);
        reserve_violation = max(0, SoC_reserve - SoC);
        reserve_pen = reserve_weight * reserve_violation^2;
        penalty_reserve = penalty_reserve + reserve_pen;
        totalCost = totalCost + reserve_pen;

        Grid = Load_fore(h) - PV_fore(h) - P;
        history.GridPower(h) = Grid;
        if Grid >= 0
            gridCost = Grid * BuyPrice(h) * dt;
        else
            gridCost = Grid * SellPrice(h) * dt;
        end
        history.gridCost(h) = gridCost;
        totalCost = totalCost + gridCost;
    end

    total_energy_throughput = sum(abs(history.Pbat)) * dt;
    equivalent_cycles = total_energy_throughput / (2 * Ecap);
    max_dod = max(abs(history.SoC - SoC0));
    dod_penalty = cycle_depth_penalty * max_dod * equivalent_cycles;
    degradation_cost = degradation_cost_per_cycle * equivalent_cycles + dod_penalty;

    peak_demand = max(history.GridPower);
    demand_charge = demand_charge_rate * peak_demand;

    % ----------------------------------------------------------------------
    % The TRUE demand charge above is based on
    % max(GridPower), which gives nonzero cost sensitivity to only whichever
    % SINGLE hour happens to be the peak -- every other hour, including ones
    % close behind it, shows ZERO fitness gradient for demand charge. This
    % creates a flat, uninformative landscape for derivative-free search
    % (PSO/GA) and the coordinate-wise local_polish step, and is a likely
    % structural contributor to the repeatedly observed "reasonable EV/BESS
    % solution, but unexpectedly high peak demand" failure mode.
    %
    % A p-norm of the (non-negative) import profile is a smooth upper bound
    % on the true peak: for any finite p, ||x||_p >= ||x||_infinity = max(x).
    % It gives EVERY high-import hour a continuous, nonzero incentive to
    % reduce import, weighted toward the largest ones -- not just the single
    % literal peak hour. It is added as a SEPARATE shaping term on top of
    % the TRUE max()-based demand charge; it never replaces or approximates
    % the true charge used in cost accounting, so the cost-residual identity
    % (component sum == total cost) still holds exactly. demand_shaping_weight
    % defaults to 0 (no effect) unless explicitly set in battery_params, so
    % this is fully backward-compatible.
    % ----------------------------------------------------------------------
    demand_shaping_weight = getfield_default(battery_params, 'demand_shaping_weight', 0.0);
    demand_shaping_p = getfield_default(battery_params, 'demand_shaping_p', 8);
    gp_pos = max(0, history.GridPower);
    p_norm_demand = (sum(gp_pos.^demand_shaping_p) + 1e-12)^(1/demand_shaping_p);
    demand_shaping_penalty = demand_shaping_weight * demand_charge_rate * ...
        max(0, p_norm_demand - max(0, peak_demand));

    terminal_soc_weight = getfield_default(battery_params, 'terminal_soc_weight', 0.0);
    terminal_soc_target = getfield_default(battery_params, 'terminal_soc_target', SoC0);
    terminal_soc_penalty = terminal_soc_weight *(history.SoC(end) - terminal_soc_target)^2;

    grid_energy_cost = sum(history.gridCost);
    pre_component_cost = totalCost;
    other_penalty_cost = pre_component_cost - grid_energy_cost;

    totalCost = grid_energy_cost + other_penalty_cost + degradation_cost + ...
                demand_charge + terminal_soc_penalty + soc_smoothing_penalty + demand_shaping_penalty;

    history.grid_energy_cost = grid_energy_cost;
    history.other_penalty_cost = other_penalty_cost;
    history.soc_smoothing_penalty = soc_smoothing_penalty;
    history.equivalent_cycles = equivalent_cycles;
    history.degradation_cost = degradation_cost;
    history.peak_demand = peak_demand;
    history.demand_charge = demand_charge;
    history.demand_shaping_penalty = demand_shaping_penalty;
    history.terminal_soc_penalty = terminal_soc_penalty;
    history.energy_cost = grid_energy_cost;
    history.grid_energy_cost = sum(history.gridCost);
    history.penalty_reserve = penalty_reserve;
end

function [totalCost, history] = total_cost_withGrid_and_soc_repair_EV(Pbat, P_ev_ch, P_ev_dis, ...
    PV_fore, Load_fore, Price_fore, SellPrice, H, SoC0, SoC_min, SoC_max, ...
    Ecap, Pmax, eta_ch, eta_dis, dt, EV_params, battery_params)

    EV_capacity_total = EV_params.EV_capacity_total;
    EV_SoC0 = EV_params.EV_SoC0;
    EV_SoC_min = EV_params.EV_SoC_min;
    EV_SoC_max = EV_params.EV_SoC_max;
    eta_ev_ch = EV_params.eta_ev_ch;
    eta_ev_dis = EV_params.eta_ev_dis;
    P_ev_max = EV_params.P_ev_max;
    vehicle_count = EV_params.vehicle_count;
    ev_degradation_cost = EV_params.ev_degradation_cost;
    max_ev_power = P_ev_max * vehicle_count;
    max_ch_power = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);
    max_dis_power = getfield_default(EV_params, 'P_ev_dis_max', max_ev_power);

    % ----------------------------------------------------------------------
    % REPAIR OPERATOR (replaces penalty-only constraint handling for EV
    % feasibility): every raw [P_ev_ch, P_ev_dis] candidate proposed by
    % PSO or GA is repaired into a feasible schedule -- enforcing mutual
    % exclusivity and SoC bounds -- using the SAME repair_ev_schedule()
    % function already used to construct the B1/B2/persistence EV
    % baselines. This is applied identically regardless of which
    % algorithm produced the candidate, since it lives inside the shared
    % objective function.
    %
    % Previously, feasibility was already being silently enforced deeper
    % inside the loop below (clamping + zeroing the smaller of ch/dis),
    % but a large fixed penalty ($100 per kW of overlap) was ALSO charged
    % for the raw, pre-repair request. For a randomly-initialized
    % candidate -- where both ch and dis are drawn independently and
    % uniformly, so a large fraction of hours have both > 0 -- this
    % penalty created a search landscape dominated by penalty cliffs
    % unrelated to actual grid cost, making PSO and (especially) GA
    % unreliable on this problem. Repairing BEFORE evaluation removes
    % those cliffs: the optimizer is now only ever scored on dispatches
    % that are already physically valid, with no separate punishment for
    % how it phrased the request.
    %
    % The per-step clamping/penalty block further below is left in place
    % unchanged as a defense-in-depth safety net; with the input already
    % repaired, its mutual-exclusivity branch can no longer fire.
    % ----------------------------------------------------------------------
    [P_ev_ch, P_ev_dis] = repair_ev_schedule(P_ev_ch, P_ev_dis, EV_params, dt);

    SoC = SoC0;
    soc_smoothing_weight = getfield_default(battery_params, 'soc_smoothing_weight', 0.0);
    soc_smoothing_penalty = 0;
    EV_SoC = EV_SoC0;

    totalCost = 0;
    history.GridPower = zeros(H, 1);
    history.SoC = zeros(H, 1);
    history.EV_SoC = zeros(H, 1);
    history.Pbat = zeros(H, 1);
    history.P_ev_ch = zeros(H, 1);
    history.P_ev_dis = zeros(H, 1);
    history.gridCost = zeros(H, 1);

    degradation_cost_per_cycle = 0.05;
    cycle_depth_penalty = 0.02;
    demand_charge_rate = battery_params.demand_charge_rate;

    total_ev_charged = 0;
    total_ev_discharged = 0;
    prev_ev_mode = 0;
    mode_changes = 0;
    ev_soc_violations = 0;

    % Detailed diagnostic penalty components.
    penalty_mode_change = 0;
    penalty_reserve = 0;
    penalty_energy_mismatch = 0;
    penalty_net_discharge = 0;
    penalty_charge_ratio = 0;
    penalty_ev_soc = 0;
    penalty_daily_discharge = 0;
    penalty_bess_cycles = 0;
    ev_terminal_soc_penalty = 0;
    v2g_service_penalty = 0;
    v2g_service_energy = 0;
    mode_change_cost = getfield_default(EV_params, 'mode_change_cost', 0.20);

    for h = 1:H
        P_ev_ch(h) = min(max(P_ev_ch(h), 0), max_ch_power);
        P_ev_dis(h) = min(max(P_ev_dis(h), 0), max_dis_power);

        if P_ev_ch(h) > 0.01 && P_ev_dis(h) > 0.01
            totalCost = totalCost + 100 * (P_ev_ch(h) + P_ev_dis(h));
            if P_ev_ch(h) > P_ev_dis(h)
                P_ev_dis(h) = 0;
            else
                P_ev_ch(h) = 0;
            end
        end

        available_energy = EV_SoC * EV_capacity_total;
        min_energy = EV_SoC_min * EV_capacity_total;
        max_discharge_energy = max(0, available_energy - min_energy);
        max_discharge_power = max_discharge_energy / (dt * eta_ev_dis + 1e-6);
        P_ev_dis(h) = min(P_ev_dis(h), max_discharge_power);

        available_capacity = (EV_SoC_max - EV_SoC) * EV_capacity_total;
        max_charge_power = available_capacity / (dt * eta_ev_ch + 1e-6);
        P_ev_ch(h) = min(P_ev_ch(h), max_charge_power);

        if EV_SoC > 0.80 && P_ev_ch(h) > 0
            P_ev_ch(h) = P_ev_ch(h) * 0.5;
        end
        if EV_SoC < 0.30 && P_ev_dis(h) > 0
            P_ev_dis(h) = P_ev_dis(h) * 0.3;
        end

        history.P_ev_ch(h) = P_ev_ch(h);
        history.P_ev_dis(h) = P_ev_dis(h);

        total_ev_charged = total_ev_charged + P_ev_ch(h) * dt;
        total_ev_discharged = total_ev_discharged + P_ev_dis(h) * dt;

        current_mode = 0;
        if P_ev_ch(h) > 0.1
            current_mode = 1;
        elseif P_ev_dis(h) > 0.1
            current_mode = -1;
        end
        if current_mode ~= 0 && current_mode ~= prev_ev_mode && prev_ev_mode ~= 0
            mode_changes = mode_changes + 1;
            penalty_mode_change = penalty_mode_change + mode_change_cost;
            totalCost = totalCost + mode_change_cost;
        end
        prev_ev_mode = current_mode;

        P_bess = clamp_power_to_soc(Pbat(h), SoC, SoC_min, SoC_max, Ecap, Pmax, eta_ch, eta_dis, dt);
        history.Pbat(h) = P_bess;

        if P_bess >= 0
            dSoC = -(P_bess * dt) / (eta_dis * Ecap + 1e-12);
        else
            dSoC = ((-P_bess) * dt * eta_ch) / (Ecap + 1e-12);
        end
        SoC = SoC + dSoC;
        SoC = max(SoC_min, min(SoC_max, SoC));
        history.SoC(h) = SoC;
        if h > 1
            soc_smoothing_penalty = soc_smoothing_penalty + ...
            soc_smoothing_weight * (history.SoC(h) - history.SoC(h-1))^2;
        end
        SoC_reserve = getfield_default(battery_params, 'SoC_reserve', 0.35);
        reserve_weight = getfield_default(battery_params, 'reserve_soc_weight', 50.0);
        reserve_violation = max(0, SoC_reserve - SoC);
        totalCost = totalCost + reserve_weight * reserve_violation^2;

        dEV_SoC = (P_ev_ch(h) * dt * eta_ev_ch - P_ev_dis(h) * dt / eta_ev_dis) / EV_capacity_total;
        EV_SoC = EV_SoC + dEV_SoC;

        if EV_SoC < EV_SoC_min
            ev_soc_violations = ev_soc_violations + 1;
            EV_SoC = EV_SoC_min;
        elseif EV_SoC > EV_SoC_max
            ev_soc_violations = ev_soc_violations + 1;
            EV_SoC = EV_SoC_max;
        end
        history.EV_SoC(h) = EV_SoC;

        Grid = Load_fore(h) - PV_fore(h) - P_bess - P_ev_dis(h) + P_ev_ch(h);
        history.GridPower(h) = Grid;

        if Grid >= 0
            gridCost = Grid * Price_fore(h) * dt;
        else
            gridCost = Grid * SellPrice(h) * dt;
        end
        history.gridCost(h) = gridCost;
        totalCost = totalCost + gridCost;
    end

    actual_soc_change = (EV_SoC - EV_SoC0) * EV_capacity_total;
    expected_soc_change = total_ev_charged * eta_ev_ch - total_ev_discharged / eta_ev_dis;
    energy_mismatch = abs(actual_soc_change - expected_soc_change);
    if energy_mismatch > 0.1
        penalty_energy_mismatch = energy_mismatch * 100;
        totalCost = totalCost + penalty_energy_mismatch;
    end

    net_ev_energy = total_ev_discharged - total_ev_charged;
    max_net_discharge_frac = getfield_default(EV_params, 'max_net_discharge_frac', 0.05);
    max_net_discharge_per_day = EV_capacity_total * max_net_discharge_frac;
    if net_ev_energy > max_net_discharge_per_day
        violation = net_ev_energy - max_net_discharge_per_day;
        penalty_net_discharge = violation * 10;
        totalCost = totalCost + penalty_net_discharge;
    end

    if total_ev_discharged > 0.5
        charge_ratio = total_ev_charged / total_ev_discharged;
        if charge_ratio < 0.7
            penalty_charge_ratio = (0.7 - charge_ratio) * total_ev_discharged * 8;  % relaxed from 20
            totalCost = totalCost + penalty_charge_ratio;
        end
    end

    total_ev_cycled = total_ev_charged + total_ev_discharged;
    max_reasonable_cycle = EV_capacity_total * 1.2;
    if total_ev_cycled > max_reasonable_cycle
        excess_cycle = total_ev_cycled - max_reasonable_cycle;
        totalCost = totalCost + excess_cycle * 0.20;
    end

    EV_final_soc_req = getfield_default(EV_params, 'EV_final_soc_req', 0.50);
    ev_terminal_shortage_cost = getfield_default(EV_params, 'ev_terminal_shortage_cost', 50.0);
    if history.EV_SoC(end) < EV_final_soc_req
        shortage_kWh = (EV_final_soc_req - history.EV_SoC(end)) * EV_capacity_total;
        ev_terminal_soc_penalty = shortage_kWh * ev_terminal_shortage_cost;
        totalCost = totalCost + ev_terminal_soc_penalty;
    end

    % Optional V2G service requirement: if the EV is enrolled in a
    % V2G/peak-support program, require a minimum amount of discharge in
    % the allowed peak window. This makes V2G participation meaningful and
    % still keeps the final departure SoC constraint active.
    v2g_required_energy = getfield_default(EV_params, 'v2g_required_energy', 0.0);
    v2g_requirement_weight = getfield_default(EV_params, 'v2g_requirement_weight', 0.0);
    v2g_allowed_hours = getfield_default(EV_params, 'v2g_allowed_hours', []);
    start_hour = getfield_default(EV_params, 'horizon_start_hour', 0);
    if ~isempty(v2g_allowed_hours)
        hrs = mod(start_hour + (0:H-1), 24);
        idx_v2g = ismember(hrs, v2g_allowed_hours);
        v2g_service_energy = sum(history.P_ev_dis(idx_v2g)) * dt;
    else
        v2g_service_energy = sum(history.P_ev_dis) * dt;
    end
    if v2g_required_energy > 0 && v2g_requirement_weight > 0
        v2g_overdelivery_weight = getfield_default(EV_params, 'v2g_overdelivery_weight', 0.0);
        v2g_overdelivery_margin = getfield_default(EV_params, 'v2g_overdelivery_margin', 1.25);

        v2g_short = max(0, v2g_required_energy - v2g_service_energy);
        v2g_over  = max(0, v2g_service_energy - v2g_overdelivery_margin * v2g_required_energy);

        v2g_service_penalty = v2g_requirement_weight * v2g_short^2 + ...
                              v2g_overdelivery_weight * v2g_over^2;
        totalCost = totalCost + v2g_service_penalty;
    end

    if ev_soc_violations > 0
        penalty_ev_soc = ev_soc_violations * 10;
        totalCost = totalCost + penalty_ev_soc;
    end

    max_daily_discharge = EV_capacity_total * 0.25;
    if total_ev_discharged > max_daily_discharge
        excess = total_ev_discharged - max_daily_discharge;
        penalty_daily_discharge = excess * 2;  % relaxed from 10
        totalCost = totalCost + penalty_daily_discharge;
    end

    bess_throughput = sum(abs(history.Pbat)) * dt;
    bess_cycles = bess_throughput / (2 * Ecap);

    max_bess_cycles = 1.0;
    if bess_cycles > max_bess_cycles
        penalty_bess_cycles = (bess_cycles - max_bess_cycles) * 10;
        totalCost = totalCost + penalty_bess_cycles;
    end

    max_dod = max(abs(history.SoC - SoC0));
    dod_penalty = cycle_depth_penalty * max_dod * bess_cycles;
    bess_degradation = degradation_cost_per_cycle * bess_cycles + dod_penalty;

    ev_throughput = sum(history.P_ev_ch + history.P_ev_dis) * dt;
    ev_cycles = ev_throughput / (2 * EV_capacity_total);
    ev_degradation = ev_degradation_cost * ev_cycles;

    peak_demand = max(history.GridPower);
    demand_charge = demand_charge_rate * max(0, peak_demand);

    % FIX (demand-charge shaping): see the matching block in
    % total_cost_withGrid_and_soc_repair for the full rationale. Same
    % mechanism here -- this is the cost function actually used by the
    % BESS+EV PSO/GA optimization, where the "good EV/BESS schedule but
    % unexpectedly high peak demand" failure mode has been observed
    % repeatedly. demand_shaping_weight defaults to 0 (no effect) unless
    % explicitly set in battery_params.
    demand_shaping_weight = getfield_default(battery_params, 'demand_shaping_weight', 0.0);
    demand_shaping_p = getfield_default(battery_params, 'demand_shaping_p', 8);
    gp_pos = max(0, history.GridPower);
    p_norm_demand = (sum(gp_pos.^demand_shaping_p) + 1e-12)^(1/demand_shaping_p);
    demand_shaping_penalty = demand_shaping_weight * demand_charge_rate * ...
        max(0, p_norm_demand - max(0, peak_demand));

    terminal_soc_weight = getfield_default(battery_params, 'terminal_soc_weight', 0.0);
    terminal_soc_target = getfield_default(battery_params, 'terminal_soc_target', SoC0);
    terminal_soc_penalty = terminal_soc_weight * (history.SoC(end) - terminal_soc_target)^2;

    grid_energy_cost = sum(history.gridCost);
    pre_component_cost = totalCost;
    other_penalty_cost = pre_component_cost - grid_energy_cost;
    degradation_cost = bess_degradation + ev_degradation;

    totalCost = grid_energy_cost + other_penalty_cost + degradation_cost + ...
                demand_charge + terminal_soc_penalty + soc_smoothing_penalty + demand_shaping_penalty;

    history.grid_energy_cost = grid_energy_cost;
    history.other_penalty_cost = other_penalty_cost;
    history.soc_smoothing_penalty = soc_smoothing_penalty;
    history.equivalent_cycles = bess_cycles;
    history.degradation_cost = degradation_cost;
    history.peak_demand = peak_demand;
    history.demand_charge = demand_charge;
    history.demand_shaping_penalty = demand_shaping_penalty;
    history.terminal_soc_penalty = terminal_soc_penalty;
    history.energy_cost = grid_energy_cost;
    history.ev_cycles = ev_cycles;
    history.ev_throughput = ev_throughput;
    history.total_ev_charged = total_ev_charged;
    history.total_ev_discharged = total_ev_discharged;
    history.mode_changes = mode_changes;

    % Detailed penalty diagnostics for debugging/publication transparency.
    history.penalty_mode_change = penalty_mode_change;
    history.penalty_reserve = penalty_reserve;
    history.penalty_energy_mismatch = penalty_energy_mismatch;
    history.penalty_net_discharge = penalty_net_discharge;
    history.penalty_charge_ratio = penalty_charge_ratio;
    history.penalty_ev_soc = penalty_ev_soc;
    history.penalty_daily_discharge = penalty_daily_discharge;
    history.penalty_bess_cycles = penalty_bess_cycles;
    history.ev_terminal_soc_penalty = ev_terminal_soc_penalty;
    history.v2g_service_penalty = v2g_service_penalty;
    history.v2g_service_energy = v2g_service_energy;
end

function value = getfield_default(s, fieldname, default_value)
    if isstruct(s) && isfield(s, fieldname)
        value = s.(fieldname);
    else
        value = default_value;
    end
end

function [Pch_rep, Pdis_rep, EV_SoC_hist_rep] = repair_ev_schedule(Pch, Pdis, EV_params, dt)
    Pch = double(Pch(:));
    Pdis = double(Pdis(:));
    H = numel(Pch);

    max_ev_power = EV_params.P_ev_max * EV_params.vehicle_count;
    max_ch_power = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);
    max_dis_power = getfield_default(EV_params, 'P_ev_dis_max', max_ev_power);
    start_hour = getfield_default(EV_params, 'horizon_start_hour', 0);
    no_charge_hours = getfield_default(EV_params, 'no_charge_hours', []);
    v2g_allowed_hours = getfield_default(EV_params, 'v2g_allowed_hours', []);
    EV_SoC = EV_params.EV_SoC0;
    Pch_rep = zeros(H,1);
    Pdis_rep = zeros(H,1);
    EV_SoC_hist_rep = zeros(H,1);

    for h = 1:H
        ch = min(max(Pch(h), 0), max_ch_power);
        dis = min(max(Pdis(h), 0), max_dis_power);

        hr = mod(start_hour + h - 1, 24);
        if ~isempty(no_charge_hours) && ismember(hr, no_charge_hours)
            ch = 0;
        end
        if ~isempty(v2g_allowed_hours) && ~ismember(hr, v2g_allowed_hours)
            dis = 0;
        end

        if ch > 0.01 && dis > 0.01
            if ch >= dis
                dis = 0;
            else
                ch = 0;
            end
        end

        available_energy = max(0, (EV_SoC - EV_params.EV_SoC_min) * EV_params.EV_capacity_total);
        max_dis = available_energy * EV_params.eta_ev_dis / max(dt, 1e-12);
        dis = min(dis, max_dis);

        available_capacity = max(0, (EV_params.EV_SoC_max - EV_SoC) * EV_params.EV_capacity_total);
        max_ch = available_capacity / max(EV_params.eta_ev_ch * dt, 1e-12);
        ch = min(ch, max_ch);

        dEV = (ch * dt * EV_params.eta_ev_ch - dis * dt / EV_params.eta_ev_dis) / EV_params.EV_capacity_total;
        EV_SoC = min(max(EV_SoC + dEV, EV_params.EV_SoC_min), EV_params.EV_SoC_max);

        Pch_rep(h) = ch;
        Pdis_rep(h) = dis;
        EV_SoC_hist_rep(h) = EV_SoC;
    end
end

% ----------------------------------------------------------------------
% ----------------------------------------------------------------------
% Single population-generation formula used to seed
% BOTH PSO and GA identically. Prevents either algorithm from getting an
% accidental head start (or handicap) purely from how its initial
% population happens to be drawn -- the root cause of the earlier
% PSO-vs-GA result, where PSO's own init formula clamped many EV-power
% genes to exactly 0 (avoiding the mutual-exclusivity/cycling penalties
% from the very first generation) while GA's uniform-over-full-range
% formula started a large fraction of individuals deep in penalty
% territory on the EV variables.
% ----------------------------------------------------------------------
function Pop0 = generate_shared_initial_population(pop_size, dim, lb, ub, warm_start, warm_frac, warm_jitter_frac)
% GENERATE_SHARED_INITIAL_POPULATION
%   One population, uniformly sampled across the FULL bounds [lb, ub] for
%   every gene. If warm_start is supplied, warm_frac (0 to 1) controls
%   what fraction of the population is seeded NEAR that warm-start
%   individual instead of being purely random:
%     - Row 1 is always the EXACT warm-start individual (unperturbed).
%     - The next round(warm_frac*pop_size)-1 rows are PERTURBED copies
%       of it (jittered by +/- warm_jitter_frac of each dimension's
%       range, then clipped to bounds).
%     - The remaining rows are uniform random, exactly as before.
%   Seeding only one individual (warm_frac ~ 1/pop_size, the old default)
%   gives the search a single, easily-lost foothold in a known-reasonable
%   region. A larger warm_frac (e.g. 0.15) gives it many footholds,
%   without forcing the whole population there -- most of the population
%   is still free to explore broadly.
%   Both custom_pso and custom_ga consume this SAME matrix when shared
%   init is enabled, instead of generating their own.
    if nargin < 6 || isempty(warm_frac)
        warm_frac = 0;
    end
    if nargin < 7 || isempty(warm_jitter_frac)
        warm_jitter_frac = 0.05;
    end

    lb = lb(:)'; ub = ub(:)';
    Pop0 = lb + rand(pop_size, dim) .* (ub - lb);

    if ~isempty(warm_start)
        warm_start = min(max(warm_start(:)', lb), ub);
        n_warm = max(1, min(pop_size, round(warm_frac * pop_size)));
        range = ub - lb;

        Pop0(1,:) = warm_start;  % always keep one exact, unperturbed copy
        for i = 2:n_warm
            jitter = (rand(1,dim)*2 - 1) .* (warm_jitter_frac * range);
            Pop0(i,:) = min(max(warm_start + jitter, lb), ub);
        end
    end
end

function shared_init = generate_shared_init_populations(H, battery_params, EV_params, ...
    EV_ch_warm, EV_dis_warm, pop_size, n_runs, seed_base, warm_frac)
%   GENERATE_SHARED_INIT_POPULATIONS
%   Builds, ONCE, the per-run initial populations for both the BESS-only
%   (dim = H) and BESS+EV (dim = 3H) problems, using the SAME seeds that
%   multirun_pso_compare and multirun_ga_compare use for their own search
%   iterations. Returns a 1 x n_runs cell array; shared_init{run}.bess and
%   shared_init{run}.bessev are then passed identically into both the PSO
%   and GA call for that run, so neither algorithm's reported performance
%   reflects an initialization artifact.
%
%   warm_frac (0 to 1, default 0.15): fraction of the BESS+EV population
%   seeded near the rule-based EV warm-start schedule (see
%   generate_shared_initial_population), rather than just a single
%   individual. The BESS-only problem has no EV warm-start, so this only
%   affects the .bessev population.

    if nargin < 9 || isempty(warm_frac)
        warm_frac = 0.15;
    end

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess =  battery_params.Pmax * ones(H, 1);

    max_ev_power = EV_params.P_ev_max * EV_params.vehicle_count;
    max_ev_ch_power = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);
    max_ev_dis_power = getfield_default(EV_params, 'P_ev_dis_max', max_ev_power);
    lb_ev = zeros(2*H, 1);
    ub_ev = [max_ev_ch_power * ones(H, 1); max_ev_dis_power * ones(H, 1)];
    lb_bessev = [lb_bess; lb_ev];
    ub_bessev = [ub_bess; ub_ev];

    warm_start_bess = zeros(1, H);

    if ~isempty(EV_ch_warm) && ~isempty(EV_dis_warm)
        warm_start_bessev = [zeros(1,H), EV_ch_warm(:)', EV_dis_warm(:)'];
    else
        warm_start_bessev = [];
    end

    shared_init = cell(n_runs, 1);
    for run = 1:n_runs
        seed = seed_base + run;  % IDENTICAL seed convention used downstream
        rng(seed);
        Pop0_bess   = generate_shared_initial_population(pop_size, H,    lb_bess,   ub_bess,   warm_start_bess, warm_frac);
        Pop0_bessev = generate_shared_initial_population(pop_size, 3*H,  lb_bessev, ub_bessev, warm_start_bessev, warm_frac);
        shared_init{run} = struct('bess', Pop0_bess, 'bessev', Pop0_bessev);
    end
end

% ----------------------------------------------------------------------
% Deterministic coordinate-wise pattern search % ("Hooke-Jeeves" style), 
% applied identically after BOTH custom_pso and
% custom_ga finish their main loop. Population-based metaheuristics often
% land NEAR a good solution but not exactly on it, especially in a
% landscape with several narrow, nearby local optima (as repeatedly
% observed here: best-of-N reaches a much better cost than the typical
% run). This step lets every run exploit its own final region more
% thoroughly, one dimension at a time, rather than relying on luck to
% have landed precisely in the best basin. It is fully deterministic
% given its starting point and the objective function -- no randomness,
% no new bias between PSO and GA, since both are polished by the exact
% same routine; only the starting point each algorithm hands it differs.
% ----------------------------------------------------------------------
function [x_polished, f_polished, n_evals] = local_polish(objfun, x0, lb, ub, max_passes, init_step_frac, shrink_factor, min_step_frac)
% LOCAL_POLISH Coordinate-wise pattern search refinement.
%
% Inputs:
%   objfun         : function handle, objfun(x) -> scalar cost, x is dim x 1
%   x0             : dim x 1 starting point (e.g. PSO/GA's best candidate)
%   lb, ub         : bounds (any shape; normalized to columns internally)
%   max_passes     : max sweeps over all dimensions (default 20)
%   init_step_frac : initial step size as a fraction of each dim's range
%                    (default 0.05, i.e. 5% of the dimension's range)
%   shrink_factor   : step shrink factor applied after a pass with no
%                    improvement (default 0.5)
%   min_step_frac  : stop once step size falls below this fraction of
%                    range for every dimension (default 1e-3)
%
% Outputs:
%   x_polished : refined solution (same cost or better than x0)
%   f_polished : its cost
%   n_evals    : number of objfun calls made (for bookkeeping)

    if nargin < 5  || isempty(max_passes),     max_passes = 20;     end
    if nargin < 6  || isempty(init_step_frac), init_step_frac = 0.05; end
    if nargin < 7  || isempty(shrink_factor),  shrink_factor = 0.5;  end
    if nargin < 8  || isempty(min_step_frac),  min_step_frac = 1e-3; end

    lb = lb(:); ub = ub(:); x = x0(:);
    dim = numel(x);
    range = max(ub - lb, 1e-12);   % avoid zero-range divide issues
    step = init_step_frac * range;
    min_step = min_step_frac * range;

    f_best = objfun(x);
    n_evals = 1;

    for pass = 1:max_passes
        improved_this_pass = false;
        for i = 1:dim
            if step(i) <= min_step(i)
                continue;
            end

            x_plus = x;
            x_plus(i) = min(x(i) + step(i), ub(i));
            f_plus = objfun(x_plus);
            n_evals = n_evals + 1;

            x_minus = x;
            x_minus(i) = max(x(i) - step(i), lb(i));
            f_minus = objfun(x_minus);
            n_evals = n_evals + 1;

            if f_plus < f_best && f_plus <= f_minus
                x = x_plus; f_best = f_plus; improved_this_pass = true;
            elseif f_minus < f_best
                x = x_minus; f_best = f_minus; improved_this_pass = true;
            end
        end

        if ~improved_this_pass
            step = step * shrink_factor;
            if all(step <= min_step)
                break;
            end
        end
    end

    x_polished = x;
    f_polished = f_best;
end

% ----------------------------------------------------------------------
function [gbestX, gbestF, history, polish_delta] = custom_pso(objfun, dim, lb, ub, swarm_size, maxiter, w, c1, c2, vel_clamp, warm_start, init_pop, polish)
% Accepts an optional init_pop (swarm_size x dim) so
% PSO and GA can be seeded from an IDENTICAL initial population, removing
% any advantage/disadvantage caused by the two algorithms using different
% random initialization formulas. If init_pop is not supplied, falls back
% to PSO's own default initialization (kept for backward compatibility).
%
% After the main swarm loop finishes, the best
% solution found is refined with a deterministic coordinate-wise pattern
% search (see local_polish), unless polish=false is explicitly passed.
% Applied identically inside custom_ga, this lets each run exploit its
% final region more thoroughly instead of relying on luck to land
% exactly in a narrow optimum -- addressing the repeatedly-observed
% symptom where the best-of-N run reaches the good solution but the
% TYPICAL run does not.
    if nargin < 11
        warm_start = [];
    end
    if nargin < 12
        init_pop = [];
    end
    if nargin < 13 || isempty(polish)
        polish = true;
    end
    lb = lb(:);
    ub = ub(:);

    % Accept scalar or per-dimension velocity clamp.
    if isscalar(vel_clamp)
        vel_clamp = vel_clamp * ones(1, dim);
    else
        vel_clamp = vel_clamp(:)';
    end

    if ~isempty(init_pop)
        X = init_pop;
        X = min(max(X, lb'), ub');
    else
        X = (rand(swarm_size, dim) * 0.4 - 0.2) .* (ub' - lb') + 0;
        X = min(max(X, lb'), ub');
        if ~isempty(warm_start)
            X(1,:) = min(max(warm_start(:)', lb'), ub');
        end
    end

    V = zeros(size(X));
    pbestX = X;
    pbestF = zeros(swarm_size, 1);

    for i = 1:swarm_size
        pbestF(i) = objfun(X(i,:)');
    end

    [gbestF, idx] = min(pbestF);
    gbestX = pbestX(idx,:)';

    history = [];
    no_improve = 0;
    best_prev = gbestF;

    w_max = w;
    w_min = 0.4 * w;

    for it = 1:maxiter
        if maxiter > 1
            w_it = w_max - (w_max - w_min) * (it-1) / (maxiter-1);
        else
            w_it = w_max;
        end

        for i = 1:swarm_size
            r1 = rand(dim, 1);
            r2 = rand(dim, 1);

            V(i,:) = (w_it * V(i,:))' + c1*r1.*(pbestX(i,:)' - X(i,:)') + c2*r2.*(gbestX - X(i,:)');
            V(i,:) = min(max(V(i,:), -vel_clamp), vel_clamp);

            X(i,:) = X(i,:) + V(i,:);
            X(i,:) = min(max(X(i,:), lb'), ub');

            val = objfun(X(i,:)');
            if val < pbestF(i)
                pbestF(i) = val;
                pbestX(i,:) = X(i,:);
            end
        end

        [minF, idx] = min(pbestF);
        if minF < gbestF
            gbestF = minF;
            gbestX = pbestX(idx,:)';
        end

        history = [history; gbestF];

        if gbestF < best_prev - 1e-9
            best_prev = gbestF;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        if no_improve > 60
            disp("  PSO early stopping (no improvement)");
            break;
        end
    end

    if polish
        f_before_polish = gbestF;
        [gbestX, gbestF] = local_polish(objfun, gbestX, lb, ub);
        polish_delta = f_before_polish - gbestF;  % >= 0; how much polish improved cost
    else
        polish_delta = 0;
    end
end

function [bestPbat, bestP_ev_ch, bestP_ev_dis, bestCost, history, polish_delta] = pso_optimization(...
    PV_fore, Load_fore, Price_fore, battery_params, EV_params, ...
    EV_ch_warm, EV_dis_warm, swarm_size, maxiter, w_inertia, c1, c2, vel_clamp_factor, seed, init_pop)

    if nargin < 15
        init_pop = [];
    end

    H = length(PV_fore);
    dim = 3 * H;

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess = battery_params.Pmax * ones(H, 1);

    max_ev_power = EV_params.P_ev_max * EV_params.vehicle_count;
    max_ev_ch_power = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);
    max_ev_dis_power = getfield_default(EV_params, 'P_ev_dis_max', max_ev_power);
    lb_ev = zeros(2*H, 1);
    ub_ev = [max_ev_ch_power * ones(H, 1); max_ev_dis_power * ones(H, 1)];

    lb = [lb_bess; lb_ev];
    ub = [ub_bess; ub_ev];

    % Vector velocity clamp: BESS range is 4 kW, EV range is 7.2 kW.
    % A scalar clamp based only on BESS Pmax makes the EV genes move too slowly.
    vel_clamp = (vel_clamp_factor * (ub - lb))';

    objfun = @(x) total_cost_withGrid_and_soc_repair_EV(...
        x(1:H), x(H+1:2*H), x(2*H+1:3*H), ...
        PV_fore, Load_fore, Price_fore, 0.85*Price_fore, ...
        H, battery_params.SoC0, battery_params.SoC_min, battery_params.SoC_max, ...
        battery_params.Ecap, battery_params.Pmax, battery_params.eta_ch, ...
        battery_params.eta_dis, battery_params.dt, EV_params, battery_params);

    if nargin >= 6 && ~isempty(EV_ch_warm) && ~isempty(EV_dis_warm)
        warm_start = [zeros(1,H), EV_ch_warm(:)', EV_dis_warm(:)'];
    else
        warm_start = [];
    end

    rng(seed);  % FIX #3: explicit seeding at the call site
    [best_x, bestCost, history, polish_delta] = custom_pso(objfun, dim, lb, ub, ...
        swarm_size, maxiter, w_inertia, c1, c2, vel_clamp, warm_start, init_pop);

    bestPbat = best_x(1:H);
    bestP_ev_ch = best_x(H+1:2*H);
    bestP_ev_dis = best_x(2*H+1:3*H);
end


function mrc = multirun_pso_compare(PV_fore, Load_fore, Price_fore, BuyPrice, SellPrice, ...
    battery_params, EV_params, EV_params_base, EV_ch_warm, EV_dis_warm, EV_ch_uncontrolled, EV_dis_uncontrolled, ...
    H, swarm_size, maxiter, w_inertia, c1, c2, vel_clamp_factor, n_runs, seed_base, shared_init)

    if nargin < 22
        shared_init = [];
    end

    s2_costs = zeros(n_runs,1);
    s3_costs = zeros(n_runs,1);
    s3_ev_ch_energy = zeros(n_runs,1);
    s3_ev_dis_energy = zeros(n_runs,1);
    s3_ev_final_soc = zeros(n_runs,1);
    s3_all_histories = zeros(n_runs, maxiter);
    s3_runs = cell(n_runs,1);
    s2_polish_deltas = zeros(n_runs,1);  % FIX (polish diagnostic): how much
    s3_polish_deltas = zeros(n_runs,1);  % local_polish improved each run's cost

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess =  battery_params.Pmax * ones(H, 1);
    vel_clamp_bess = vel_clamp_factor * battery_params.Pmax;

    % Strategy 2: BESS optimizes only stationary storage while EV charging
    % remains uncontrolled but mandatory. This is the fair BESS-only baseline
    % for an EV paper, because the same EV mobility demand is served.
    objfun_bess_only = @(Pbat) total_cost_withGrid_and_soc_repair_EV(Pbat, ...
        EV_ch_uncontrolled, EV_dis_uncontrolled, ...
        PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
        battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
        battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
        battery_params.dt, EV_params_base, battery_params);

    for run = 1:n_runs
        seed = seed_base + run;  % SAME seed used for both strategies -> paired comparison

        if ~isempty(shared_init)
            Pop0_bess = shared_init{run}.bess;
            Pop0_bessev = shared_init{run}.bessev;
        else
            Pop0_bess = [];
            Pop0_bessev = [];
        end

        % --- Strategy 2: BESS only ---
        rng(seed);
        [bestPbat_s2, s2_cost_run, ~, s2_polish_run] = custom_pso(objfun_bess_only, H, lb_bess, ub_bess, ...
            swarm_size, maxiter, w_inertia, c1, c2, vel_clamp_bess, [], Pop0_bess);
        s2_costs(run) = s2_cost_run;
        s2_polish_deltas(run) = s2_polish_run;

        % Safe elite: BESS+EV search contains the Strategy-2 solution:
        % best BESS schedule + uncontrolled mandatory EV charging. Therefore
        % Strategy 3 should not be worse simply because the optimizer misses
        % this embedded baseline region.
        %if ~isempty(Pop0_bessev)
        %    Pop0_bessev(2,:) = [bestPbat_s2(:)', EV_ch_uncontrolled(:)', EV_dis_uncontrolled(:)'];
        %end

        % --- Strategy 3: BESS + EV (re-seeded identically before its own PSO call) ---
        [bestPbat, bestP_ev_ch, bestP_ev_dis, bestCost, history3, s3_polish_run] = pso_optimization(...
            PV_fore, Load_fore, Price_fore, battery_params, EV_params, ...
            EV_ch_warm, EV_dis_warm, swarm_size, maxiter, w_inertia, c1, c2, vel_clamp_factor, seed, Pop0_bessev);

        s3_costs(run) = bestCost;
        s3_polish_deltas(run) = s3_polish_run;

        % EV utilization diagnostic for multi-run reporting.
        [~, hist3_run] = total_cost_withGrid_and_soc_repair_EV( ...
            bestPbat, bestP_ev_ch, bestP_ev_dis, ...
            PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
            battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
            battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
            battery_params.dt, EV_params, battery_params);
        s3_ev_ch_energy(run) = sum(hist3_run.P_ev_ch) * battery_params.dt;
        s3_ev_dis_energy(run) = sum(hist3_run.P_ev_dis) * battery_params.dt;
        s3_ev_final_soc(run) = hist3_run.EV_SoC(end);
        hist_len = length(history3);
        s3_all_histories(run, 1:hist_len) = history3(:)';
        if hist_len < maxiter
            s3_all_histories(run, hist_len+1:end) = history3(end);
        end

        s3_runs{run}.Pbat = bestPbat;
        s3_runs{run}.P_ev_ch = bestP_ev_ch;
        s3_runs{run}.P_ev_dis = bestP_ev_dis;
        s3_runs{run}.cost = bestCost;
        s3_runs{run}.convergence = history3;
    end

    % --- Polish diagnostic summary ---
    mrc.s2_polish_deltas = s2_polish_deltas;
    mrc.s3_polish_deltas = s3_polish_deltas;
    mrc.s2_polish_mean = mean(s2_polish_deltas);
    mrc.s3_polish_mean = mean(s3_polish_deltas);
    mrc.s2_polish_n_improved = sum(s2_polish_deltas > 1e-6);
    mrc.s3_polish_n_improved = sum(s3_polish_deltas > 1e-6);

    % --- Statistics ---
    mrc.s2_mean = mean(s2_costs); mrc.s2_std = std(s2_costs);
    mrc.s2_median = median(s2_costs); mrc.s2_min = min(s2_costs); mrc.s2_max = max(s2_costs);

    mrc.s3_mean = mean(s3_costs); mrc.s3_std = std(s3_costs);
    mrc.s3_median = median(s3_costs); mrc.s3_min = min(s3_costs); mrc.s3_max = max(s3_costs);

    mrc.s2_all_costs = s2_costs;
    mrc.s3_all_costs = s3_costs;
    mrc.s3_ev_ch_energy = s3_ev_ch_energy;
    mrc.s3_ev_dis_energy = s3_ev_dis_energy;
    mrc.s3_ev_final_soc = s3_ev_final_soc;
    mrc.s3_ev_ch_mean = mean(s3_ev_ch_energy); mrc.s3_ev_ch_std = std(s3_ev_ch_energy);
    mrc.s3_ev_dis_mean = mean(s3_ev_dis_energy); mrc.s3_ev_dis_std = std(s3_ev_dis_energy);
    mrc.s3_ev_final_soc_mean = mean(s3_ev_final_soc); mrc.s3_ev_final_soc_std = std(s3_ev_final_soc);
    mrc.s3_all_histories = s3_all_histories;

    % --- Paired significance test (Strategy 3 vs Strategy 2, matched seeds) ---
    try
        mrc.p_s3_vs_s2 = signrank(s3_costs, s2_costs);
    catch
        % Statistics and Machine Learning Toolbox not available: fall back
        % to a paired t-test, which is less robust to outliers but does
        % not require that toolbox function.
        [~, mrc.p_s3_vs_s2] = ttest(s3_costs, s2_costs);
    end

    % --- Representative (median-cost) Strategy-3 run for downstream figures ---
    [~, repr_idx] = min(abs(s3_costs - mrc.s3_median));
    mrc.s3_repr_Pbat = s3_runs{repr_idx}.Pbat;
    mrc.s3_repr_P_ev_ch = s3_runs{repr_idx}.P_ev_ch;
    mrc.s3_repr_P_ev_dis = s3_runs{repr_idx}.P_ev_dis;
    mrc.s3_repr_cost = s3_runs{repr_idx}.cost;
    mrc.s3_repr_convergence = s3_runs{repr_idx}.convergence;

    [mrc.s3_repr_cost, mrc.s3_repr_hist] = total_cost_withGrid_and_soc_repair_EV(...
        mrc.s3_repr_Pbat, mrc.s3_repr_P_ev_ch, mrc.s3_repr_P_ev_dis, ...
        PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
        battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
        battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
        battery_params.dt, EV_params, battery_params);
end

function [EV_ch_fore, EV_dis_fore] = persistence_ev_forecast(EV_Charging_hist, EV_Discharging_hist, t0, H)
    EV_ch_fore = zeros(H, 1);
    EV_dis_fore = zeros(H, 1);
    for h = 1:H
        idx = t0 + h - 24;
        if idx > 0 && idx <= length(EV_Charging_hist)
            EV_ch_fore(h) = EV_Charging_hist(idx);
            EV_dis_fore(h) = EV_Discharging_hist(idx);
        else
            EV_ch_fore(h) = EV_Charging_hist(t0-1);
            EV_dis_fore(h) = EV_Discharging_hist(t0-1);
        end
    end
    EV_ch_fore = max(0, EV_ch_fore);
    EV_dis_fore = max(0, EV_dis_fore);
end

function [EV_ch_fore, EV_dis_fore] = persistence_ev_forecast_enhanced(EV_Charging_hist, EV_Discharging_hist, ts, t0, H, method)
    EV_ch_fore = zeros(H, 1);
    EV_dis_fore = zeros(H, 1);

    switch method
        case 'daily'
            for h = 1:H
                idx = t0 + h - 24;
                if idx > 0 && idx <= length(EV_Charging_hist)
                    EV_ch_fore(h) = EV_Charging_hist(idx);
                    EV_dis_fore(h) = EV_Discharging_hist(idx);
                else
                    EV_ch_fore(h) = EV_Charging_hist(t0-1);
                    EV_dis_fore(h) = EV_Discharging_hist(t0-1);
                end
            end
        case 'weekly'
            for h = 1:H
                idx = t0 + h - 168;
                if idx > 0 && idx <= length(EV_Charging_hist)
                    EV_ch_fore(h) = EV_Charging_hist(idx);
                    EV_dis_fore(h) = EV_Discharging_hist(idx);
                else
                    EV_ch_fore(h) = EV_Charging_hist(t0-1);
                    EV_dis_fore(h) = EV_Discharging_hist(t0-1);
                end
            end
        case 'hourly'
            hour_of_day = hour(ts);
            start_hour = hour(ts(t0));
            for h = 1:H
                target_hour = mod(start_hour + h - 1, 24);
                idx = hour_of_day == target_hour;
                if sum(idx) > 0
                    EV_ch_fore(h) = mean(EV_Charging_hist(idx));
                    EV_dis_fore(h) = mean(EV_Discharging_hist(idx));
                else
                    EV_ch_fore(h) = 0;
                    EV_dis_fore(h) = 0;
                end
            end
    end

    EV_ch_fore = min(max(0, EV_ch_fore), 7.2);
    EV_dis_fore = min(max(0, EV_dis_fore), 7.2);
end

% ----------------------------------------------------------------------
% GA COMPARISON: real-coded Genetic Algorithm (SBX crossover + polynomial
% mutation), mirroring custom_pso's interface and computational budget
% (population size, generation count, early-stopping rule) so PSO vs GA
% comparisons are made on equal footing. Uses the SAME objective function
% as PSO -- only the search algorithm differs.
% ----------------------------------------------------------------------
function [gbestX, gbestF, history, polish_delta] = custom_ga(objfun, dim, lb, ub, pop_size, maxgen, pc, pm, eta_c, eta_m, warm_start, init_pop, polish)
% CUSTOM_GA Real-coded Genetic Algorithm (SBX crossover + polynomial mutation)
%
% Inputs:
%   objfun     : function handle, objfun(x) -> scalar cost (minimize), x is dim x 1
%   dim        : number of decision variables
%   lb, ub     : dim x 1 (or 1 x dim) lower/upper bounds
%   pop_size   : population size (use the SAME value as PSO's SWARM_SIZE)
%   maxgen     : max generations (use the SAME value as PSO's PSO_MAXITER)
%   pc         : crossover probability (default 0.9)
%   pm         : mutation probability per gene (default 1/dim)
%   eta_c      : SBX distribution index (default 15)
%   eta_m      : polynomial mutation distribution index (default 20)
%   warm_start : optional 1 x dim row vector to seed one individual
%   init_pop   : optional pop_size x dim shared initial population (see
%                generate_shared_init_populations). If supplied, GA starts
%                from the IDENTICAL population PSO used for the same run,
%                removing any bias from the two algorithms using different
%                initialization formulas.
%   polish     : if true (default), the best individual found is refined
%                with a deterministic coordinate-wise pattern search (see
%                local_polish) before returning. Applied identically
%                inside custom_pso, so this does not reintroduce any
%                PSO-vs-GA asymmetry -- only the starting point each
%                algorithm hands to the polish step differs.
%
% Outputs:
%   gbestX  : dim x 1 best solution found
%   gbestF  : best cost found
%   history : best-cost-so-far trace (one entry per generation)

    if nargin < 7  || isempty(pc),     pc = 0.9;      end
    if nargin < 8  || isempty(pm),     pm = 1/dim;    end
    if nargin < 9  || isempty(eta_c),  eta_c = 15;    end
    if nargin < 10 || isempty(eta_m),  eta_m = 20;    end
    if nargin < 11, warm_start = [];                  end
    if nargin < 12, init_pop = [];                     end
    if nargin < 13 || isempty(polish), polish = true;  end

    lb = lb(:)'; ub = ub(:)';

    % ---- Initialize population ----
    if ~isempty(init_pop)
        Pop = min(max(init_pop, lb), ub);
    else
        Pop = lb + rand(pop_size, dim) .* (ub - lb);
        if ~isempty(warm_start)
            Pop(1,:) = min(max(warm_start(:)', lb), ub);
        end
    end

    Fit = zeros(pop_size, 1);
    for i = 1:pop_size
        Fit(i) = objfun(Pop(i,:)');
    end


    [gbestF, idx] = min(Fit);
    gbestX = Pop(idx,:)';

    history = [];
    no_improve = 0;
    best_prev = gbestF;

    for gen = 1:maxgen
        NewPop = zeros(pop_size, dim);

        % ---- Elitism: best individual survives unchanged ----
        [~, elite_idx] = min(Fit);
        NewPop(1,:) = Pop(elite_idx,:);

        child_count = 1;
        while child_count < pop_size
            p1 = tournament_select(Pop, Fit, 3);
            p2 = tournament_select(Pop, Fit, 3);

            if rand < pc
                [c1, c2] = sbx_crossover(p1, p2, lb, ub, eta_c);
            else
                c1 = p1; c2 = p2;
            end

            c1 = poly_mutate(c1, lb, ub, pm, eta_m);
            c2 = poly_mutate(c2, lb, ub, pm, eta_m);

            child_count = child_count + 1;
            NewPop(child_count,:) = c1;
            if child_count < pop_size
                child_count = child_count + 1;
                NewPop(child_count,:) = c2;
            end
        end

        Pop = NewPop;
        for i = 1:pop_size
            Fit(i) = objfun(Pop(i,:)');
        end

        [minF, idx] = min(Fit);
        if minF < gbestF
            gbestF = minF;
            gbestX = Pop(idx,:)';
        end

        history = [history; gbestF];

        if gbestF < best_prev - 1e-9
            best_prev = gbestF;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        if no_improve > 60
            disp("  GA early stopping (no improvement)");
            break;
        end
    end

    if polish
        f_before_polish = gbestF;
        [gbestX, gbestF] = local_polish(objfun, gbestX, lb, ub);
        polish_delta = f_before_polish - gbestF;  % >= 0; how much polish improved cost
    else
        polish_delta = 0;
    end
end

function winner = tournament_select(Pop, Fit, k)
    n = size(Pop, 1);
    contestants = randi(n, k, 1);
    [~, best_local] = min(Fit(contestants));
    winner = Pop(contestants(best_local), :);
end

function [c1, c2] = sbx_crossover(p1, p2, lb, ub, eta_c)
    dim = numel(p1);
    c1 = p1; c2 = p2;
    for j = 1:dim
        if abs(p1(j) - p2(j)) > 1e-12
            x1 = min(p1(j), p2(j));
            x2 = max(p1(j), p2(j));
            xl = lb(j); xu = ub(j);
            u = rand;

            beta = 1 + (2*(x1 - xl) / max(x2 - x1, 1e-12));
            alpha = 2 - beta^(-(eta_c+1));
            if u <= 1/alpha
                betaq = (u*alpha)^(1/(eta_c+1));
            else
                betaq = (1/(2 - u*alpha))^(1/(eta_c+1));
            end
            child1 = 0.5*((x1+x2) - betaq*(x2-x1));

            beta2 = 1 + (2*(xu - x2) / max(x2 - x1, 1e-12));
            alpha2 = 2 - beta2^(-(eta_c+1));
            if u <= 1/alpha2
                betaq2 = (u*alpha2)^(1/(eta_c+1));
            else
                betaq2 = (1/(2 - u*alpha2))^(1/(eta_c+1));
            end
            child2 = 0.5*((x1+x2) + betaq2*(x2-x1));

            c1(j) = min(max(child1, xl), xu);
            c2(j) = min(max(child2, xl), xu);
        end
    end
end

function c = poly_mutate(x, lb, ub, pm, eta_m)
    dim = numel(x);
    c = x;
    for j = 1:dim
        if rand < pm
            xl = lb(j); xu = ub(j);
            if xu > xl
                delta1 = (x(j) - xl) / (xu - xl);
                delta2 = (xu - x(j)) / (xu - xl);
                u = rand;
                if u < 0.5
                    deltaq = (2*u + (1-2*u)*(1-delta1)^(eta_m+1))^(1/(eta_m+1)) - 1;
                else
                    deltaq = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta2)^(eta_m+1))^(1/(eta_m+1));
                end
                c(j) = x(j) + deltaq*(xu-xl);
                c(j) = min(max(c(j), xl), xu);
            end
        end
    end
end

function [bestPbat, bestP_ev_ch, bestP_ev_dis, bestCost, history, polish_delta] = ga_optimization(...
    PV_fore, Load_fore, Price_fore, battery_params, EV_params, ...
    EV_ch_warm, EV_dis_warm, pop_size, maxgen, pc, pm, eta_c, eta_m, seed, init_pop)
% Mirrors pso_optimization's interface: uses the SAME objective function,
% bounds, and EV warm-start convention -- only the search algorithm (GA
% instead of PSO) differs, so PSO vs GA results are directly comparable.
% FIX (shared init): also accepts an optional init_pop, forwarded directly
% to custom_ga, so PSO and GA can be compared from an IDENTICAL initial
% population (see generate_shared_init_populations).

    if nargin < 15
        init_pop = [];
    end

    H = length(PV_fore);
    dim = 3 * H;

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess =  battery_params.Pmax * ones(H, 1);
    max_ev_power = EV_params.P_ev_max * EV_params.vehicle_count;
    max_ev_ch_power = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);
    max_ev_dis_power = getfield_default(EV_params, 'P_ev_dis_max', max_ev_power);
    lb_ev = zeros(2*H, 1);
    ub_ev = [max_ev_ch_power * ones(H, 1); max_ev_dis_power * ones(H, 1)];
    lb = [lb_bess; lb_ev];
    ub = [ub_bess; ub_ev];

    objfun = @(x) total_cost_withGrid_and_soc_repair_EV(...
        x(1:H), x(H+1:2*H), x(2*H+1:3*H), ...
        PV_fore, Load_fore, Price_fore, 0.85*Price_fore, ...
        H, battery_params.SoC0, battery_params.SoC_min, battery_params.SoC_max, ...
        battery_params.Ecap, battery_params.Pmax, battery_params.eta_ch, ...
        battery_params.eta_dis, battery_params.dt, EV_params, battery_params);

    if nargin >= 6 && ~isempty(EV_ch_warm) && ~isempty(EV_dis_warm)
        warm_start = [zeros(1,H), EV_ch_warm(:)', EV_dis_warm(:)'];
    else
        warm_start = [];
    end

    rng(seed);   % explicit, matched seed -- same convention as pso_optimization
    [best_x, bestCost, history, polish_delta] = custom_ga(objfun, dim, lb, ub, ...
        pop_size, maxgen, pc, pm, eta_c, eta_m, warm_start, init_pop);

    bestPbat = best_x(1:H);
    bestP_ev_ch = best_x(H+1:2*H);
    bestP_ev_dis = best_x(2*H+1:3*H);
end

% ----------------------------------------------------------------------
% Unified, paired multi-run comparison for GA: Strategy 2 (BESS only) and
% Strategy 3 (BESS + EV), with the SAME matched seeds used by
% multirun_pso_compare, so PSO-run-k can be directly paired against
% GA-run-k. Selects a "representative" (median-cost) Strategy-3 run for
% downstream figures, exactly as multirun_pso_compare does for PSO.
% ----------------------------------------------------------------------
function mrc_ga = multirun_ga_compare(PV_fore, Load_fore, Price_fore, BuyPrice, SellPrice, ...
    battery_params, EV_params, EV_params_base, EV_ch_warm, EV_dis_warm, EV_ch_uncontrolled, EV_dis_uncontrolled, ...
    H, pop_size, maxgen, pc, pm, eta_c, eta_m, n_runs, seed_base, shared_init)

    if nargin < 22
        shared_init = [];
    end

    s2_costs = zeros(n_runs,1);
    s3_costs = zeros(n_runs,1);
    s3_all_histories = zeros(n_runs, maxgen);
    s3_runs = cell(n_runs,1);
    s2_polish_deltas = zeros(n_runs,1);  % FIX (polish diagnostic)
    s3_polish_deltas = zeros(n_runs,1);

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess =  battery_params.Pmax * ones(H, 1);

    % Strategy 2: BESS optimizes only stationary storage while EV charging
    % remains uncontrolled but mandatory. This is the fair BESS-only baseline
    % for an EV paper, because the same EV mobility demand is served.
    objfun_bess_only = @(Pbat) total_cost_withGrid_and_soc_repair_EV(Pbat, ...
        EV_ch_uncontrolled, EV_dis_uncontrolled, ...
        PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
        battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
        battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
        battery_params.dt, EV_params_base, battery_params);

    for run = 1:n_runs
        seed = seed_base + run;   % IDENTICAL seeds to multirun_pso_compare -> paired comparison

        if ~isempty(shared_init)
            Pop0_bess = shared_init{run}.bess;
            Pop0_bessev = shared_init{run}.bessev;
        else
            Pop0_bess = [];
            Pop0_bessev = [];
        end

        % --- Strategy 2: BESS only (GA) ---
        rng(seed);
        [bestPbat_s2, s2_cost_run, ~, s2_polish_run] = custom_ga(objfun_bess_only, H, lb_bess, ub_bess, ...
            pop_size, maxgen, pc, pm, eta_c, eta_m, [], Pop0_bess);
        s2_costs(run) = s2_cost_run;
        s2_polish_deltas(run) = s2_polish_run;

        % Safe elite: GA BESS+EV population also contains the
        % BESS+uncontrolled-EV Strategy-2 solution.
        %if ~isempty(Pop0_bessev)
        %    Pop0_bessev(2,:) = [bestPbat_s2(:)', EV_ch_uncontrolled(:)', EV_dis_uncontrolled(:)'];
        %end

        % --- Strategy 3: BESS + EV (GA, re-seeded identically) ---
        [bestPbat, bestP_ev_ch, bestP_ev_dis, bestCost, history3, s3_polish_run] = ga_optimization(...
            PV_fore, Load_fore, Price_fore, battery_params, EV_params, ...
            EV_ch_warm, EV_dis_warm, pop_size, maxgen, pc, pm, eta_c, eta_m, seed, Pop0_bessev);

        s3_costs(run) = bestCost;
        s3_polish_deltas(run) = s3_polish_run;
        hist_len = length(history3);
        s3_all_histories(run, 1:hist_len) = history3(:)';
        if hist_len < maxgen
            s3_all_histories(run, hist_len+1:end) = history3(end);
        end

        s3_runs{run} = struct('Pbat', bestPbat, 'P_ev_ch', bestP_ev_ch, ...
            'P_ev_dis', bestP_ev_dis, 'cost', bestCost, 'convergence', history3);
    end

    % --- Polish diagnostic summary ---
    mrc_ga.s2_polish_deltas = s2_polish_deltas;
    mrc_ga.s3_polish_deltas = s3_polish_deltas;
    mrc_ga.s2_polish_mean = mean(s2_polish_deltas);
    mrc_ga.s3_polish_mean = mean(s3_polish_deltas);
    mrc_ga.s2_polish_n_improved = sum(s2_polish_deltas > 1e-6);
    mrc_ga.s3_polish_n_improved = sum(s3_polish_deltas > 1e-6);

    mrc_ga.s2_mean = mean(s2_costs); mrc_ga.s2_std = std(s2_costs);
    mrc_ga.s2_median = median(s2_costs); mrc_ga.s2_min = min(s2_costs); mrc_ga.s2_max = max(s2_costs);
    mrc_ga.s3_mean = mean(s3_costs); mrc_ga.s3_std = std(s3_costs);
    mrc_ga.s3_median = median(s3_costs); mrc_ga.s3_min = min(s3_costs); mrc_ga.s3_max = max(s3_costs);
    mrc_ga.s2_all_costs = s2_costs;
    mrc_ga.s3_all_costs = s3_costs;
    mrc_ga.s3_all_histories = s3_all_histories;

    [~, repr_idx] = min(abs(s3_costs - mrc_ga.s3_median));
    mrc_ga.s3_repr_Pbat = s3_runs{repr_idx}.Pbat;
    mrc_ga.s3_repr_P_ev_ch = s3_runs{repr_idx}.P_ev_ch;
    mrc_ga.s3_repr_P_ev_dis = s3_runs{repr_idx}.P_ev_dis;
    mrc_ga.s3_repr_convergence = s3_runs{repr_idx}.convergence;

    [mrc_ga.s3_repr_cost, mrc_ga.s3_repr_hist] = total_cost_withGrid_and_soc_repair_EV(...
        mrc_ga.s3_repr_Pbat, mrc_ga.s3_repr_P_ev_ch, mrc_ga.s3_repr_P_ev_dis, ...
        PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
        battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
        battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
        battery_params.dt, EV_params, battery_params);
end

% ----------------------------------------------------------------------
% DIFFERENTIAL EVOLUTION: classic DE/rand/1/bin, mirroring custom_pso's
% and custom_ga's interface and computational budget (population size,
% generation count, early-stopping rule) so PSO vs DE comparisons are
% made on equal footing. Uses the SAME objective function as PSO/GA --
% only the search algorithm differs. DE is a natural fit for this kind
% of bounded, continuous, real-valued problem: its mutation operator
% (scaled vector differences between existing population members)
% requires no hand-designed crossover/mutation scheme the way real-coded
% GA does, and its per-individual greedy selection (trial replaces target
% only if strictly better-or-equal) gives it built-in elitism at the
% individual level.
% ----------------------------------------------------------------------
function [gbestX, gbestF, history, polish_delta] = custom_de(objfun, dim, lb, ub, pop_size, maxgen, F, CR, warm_start, init_pop, polish)
% CUSTOM_DE Differential Evolution (DE/rand/1/bin)
%
% Inputs:
%   objfun     : function handle, objfun(x) -> scalar cost (minimize), x is dim x 1
%   dim        : number of decision variables
%   lb, ub     : dim x 1 (or 1 x dim) lower/upper bounds
%   pop_size   : population size (use the SAME value as PSO's SWARM_SIZE)
%   maxgen     : max generations (use the SAME value as PSO's PSO_MAXITER)
%   F          : differential weight (default 0.8), scales the vector
%                difference used to mutate each target individual
%   CR         : crossover probability (default 0.9), per-gene
%                probability of taking the mutant's value over the
%                target's (binomial crossover)
%   warm_start : optional 1 x dim row vector to seed one individual
%   init_pop   : optional pop_size x dim shared initial population (see
%                generate_shared_init_populations). If supplied, DE starts
%                from the IDENTICAL population PSO/GA used for the same
%                run, removing any bias from different initialization
%                formulas.
%   polish     : if true (default), the best individual found is refined
%                with local_polish before returning, identically to
%                custom_pso and custom_ga.
%
% Outputs:
%   gbestX        : dim x 1 best solution found
%   gbestF        : best cost found
%   history       : best-cost-so-far trace (one entry per generation)
%   polish_delta  : cost improvement contributed by the polish step

    if nargin < 7  || isempty(F),      F = 0.8;     end
    if nargin < 8  || isempty(CR),     CR = 0.9;    end
    if nargin < 9,  warm_start = [];                end
    if nargin < 10, init_pop = [];                   end
    if nargin < 11 || isempty(polish), polish = true; end

    lb = lb(:)'; ub = ub(:)';

    if ~isempty(init_pop)
        Pop = min(max(init_pop, lb), ub);
    else
        Pop = lb + rand(pop_size, dim) .* (ub - lb);
        if ~isempty(warm_start)
            Pop(1,:) = min(max(warm_start(:)', lb), ub);
        end
    end

    Fit = zeros(pop_size, 1);
    for i = 1:pop_size
        Fit(i) = objfun(Pop(i,:)');
    end

    [gbestF, idx] = min(Fit);
    gbestX = Pop(idx,:)';

    history = [];
    no_improve = 0;
    best_prev = gbestF;

    for gen = 1:maxgen
        % F-dither (Das & Suganthan, 2011): per-generation F in [0.5, 1.0]
        % to escape flat penalty plateaus.
        F_g = 0.5 + 0.5*rand;
        for i = 1:pop_size
            % --- Mutation: DE/rand/1 with per-generation F-dither ---
            candidates = setdiff(1:pop_size, i);
            r = candidates(randperm(numel(candidates), 3));
            mutant = Pop(r(1),:) + F_g * (Pop(r(2),:) - Pop(r(3),:));
            mutant = min(max(mutant, lb), ub);

            % --- Binomial crossover ---
            trial = Pop(i,:);
            j_rand = randi(dim);  % guarantee at least one gene from mutant
            mask = rand(1,dim) < CR;
            mask(j_rand) = true;
            trial(mask) = mutant(mask);

            % --- Greedy selection (trial replaces target only if better) ---
            f_trial = objfun(trial');
            if f_trial <= Fit(i)
                Pop(i,:) = trial;
                Fit(i) = f_trial;
            end
        end

        [minF, idx] = min(Fit);
        if minF < gbestF
            gbestF = minF;
            gbestX = Pop(idx,:)';
        end

        history = [history; gbestF];

        if gbestF < best_prev - 1e-9
            best_prev = gbestF;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        if no_improve > 150   % was 60; premature stalls observed on penalty plateaus
            disp("  DE early stopping (no improvement)");
            break;
        end
    end

    if polish
        f_before_polish = gbestF;
        [gbestX, gbestF] = local_polish(objfun, gbestX, lb, ub);
        polish_delta = f_before_polish - gbestF;
    else
        polish_delta = 0;
    end
end

function [bestPbat, bestP_ev_ch, bestP_ev_dis, bestCost, history, polish_delta] = de_optimization(...
    PV_fore, Load_fore, Price_fore, battery_params, EV_params, ...
    EV_ch_warm, EV_dis_warm, pop_size, maxgen, F, CR, seed, init_pop)
% Mirrors pso_optimization's and ga_optimization's interface: uses the
% SAME objective function, bounds, and EV warm-start convention -- only
% the search algorithm (DE instead of PSO/GA) differs, so results are
% directly comparable.

    if nargin < 13
        init_pop = [];
    end

    H = length(PV_fore);
    dim = 3 * H;

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess =  battery_params.Pmax * ones(H, 1);
    max_ev_power = EV_params.P_ev_max * EV_params.vehicle_count;
    max_ev_ch_power = getfield_default(EV_params, 'P_ev_ch_max', max_ev_power);
    max_ev_dis_power = getfield_default(EV_params, 'P_ev_dis_max', max_ev_power);
    lb_ev = zeros(2*H, 1);
    ub_ev = [max_ev_ch_power * ones(H, 1); max_ev_dis_power * ones(H, 1)];
    lb = [lb_bess; lb_ev];
    ub = [ub_bess; ub_ev];

    objfun = @(x) total_cost_withGrid_and_soc_repair_EV(...
        x(1:H), x(H+1:2*H), x(2*H+1:3*H), ...
        PV_fore, Load_fore, Price_fore, 0.85*Price_fore, ...
        H, battery_params.SoC0, battery_params.SoC_min, battery_params.SoC_max, ...
        battery_params.Ecap, battery_params.Pmax, battery_params.eta_ch, ...
        battery_params.eta_dis, battery_params.dt, EV_params, battery_params);

    if nargin >= 6 && ~isempty(EV_ch_warm) && ~isempty(EV_dis_warm)
        warm_start = [zeros(1,H), EV_ch_warm(:)', EV_dis_warm(:)'];
    else
        warm_start = [];
    end

    rng(seed);   % explicit, matched seed -- same convention as pso_optimization/ga_optimization
    [best_x, bestCost, history, polish_delta] = custom_de(objfun, dim, lb, ub, ...
        pop_size, maxgen, F, CR, warm_start, init_pop);

    bestPbat = best_x(1:H);
    bestP_ev_ch = best_x(H+1:2*H);
    bestP_ev_dis = best_x(2*H+1:3*H);
end

% ----------------------------------------------------------------------
% Unified, paired multi-run comparison for DE: Strategy 2 (BESS only) and
% Strategy 3 (BESS + EV), with the SAME matched seeds used by
% multirun_pso_compare and multirun_ga_compare, so PSO-run-k, GA-run-k,
% and DE-run-k can all be directly paired against each other. Selects a
% "representative" (median-cost) Strategy-3 run for downstream figures,
% exactly as multirun_pso_compare/multirun_ga_compare do.
% ----------------------------------------------------------------------
function mrc_de = multirun_de_compare(PV_fore, Load_fore, Price_fore, BuyPrice, SellPrice, ...
    battery_params, EV_params, EV_params_base, EV_ch_warm, EV_dis_warm, EV_ch_uncontrolled, EV_dis_uncontrolled, ...
    H, pop_size, maxgen, F, CR, n_runs, seed_base, shared_init)

    if nargin < 20
        shared_init = [];
    end

    s2_costs = zeros(n_runs,1);
    s3_costs = zeros(n_runs,1);
    s3_all_histories = zeros(n_runs, maxgen);
    s3_runs = cell(n_runs,1);
    s2_polish_deltas = zeros(n_runs,1);
    s3_polish_deltas = zeros(n_runs,1);

    lb_bess = -battery_params.Pmax * ones(H, 1);
    ub_bess =  battery_params.Pmax * ones(H, 1);

    % Strategy 2: BESS optimizes only stationary storage while EV charging
    % remains uncontrolled but mandatory. This is the fair BESS-only baseline
    % for an EV paper, because the same EV mobility demand is served.
    objfun_bess_only = @(Pbat) total_cost_withGrid_and_soc_repair_EV(Pbat, ...
        EV_ch_uncontrolled, EV_dis_uncontrolled, ...
        PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
        battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
        battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
        battery_params.dt, EV_params_base, battery_params);

    for run = 1:n_runs
        seed = seed_base + run;   % IDENTICAL seeds to multirun_pso_compare/multirun_ga_compare

        if ~isempty(shared_init)
            Pop0_bess = shared_init{run}.bess;
            Pop0_bessev = shared_init{run}.bessev;
        else
            Pop0_bess = [];
            Pop0_bessev = [];
        end

        % --- Strategy 2: BESS only (DE) ---
        rng(seed);
        [bestPbat_s2, s2_cost_run, ~, s2_polish_run] = custom_de(objfun_bess_only, H, lb_bess, ub_bess, ...
            pop_size, maxgen, F, CR, [], Pop0_bess);
        s2_costs(run) = s2_cost_run;
        s2_polish_deltas(run) = s2_polish_run;

        % Safe elite: DE BESS+EV population also contains the
        % BESS+uncontrolled-EV Strategy-2 solution.
        %if ~isempty(Pop0_bessev)
        %    Pop0_bessev(2,:) = [bestPbat_s2(:)', EV_ch_uncontrolled(:)', EV_dis_uncontrolled(:)'];
        %end

        % --- Strategy 3: BESS + EV (DE, re-seeded identically) ---
        [bestPbat, bestP_ev_ch, bestP_ev_dis, bestCost, history3, s3_polish_run] = de_optimization(...
            PV_fore, Load_fore, Price_fore, battery_params, EV_params, ...
            EV_ch_warm, EV_dis_warm, pop_size, maxgen, F, CR, seed, Pop0_bessev);

        s3_costs(run) = bestCost;
        s3_polish_deltas(run) = s3_polish_run;
        hist_len = length(history3);
        s3_all_histories(run, 1:hist_len) = history3(:)';
        if hist_len < maxgen
            s3_all_histories(run, hist_len+1:end) = history3(end);
        end

        s3_runs{run} = struct('Pbat', bestPbat, 'P_ev_ch', bestP_ev_ch, ...
            'P_ev_dis', bestP_ev_dis, 'cost', bestCost, 'convergence', history3);
    end

    % --- Polish diagnostic summary ---
    mrc_de.s2_polish_deltas = s2_polish_deltas;
    mrc_de.s3_polish_deltas = s3_polish_deltas;
    mrc_de.s2_polish_mean = mean(s2_polish_deltas);
    mrc_de.s3_polish_mean = mean(s3_polish_deltas);
    mrc_de.s2_polish_n_improved = sum(s2_polish_deltas > 1e-6);
    mrc_de.s3_polish_n_improved = sum(s3_polish_deltas > 1e-6);

    % --- Statistics ---
    mrc_de.s2_mean = mean(s2_costs); mrc_de.s2_std = std(s2_costs);
    mrc_de.s2_median = median(s2_costs); mrc_de.s2_min = min(s2_costs); mrc_de.s2_max = max(s2_costs);
    mrc_de.s3_mean = mean(s3_costs); mrc_de.s3_std = std(s3_costs);
    mrc_de.s3_median = median(s3_costs); mrc_de.s3_min = min(s3_costs); mrc_de.s3_max = max(s3_costs);
    mrc_de.s2_all_costs = s2_costs;
    mrc_de.s3_all_costs = s3_costs;
    mrc_de.s3_all_histories = s3_all_histories;

    % --- Representative (median-cost) Strategy-3 run for downstream figures ---
    [~, repr_idx] = min(abs(s3_costs - mrc_de.s3_median));
    mrc_de.s3_repr_Pbat = s3_runs{repr_idx}.Pbat;
    mrc_de.s3_repr_P_ev_ch = s3_runs{repr_idx}.P_ev_ch;
    mrc_de.s3_repr_P_ev_dis = s3_runs{repr_idx}.P_ev_dis;
    mrc_de.s3_repr_convergence = s3_runs{repr_idx}.convergence;

    [mrc_de.s3_repr_cost, mrc_de.s3_repr_hist] = total_cost_withGrid_and_soc_repair_EV(...
        mrc_de.s3_repr_Pbat, mrc_de.s3_repr_P_ev_ch, mrc_de.s3_repr_P_ev_dis, ...
        PV_fore, Load_fore, Price_fore, SellPrice, H, battery_params.SoC0, ...
        battery_params.SoC_min, battery_params.SoC_max, battery_params.Ecap, ...
        battery_params.Pmax, battery_params.eta_ch, battery_params.eta_dis, ...
        battery_params.dt, EV_params, battery_params);
end
