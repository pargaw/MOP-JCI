import numpy as np
import pandas as pd

# Load global variables
import sys
sys.path.append('../')
from config import *

#######################################################################################################################
# Synthetic data
#######################################################################################################################

def simulate_synth_covs (sample_no, corr_cov=False):
    # Create dictionary for each covariate index and mean
    X_means = {0: 66.0, # age
               1: 6.2, # wbc
               2: 0.8, # lymphocyte
               3: 183.0, # platelet
               4: 68.0, # serum_creat
               5: 31.0, # aspartete
               6: 16.0, # ALT
               7: 339.0, # lactate
               8: 76.0,  # creat_kinase
               9: 9.0 # time
              }
    
    # Create X
    X = np.round(np.random.normal(size=(sample_no, 1), loc=X_means[0], scale=4.1)) # age
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=X_means[1], scale=1.0) * 10.0) / 10.0])  # white blood cell count
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=X_means[2], scale=0.1) * 10.0) / 10.0])  # Lymphocyte count
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=X_means[3], scale=20.4))])  # Platelet count
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=X_means[4], scale=6.6))])  # Serum creatinine
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=X_means[5], scale=5.1))])  # Aspartete aminotransferase
    
    if corr_cov: 
        # Add correlation between time and ALT
        means = [X_means[6], X_means[9]]  
        stds = [2.56, 2.44]
        corr = 0.8
        covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
                [stds[0]*stds[1]*corr,           stds[1]**2]]  
    
        corr_alt, corr_time = np.random.multivariate_normal(means, covs, sample_no).T
        corr_time = corr_time.reshape(sample_no,1)
        corr_alt = corr_alt.reshape(sample_no,1)
        X = np.block([X,
                      np.round(corr_alt)])
    else:
        X = np.block([X, np.round(
            np.random.normal(size=(sample_no, 1), loc=X_means[6], scale=5.1))])  # Alanine aminotransferase
    
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=X_means[7], scale=51))])  # Lactate dehydrogenase
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=X_means[8], scale=21))])  # Creatine kinase

    # Time from study
    if corr_cov:
        X = np.block([X, corr_time])  
    else:
        X = np.block([X, np.random.uniform(size=(sample_no, 1)) * 11 + 4])

    return X, X_means
    

def get_train_test_data (num_outcomes, sample_no, X, W, Y_0, Y_1):
    # training
    Y_train = [None]*num_outcomes                         
    Y_0_train = [None]*num_outcomes
    Y_1_train = [None]*num_outcomes
    Y_cf_train = [None]*num_outcomes
    T_true_train = [None]*num_outcomes

    train_index = list(np.random.choice(range(sample_no), train_sample_no, replace=False))

    X_train = X[train_index]
    W_train = W[train_index]

    for i in range(num_outcomes):                          
        Y_0_train[i] = Y_0[i][train_index]
        Y_1_train[i] = Y_1[i][train_index]

        Y_train[i] = W_train * Y_1_train[i] + (1 - W_train) * Y_0_train[i]
        Y_cf_train[i] = W_train * Y_0_train[i] + (1 - W_train) * Y_1_train[i]
        T_true_train[i] = Y_1[i][train_index] - Y_0[i][train_index]

    # testing
    Y_test = [None]*num_outcomes
    Y_0_test = [None]*num_outcomes
    Y_1_test = [None]*num_outcomes
    T_true_test = [None]*num_outcomes
    Y_cf_test = [None]*num_outcomes

    test_index = list(set(list(range(sample_no))) - set(train_index))

    X_test = X[test_index]
    W_test = W[test_index]

    for i in range(num_outcomes):   
        Y_0_test[i] = Y_0[i][test_index]
        Y_1_test[i] = Y_1[i][test_index]   
        
        Y_test[i] = W_test * Y_1_test[i] + (1 - W_test) * Y_0_test[i]
        Y_cf_test[i] = W_test * Y_0_test[i] + (1 - W_test) * Y_1_test[i]
        T_true_test[i] = Y_1_test[i] - Y_0_test[i]

    # combine data
    train_data = (X_train, W_train, Y_train, Y_0_train, Y_1_train, Y_cf_train, T_true_train)
    test_data = (X_test, W_test, Y_test, Y_0_test, Y_1_test, Y_cf_test, T_true_test)

    return train_data, test_data, train_index, test_index

def synth_outcome (type, X, var, var_shift, cov_inds, outcome_ind):
    # sample random coefficients
    coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]
    BetaB = np.random.choice(coeffs_, size=9, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])

    if outcome_ind == 1: # first outcome
        logi0 = lambda x, shift: 1 / (1 + np.exp(-(x - shift))) + 20
        logi1 = lambda x, shift: 20 / (1 + np.exp(-(x - shift)))

    elif outcome_ind == 2: # second outcome
        # Normal range of ALT: 4 - 36 U/L
        logi0 = lambda x, shift: 1 / (1 + np.exp(-(x - shift))) + x
        logi1 = lambda x, shift: x / (1 + np.exp(-(x - shift))) + x

    # calculate outcome
    if type == 'corr_out':
        other_cov_inds = [j for j in range(10) if j not in cov_inds]
        BetaB = np.random.choice(coeffs_, size=len(other_cov_inds), replace=True, 
                                 p=[0.6, 0.1, 0.1, 0.1, 0.1])
    else:
        other_cov_inds = [j for j in range(10) if j not in cov_inds]
    
    MU_0_ = np.dot(X[:, other_cov_inds], BetaB)
    MU_1_ = np.dot(X[:, other_cov_inds], BetaB)

    MU_0 = MU_0_ + logi0(var, var_shift)
    MU_1 = MU_1_ + logi1(var, var_shift)

    return MU_0, MU_1


def create_synth_uncorr_out (num_outcomes):
    """ 
        Based on the initial clinical trial of remdesivir (uncorrelated outcomes) 
    """    
    sample_no = train_sample_no + test_sample_no

    # define weights
    W = np.random.binomial(1, 0.5, size=sample_no)

    # define X
    X, X_means = simulate_synth_covs(sample_no)

    X_original = X

    X_ = pd.DataFrame(X)
    X_ = normalize_mean(X_)
    X = np.array(X_)

    # first outcome
    cov_ind_1 = 9 # time
    VAR = X_original[:, cov_ind_1]
    VAR_SHIFT = X_means[cov_ind_1]
    MU_0, MU_1 = synth_outcome ('uncorr_out', X, VAR, VAR_SHIFT, [cov_ind_1], 1)

    Y_0 = [np.random.normal(scale=0.1, size=len(X)) + MU_0]
    Y_1 = [np.random.normal(scale=0.1, size=len(X)) + MU_1]

    # second outcome
    cov_ind_2 = 6 # ALT
    VAR = X_original[:, cov_ind_2]
    VAR_SHIFT = X_means[cov_ind_2]
    MU_0, MU_1 = synth_outcome ('uncorr_out', X, VAR, VAR_SHIFT, [cov_ind_2], 2)

    Y_0.append(np.random.normal(scale=0.1, size=len(X)) + MU_0)
    Y_1.append(np.random.normal(scale=0.1, size=len(X)) + MU_1)

    # get train/test data
    train_data, test_data, \
        train_index, test_index = get_train_test_data (num_outcomes, sample_no, X, W, Y_0, Y_1)

    X_original_train = np.array(X_original)[train_index]
    X_original_test = np.array(X_original)[test_index]

    return train_data, test_data, X_original_train, X_original_test


def create_synth_corr_cov (num_outcomes):
    """ 
        Based on the initial clinical trial of remdesivir (correlated covariates) 
    """
    sample_no = train_sample_no + test_sample_no

    # define weights
    W = np.random.binomial(1, 0.5, size=sample_no)

    # define X
    X, X_means = simulate_synth_covs(sample_no, True)

    X_original = X

    X_ = pd.DataFrame(X)
    X_ = normalize_mean(X_)
    X = np.array(X_)

    # first outcome
    cov_ind_1 = 9 # time
    VAR = X_original[:, cov_ind_1]
    VAR_SHIFT = X_means[cov_ind_1]
    MU_0, MU_1 = synth_outcome ('corr_cov', X, VAR, VAR_SHIFT, [cov_ind_1], 1)

    Y_0 = [np.random.normal(scale=0.1, size=len(X)) + MU_0]
    Y_1 = [np.random.normal(scale=0.1, size=len(X)) + MU_1]
    
    # second outcome
    cov_ind_2 = 6 # ALT
    VAR = X_original[:, cov_ind_2]
    VAR_SHIFT = X_means[cov_ind_2]
    MU_0, MU_1 = synth_outcome ('corr_cov', X, VAR, VAR_SHIFT, [cov_ind_2], 2)

    Y_0.append(np.random.normal(scale=0.1, size=len(X)) + MU_0)
    Y_1.append(np.random.normal(scale=0.1, size=len(X)) + MU_1)

    # get train/test data
    train_data, test_data, \
        train_index, test_index = get_train_test_data (num_outcomes, sample_no, X, W, Y_0, Y_1)

    X_original_train = np.array(X_original)[train_index]
    X_original_test = np.array(X_original)[test_index]

    return train_data, test_data, X_original_train, X_original_test


def create_synth_hetsked (num_outcomes, std=0.8):
    """ 
        Based on the initial clinical trial of remdesivir (heteroskedastic data) 
    """  
    sample_no = train_sample_no + test_sample_no

    # define weights
    W = np.random.binomial(1, 0.5, size=sample_no)

    # define X
    X, X_means = simulate_synth_covs(sample_no)

    X_original = X

    X_ = pd.DataFrame(X)
    X_ = normalize_mean(X_)
    X = np.array(X_)

    # first outcome
    cov_ind_1 = 9 # time
    VAR = X_original[:, cov_ind_1]
    VAR_SHIFT = X_means[cov_ind_1]
    MU_0, MU_1 = synth_outcome ('uncorr_out', X, VAR, VAR_SHIFT, [cov_ind_1], 1)

    # add heteroskedasticity dependent on time
    het_std = std * (X[:, cov_ind_1] + np.abs(min(X[:, cov_ind_1])))
    Y_0 = [np.random.normal(scale=het_std, size=len(X)) + MU_0]
    Y_1 = [np.random.normal(scale=het_std, size=len(X)) + MU_1]

    # second outcome
    cov_ind_2 = 6 # ALT
    VAR = X_original[:, cov_ind_2]
    VAR_SHIFT = X_means[cov_ind_2]
    MU_0, MU_1 = synth_outcome ('uncorr_out', X, VAR, VAR_SHIFT, [cov_ind_2], 2)
    
    # add heteroskedasticity dependent on chosen variable
    het_std = std * (X[:, cov_ind_2] + np.abs(min(X[:, cov_ind_2])))
    Y_0.append(np.random.normal(scale=het_std, size=len(X)) + MU_0)
    Y_1.append(np.random.normal(scale=het_std, size=len(X)) + MU_1)

    # get train/test data
    train_data, test_data, \
        train_index, test_index = get_train_test_data (num_outcomes, sample_no, X, W, Y_0, Y_1)

    X_original_train = np.array(X_original)[train_index]
    X_original_test = np.array(X_original)[test_index]

    return train_data, test_data, X_original_train, X_original_test


#######################################################################################################################
# Semi-synthetic data (IHDP)
#######################################################################################################################

def create_IHDP(num_outcomes, test_frac=0.2, noise=0.1):    
    # Load ihdp data
    Dataset = simulate_ihdp(num_outcomes, noise) 
    
    # Split to train/test
    num_samples = len(Dataset)
    train_size = int(np.floor(num_samples * (1 - test_frac)))

    train_index = list(np.random.choice(range(num_samples), train_size, replace=False))
    test_index = list(set(list(range(num_samples))) - set(train_index))

    # Extract for covariates, and treatment
    feat_name = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'
    
    Data_train = Dataset.loc[Dataset.index[train_index]]
    Data_test = Dataset.loc[Dataset.index[test_index]]
    
    # training
    X_train = np.array(Data_train[feat_name.split()])
    W_train = np.array(Data_train['Treatment'])
    
    Y_0_train = [None]*num_outcomes
    Y_1_train = [None]*num_outcomes
    Y_cf_train = [None]*num_outcomes
    Y_train = [None]*num_outcomes  
    T_true_train = [None]*num_outcomes
    
    for i in range(num_outcomes):    
        Y_0_train[i] = np.array(Data_train['Y0_{}'.format(i)])
        Y_1_train[i] = np.array(Data_train['Y1_{}'.format(i)])
        
        Y_train[i] = np.array(Data_train['Response_{}'.format(i)])
        Y_cf_train[i] = W_train * Y_0_train[i] + (1 - W_train) * Y_1_train[i]
        T_true_train[i] = np.array(Data_train['TE_{}'.format(i)])

    # testing
    X_test = np.array(Data_test[feat_name.split()])
    W_test = np.array(Data_test['Treatment'])
    
    Y_0_test = [None]*num_outcomes
    Y_1_test = [None]*num_outcomes
    Y_cf_test = [None]*num_outcomes
    Y_test = [None]*num_outcomes  
    T_true_test = [None]*num_outcomes
    
    for i in range(num_outcomes):   
        Y_0_test[i] = np.array(Data_test['Y0_{}'.format(i)])
        Y_1_test[i] = np.array(Data_test['Y1_{}'.format(i)])
        
        Y_test[i] = np.array(Data_test['Response_{}'.format(i)])
        Y_cf_test[i] = W_test * Y_0_test[i] + (1 - W_test) * Y_1_test[i]
        T_true_test[i] = np.array(Data_test['TE_{}'.format(i)])

    train_data = (X_train, W_train, Y_train, Y_0_train, Y_1_train, Y_cf_train, T_true_train)
    test_data = (X_test, W_test, Y_test, Y_0_test, Y_1_test, Y_cf_test, T_true_test)

    return train_data, test_data

def simulate_ihdp (num_outcomes, noise):  
    # Set seeds for certain characteristics
    seed_1, seed_2 = 458, 39

    # Load ihdp data
    db = pd.read_csv(path_to_ihdp, header = None)
    # Rename columns
    col = ['Treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1', ]
    for i in range(1, 26):
        col.append("X" + str(i))
    db.columns = col
        
    # Extract 25 covariates, and treatment
    covs = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'.split()

    X = np.array(db[covs])
    W = np.array(db['Treatment'])

    full_db = pd.DataFrame(W, columns=['Treatment'])
    
    ################################ Outcome 1 ################################
    # cognitive development score (stanford-binet): 80 - 160+
    rng = np.random.default_rng(seed_1)
    
    coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]
    dim_x = X.shape[1] 

    beta_b = rng.choice(coeffs_, size=dim_x, replace=True, 
                            p=[0.6, 0.1, 0.1, 0.1, 0.1])

    Y0_hat = np.dot(X, beta_b) + 84 # avg. results for control: 84
    Y1_hat = np.exp(np.dot(X + 0.5, beta_b)) + 80 # avg. results for treated: 94

    offset = np.mean(Y1_hat[W==1] - Y0_hat[W==1]) - 4
    Y1_hat = Y1_hat + offset

    Y0 = rng.normal(scale=noise, size=len(X)) + Y0_hat
    Y1 = rng.normal(scale=noise, size=len(X)) + Y1_hat

    tau = Y1_hat - Y0_hat
    Y = np.transpose(np.array([W * Y1 + (1 - W) * Y0, tau]))

    # Add outcomes to dataset
    y_db = pd.DataFrame(Y, columns=['Response_0', 'TE_0'])
    full_db = full_db.join(y_db)
    full_db['Y0_0'] = Y0
    full_db['Y1_0'] = Y1
    
    ################################ Outcome 2 ################################
    # health status score: 0 - 3
    if num_outcomes > 1:    
        rng = np.random.default_rng(seed_2) # 498

        coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]
        dim_x = X.shape[1] 

        beta_b = rng.choice(coeffs_, size=dim_x, replace=True, 
                                p=[0.6, 0.1, 0.1, 0.1, 0.1])

        Y0_hat = np.dot(X, beta_b)
        Y1_hat = np.exp(np.dot(X + 0.5, beta_b)) / 2 # scaled to keep results within range

        offset = np.mean(Y1_hat[W==1] - Y0_hat[W==1]) - 4
        Y1_hat = Y1_hat + offset

        Y0 = rng.normal(scale=noise, size=len(X)) + Y0_hat
        Y1 = rng.normal(scale=noise, size=len(X)) + Y1_hat

        tau = Y1_hat - Y0_hat
        Y = np.transpose(np.array([W * Y1 + (1 - W) * Y0, tau]))

        # Add outcomes to dataset
        y_db = pd.DataFrame(Y, columns=['Response_1', 'TE_1'])
        full_db = full_db.join(y_db)
        full_db['Y0_1'] = Y0
        full_db['Y1_1'] = Y1

    # Add covariates to dataset
    x_db = pd.DataFrame(X, columns=covs)
    full_db = full_db.join(x_db)
    
    return full_db



def normalize_mean(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (result[feature_name] - result[feature_name].mean()) / result[feature_name].std()
    return result
