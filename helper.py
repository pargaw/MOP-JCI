# -*- coding: utf-8 -*-
import numpy as np
import pickle

from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc,RegressionErrFunc,AbsErrorErrFunc

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')
from sgarden.skgarden import RandomForestQuantileRegressor

# -----------------------------------------------------------------------------
# SCR
# -----------------------------------------------------------------------------

class RegressorAdapter_HTE(RegressorAdapter):
    """ Conditional mean estimator, formulated as neural net
        mode= 0 is control 1 is treated
    """

    def __init__(self, model, mode=None):
        super(RegressorAdapter_HTE, self).__init__(model)
        # Instantiate model
        self.model = model
        self.mode = mode

    def fit(self, x, y):
        return

    def predict(self, x):
        # cmgp output: TE_est, Y_est_0, Y_est_1
        # rf ouputs: Y_est
        if self.mode is 1:
            y_hat = self.model.predict(x)[2]
        elif self.mode is 0:
            y_hat = self.model.predict(x)[1]
        else:
            y_hat = self.model.predict(x)
        y_hat = y_hat.squeeze()

        return y_hat


class RegressorNc_scr(RegressorNc):
    def predict(self, x, nc, significance=None, prediction=None):
        n_test = x.shape[0]
        if prediction is None:
            prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)
 
        if significance:
            intervals = np.zeros((x.shape[0], 2))
            # err_dist 
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            
            if prediction.ndim > 1:  # CQR
                intervals[:, 0] = prediction[:, 0] - err_dist[0, :]
                intervals[:, 1] = prediction[:, -1] + err_dist[1, :]
            else:  # regular conformal prediction
                err_dist *= norm
                intervals[:, 0] = prediction - err_dist[0, :]
                intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                # err_dist
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals

class IcpRegressor_scr(IcpRegressor):
    def predict(self, x, significance=None, est_input=None):
        n_significance = (99 if significance is None
                          else np.array(significance).size)
        
        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :],
                                             self.cal_scores[condition],
                                             significance,
                                             est_input)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction

    def predict_given_scores(self, x, significance=None, cal_scores=None, est_input=None):
        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :],
                                             cal_scores[0],
                                             significance,
                                             est_input)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction
    
    def calibrate_adapter(self, x, y, significance_flag = 'W_sig',increment=False):
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i]))
                                     for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :],
                                                    self.cal_y[idx])
                self.cal_scores[cond] = np.sort(cal_scores,0)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
            self.cal_scores = {0: np.sort(cal_scores,0)[::-1],1: cal_scores}


# -----------------------------------------------------------------------------
# SCR using RF
# -----------------------------------------------------------------------------

class ForestRegressorAdapter(RegressorAdapter):
    """ 
    Random Forest estimator (RF)
    """

    def __init__(self,
                 model,
                 fit_params=None,
                 params=None,
                 name = None):
        """ Initialization
        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        params : dictionary of parameters
                params["n_estimators"] : integer, number of trees
                params["max_features"] : string, int or float, number of features at split
                params["max_depth"] : integer, max number of levels in tree
                params["min_samples_split"] : int or float, min number of samples for split
                params["min_samples_leaf"]: int or float, min number of samples at each leaf
                params["bootstrap"]: bool, selecting boostrapped samples for training
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation/feature sampling     
        """
        super(ForestRegressorAdapter, self).__init__(model, fit_params)
        
        # Instantiate model
        self.name = name
        self.params = params
        self.model = RandomForestRegressor(n_estimators=params["n_estimators"],
                                        max_features=params["max_features"],
                                        max_depth=params["max_depth"],
                                        min_samples_split=params["min_samples_split"],
                                        min_samples_leaf=params["min_samples_leaf"],
                                        bootstrap=params["bootstrap"],
                                        random_state=params["random_state"]
                                        )

    def fit(self, x, y, tune = False):
        """ Fit the model to data
        Parameters
        ----------
        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)
        """
        # Hyperparam search
        if tune == True: 
            param_dict = self.hyperparameter_search(x, y)
            self.params = param_dict
            self.model = RandomForestRegressor(n_estimators=param_dict["n_estimators"],
                                            max_features=param_dict["max_features"],
                                            max_depth=param_dict["max_depth"],
                                            min_samples_split=param_dict["min_samples_split"],
                                            min_samples_leaf=param_dict["min_samples_leaf"],
                                            bootstrap=param_dict["bootstrap"],
                                            random_state=param_dict["random_state"]
                                            )
        self.model.fit(x, y)

    def hyperparameter_search(self,x, y):        
        name = self.name
        
        # param grid
        param_grid = dict(
            max_features = [1, 'auto', 'sqrt'],
            max_depth = [int(x) for x in np.linspace(2, 50, num = 5)],
            min_samples_split = [2, 3, 5, 10, 20, 30, 50, 70, 100],
            min_samples_leaf = [1, 5, 10, 20, 25],
            bootstrap = [True, False],
            random_state = [0, None]
           )

        try:
            param_dict = pickle.load(open('Params/param_dict_{}.pkl'.format(name),'rb')) 
            print('loading presaved params')
            return param_dict
        except:
            pass
        
        rf = RandomForestRegressor()
        search = HalvingRandomSearchCV(
            rf,
            param_grid,
            n_candidates = 150,
            resource="n_estimators",
            max_resources=500,
            min_resources=50,
            scoring='neg_mean_squared_error',
            n_jobs=2,
            random_state=0,
        ).fit(x, y)

        param_dict = search.best_params_
        print('RF tuning complete')
    
        pickle.dump(param_dict,open('Params/param_dict_{}.pkl'.format(name),'wb')) 
        return param_dict


# -----------------------------------------------------------------------------
# SCQR
# -----------------------------------------------------------------------------

# CQR symmetric error function
class QuantileRegErrFunc(RegressionErrFunc):

    def __init__(self):
        super(QuantileRegErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:,0]
        y_upper = prediction[:,-1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high,error_low)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc,0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])

class RegressorNc_quantile(RegressorNc):
    def __init__(self,
                 model,
	             err_func=AbsErrorErrFunc(),
	             normalizer=None,
	             beta=1e-6):
                 super(RegressorNc, self).__init__(model,
		                                  err_func,
		                                  normalizer,
		                                  beta)
    
    def predict(self, x, nc, significance=None, prediction = None, significance_flag = 'W_sig'):
        n_test = x.shape[0]
        if prediction is None:
            prediction = self.model.predict(x,significance_flag)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], 2))
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            
            if prediction.ndim > 1: # CQR
                intervals[:, 0] = prediction[:,0] - err_dist[0, :]
                intervals[:, 1] = prediction[:,-1] + err_dist[1, :]
            else: # regular conformal prediction
                err_dist *= norm
                intervals[:, 0] = prediction - err_dist[0, :]
                intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else: # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals
    
    def score_quantiles(self, x, y=None, significance_flag = 'W_sig'):
        """Calculates the nonconformity score of a set of samples.
        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.

        y : numpy array of shape [n_samples]
            Outputs of examples for which to calculate a nonconformity score.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of samples.
        """
        prediction = self.model.predict(x,significance_flag)
        n_test = x.shape[0]
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)
        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y)
        else:
            ret_val = self.err_func.apply(prediction, y) / norm
        return ret_val

class IcpRegressor_quantile(IcpRegressor):
    def __init__(self, nc_function, condition=None):
        super(IcpRegressor, self).__init__(nc_function, condition)

    def predict(self, x, significance=None,est_input = None,significance_flag = 'W_sig'):
        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                          for i in range(x.shape[0])])
        
        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :],
                                             self.cal_scores[condition],
                                             significance,est_input,significance_flag )
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction
    
    def predict_given_scores(self, x, significance=None, cal_scores=None, est_input=None,significance_flag = 'W_sig'):
        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                
                p = self.nc_function.predict(x[idx, :],
                                             cal_scores[0],
                                             significance,
                                             est_input,significance_flag)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction
    
    
    def calibrate_quantiles(self, x, y, significance_flag = 'W_sig',increment=False):


        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i]))
                                     for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score_quantiles(self.cal_x[idx, :],
                                                    self.cal_y[idx],significance_flag)
                self.cal_scores[cond] = np.sort(cal_scores,0)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score_quantiles(self.cal_x, self.cal_y,significance_flag)
            #self.cal_scores = {0: np.sort(cal_scores,0)[::-1]}
            ## CAl scores EDITTED HERE
            self.cal_scores = {0: np.sort(cal_scores,0)[::-1],1: cal_scores}

# -----------------------------------------------------------------------------
# SCQR using RF
# -----------------------------------------------------------------------------

class QuantileForestRegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)
    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.
    """

    def __init__(self,
                 model,
                 fit_params=None,
                 S_miscoverage = .8,
                 W_miscoverage = .05,
                 params=None, name = None):
        """ Initialization
        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.
        """
        super(QuantileForestRegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.S_miscoverage = S_miscoverage
        self.W_miscoverage = W_miscoverage
        # self.quantiles = [self.W_miscoverage/2,1-self.W_miscoverage/2] #unused
        # self.cv_quantiles = self.quantiles # Unused
        self.params = params
        
        self.name = name
        
        self.rfqr = {}
        self.rfqr['W_low'] = RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_split=params["min_samples_split"],
                                                  n_estimators=params["n_estimators"])
        self.rfqr['W_hi'] = RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_split=params["min_samples_split"],
                                                  n_estimators=params["n_estimators"])
        self.rfqr['S_low'] = RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_split=params["min_samples_split"],
                                                  n_estimators=params["n_estimators"])
        self.rfqr['S_hi'] =RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_split=params["min_samples_split"],
                                                  n_estimators=params["n_estimators"])
        self.rfqr['mse'] = RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_split=params["min_samples_split"],
                                                  n_estimators=params["n_estimators"])

    def fit(self, x, y,tune = False):
        """ Fit the model to data
        Parameters
        ----------
        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)
        """
        '''if self.params["CV"]: #unused currently
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles = self.tune_params_cv.CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)'''

        self.rfqr['mse'].fit(x, y)
        self.rfqr['S_low'].fit(x, y)
        self.rfqr['S_hi'].fit(x, y)
        self.rfqr['W_low'].fit(x, y)
        self.rfqr['W_hi'].fit(x, y)


    
    def predict(self, x,significance_flag = 'mse'):
        """ Estimate the conditional low and high quantiles given the features
        Parameters
        ----------
        x : numpy array of training features (nXp)
        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)
        """
        S_miscoverage=self.S_miscoverage 
        W_miscoverage =self.W_miscoverage 
        if significance_flag == 'mse':
            mean = self.rfqr['mse'].predict(x,quantile = 50)
            return mean
        elif significance_flag == 'S_sig':
            lower = self.rfqr['S_low'].predict(x,quantile = (S_miscoverage/2)*100)
            upper = self.rfqr['S_hi'].predict(x,quantile = (1-S_miscoverage/2)*100)
            
            ret_val = np.zeros((len(lower),2))
            ret_val[:,0] = lower
            ret_val[:,1] = upper
            return ret_val
        elif significance_flag == 'W_sig':
            lower = self.rfqr['W_low'].predict(x,quantile = (W_miscoverage/2)*100)
            upper = self.rfqr['W_hi'].predict(x,quantile = (1-W_miscoverage/2)*100)

            ret_val = np.zeros((len(lower),2))
            ret_val[:,0] = lower
            ret_val[:,1] = upper
            return ret_val