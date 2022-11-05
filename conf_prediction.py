import numpy as np
from helper import IcpRegressor_scr, RegressorNc_scr, RegressorNc_quantile, IcpRegressor_quantile, QuantileRegErrFunc
from nonconformist.nc import AbsErrorErrFunc

# These method were adapted from:
# • SCR: https://github.com/ryantibs/conformal/blob/master/LICENSE
# • SCQR: https://github.com/yromano/cqr/blob/master/LICENSE

##############################################################################################################
# SCR
##############################################################################################################

def SCR (self, i, treat_data, control_data):
    # Randomly split into two equal sized subsets I_treat, I_val
    train_rows_treat, val_rows_treat, train_outcome_treat, val_labels_treat = treat_data
    train_rows_control, val_rows_control, train_outcome_control, val_labels_control = control_data

    # combine validation set
    val_rows = np.concatenate([val_rows_treat, val_rows_control])

    ######################################## Treatment ########################################
    
    # do conformal prediction, get all CIs
    nc_treat = RegressorNc_scr(self.estimator_treat_dict[i], AbsErrorErrFunc())

    # Fit with training data, calibrate with validation
    icp_treat = IcpRegressor_scr(nc_treat)
    icp_treat.fit(train_rows_treat, train_outcome_treat.reshape((train_outcome_treat.shape[0], 1)))
    icp_treat.calibrate_adapter(val_rows_treat, val_labels_treat)
    cal_scores_treat = icp_treat.cal_scores
    
    ######################################## Control ########################################

    # do conformal prediction, get all CIs
    nc_control = RegressorNc_scr(self.estimator_control_dict[i], AbsErrorErrFunc())

    # Fit with training data, calibrate with validation
    icp_control = IcpRegressor_scr(nc_control)
    icp_control.fit(train_rows_control, train_outcome_control.reshape((train_outcome_control.shape[0], 1)))
    icp_control.calibrate_adapter(val_rows_control, val_labels_control)
    cal_scores_control = icp_control.cal_scores
    
    ######################################## Mu ########################################

    # calculate mu for full group (estimator predictions on validation set)
    val_est_treat_treat = self.estimator_treat_dict[i].predict(val_rows_treat)
    val_est_treat_control = self.estimator_control_dict[i].predict(val_rows_treat)
    val_est_treat_CATE = val_est_treat_treat - val_est_treat_control

    val_est_control_treat = self.estimator_treat_dict[i].predict(val_rows_control)
    val_est_control_control = self.estimator_control_dict[i].predict(val_rows_control)
    val_est_control_CATE = val_est_control_treat - val_est_control_control

    # calculate mu mean
    val_est = np.concatenate([val_est_treat_CATE, val_est_control_CATE])
    est_mean = float(np.mean(val_est))
    
    ######################################## Upper/lower CI bounds ########################################
    
    # combine validation set
    val_rows_est_treat = np.concatenate([val_est_treat_treat, val_est_control_treat])
    val_rows_est_control = np.concatenate([val_est_treat_control, val_est_control_control])
    
    intv_treat = icp_treat.predict(val_rows, significance=self.significance, est_input=val_rows_est_treat)
    intv_control = icp_control.predict(val_rows, significance=self.significance, est_input=val_rows_est_control)
    intv = self.get_TE_CI(intv_treat, intv_control)
    intv_len = np.mean(intv[:, 1] - intv[:, 0])
    
    intv_treat_split = icp_treat.predict(val_rows, significance=self.sig_for_split, est_input=val_rows_est_treat)
    intv_control_split = icp_control.predict(val_rows, significance=self.sig_for_split, est_input=val_rows_est_control)
    intv_split = self.get_TE_CI(intv_treat_split, intv_control_split)

    # return results
    intv_info = intv, intv_len, intv_split
    icp = icp_treat, icp_control
    cal_scores = cal_scores_treat, cal_scores_control
    val_est_res = val_est_treat_treat, val_est_treat_control, val_est_control_treat, val_est_control_control

    return est_mean, intv_info, icp, cal_scores, val_est_res

def SCR_r (self, node, i, val_set_treat, val_set_control, val_idx_treat, val_idx_control):
    # combine validation set
    val_set = np.concatenate([val_set_treat, val_set_control])

    ######################################## Treatment ########################################

    # do conformal prediction, get all CIs
    cal_scores_treat = {0: np.sort(node.cal_scores_treat_dict[i][1][val_idx_treat],0)[::-1],1:node.cal_scores_treat_dict[i][1][val_idx_treat]}

    ######################################## Control ########################################

    # do conformal prediction, get all CIs
    cal_scores_control = {0: np.sort(node.cal_scores_control_dict[i][1][val_idx_control],0)[::-1],1:node.cal_scores_control_dict[i][1][val_idx_control]}

    ######################################## Mu ########################################

    # calculate mu for branch (estimator predictions on validation set)
    val_set_est_treat_treat = node.est_treat_treat_dict[i][val_idx_treat]
    val_set_est_control_treat = node.est_control_treat_dict[i][val_idx_control]
    val_set_est_treat = np.concatenate([val_set_est_treat_treat, val_set_est_control_treat])
    
    val_set_est_treat_control = node.est_treat_control_dict[i][val_idx_treat]
    val_set_est_control_control = node.est_control_control_dict[i][val_idx_control]
    val_set_est_control = np.concatenate([val_set_est_treat_control, val_set_est_control_control])
    
    # calculate mu mean for branch
    val_est = val_set_est_treat - val_set_est_control
    est_mean = float(np.mean(val_est))

    ######################################## Upper/lower CI bounds ########################################

    intv_treat = node.conf_pred_treat_dict[i].predict_given_scores(val_set,
                                                            significance=self.significance,
                                                            cal_scores=cal_scores_treat,
                                                            est_input=val_set_est_treat)
    intv_control = node.conf_pred_control_dict[i].predict_given_scores(val_set,
                                                                significance=self.significance,
                                                                cal_scores=cal_scores_control,
                                                                est_input=val_set_est_control)
    intv = self.get_TE_CI(intv_treat, intv_control)
    intv_len = np.mean(intv[:, 1] - intv[:, 0])

    intv_treat_split = node.conf_pred_treat_dict[i].predict_given_scores(val_set,
                                                                    significance=self.sig_for_split,
                                                                    cal_scores=cal_scores_treat,
                                                                    est_input=val_set_est_treat)
    intv_control_split = node.conf_pred_control_dict[i].predict_given_scores(val_set,
                                                                        significance=self.sig_for_split,
                                                                        cal_scores=cal_scores_control,
                                                                        est_input=val_set_est_control)
    intv_split = self.get_TE_CI(intv_treat_split, intv_control_split)

    # return results
    intv_info = intv, intv_len, intv_split
    cal_scores = cal_scores_treat, cal_scores_control
    val_set_est_res = val_set_est_treat_treat, val_set_est_treat_control, val_set_est_control_treat, val_set_est_control_control

    return est_mean, intv_info, cal_scores, val_set_est_res


##############################################################################################################
# SCQR
##############################################################################################################

def SCQR (self, i, treat_data, control_data):
    # Randomly split into two equal sized subsets I_treat, I_val
    train_rows_treat, val_rows_treat, train_outcome_treat, val_labels_treat = treat_data
    train_rows_control, val_rows_control, train_outcome_control, val_labels_control = control_data

    # combine validation set
    val_rows = np.concatenate([val_rows_treat, val_rows_control])
    
    ######################################## Treatment ########################################
    
    # do conformal prediction, get all CIs
    nc_treat = RegressorNc_quantile(self.estimator_treat_dict[i], QuantileRegErrFunc())
    
    # Fit with training data, calibrate with validation
    icp_treat = IcpRegressor_quantile(nc_treat)

    icp_treat.fit(train_rows_treat, train_outcome_treat)
    
    # calculate mu for full group (estimator predictions on validation set)
    val_est_treat_treat = [self.estimator_treat_dict[i].predict(val_rows_treat, significance_flag= 'S_sig'),
                            self.estimator_treat_dict[i].predict(val_rows_treat, significance_flag= 'W_sig'),
                            self.estimator_treat_dict[i].predict(val_rows_treat, significance_flag= 'mse')]
    val_est_control_treat = [self.estimator_treat_dict[i].predict(val_rows_control, significance_flag= 'S_sig'),
                             self.estimator_treat_dict[i].predict(val_rows_control, significance_flag= 'W_sig'),
                             self.estimator_treat_dict[i].predict(val_rows_control, significance_flag= 'mse')]
    val_rows_est_treat_S = np.concatenate([val_est_treat_treat[0], val_est_control_treat[0]])
    val_rows_est_treat_W = np.concatenate([val_est_treat_treat[1], val_est_control_treat[1]])

    # W
    icp_treat.calibrate_quantiles(val_rows_treat, val_labels_treat, significance_flag = 'W_sig')
    cal_scores_treat_W = icp_treat.cal_scores
    intv_treat = icp_treat.predict(val_rows, significance=self.significance, est_input=val_rows_est_treat_W, significance_flag = 'W_sig')
    
    # S
    icp_treat.calibrate_quantiles(val_rows_treat, val_labels_treat,significance_flag = 'S_sig')
    cal_scores_treat_S = icp_treat.cal_scores
    intv_treat_split = icp_treat.predict(val_rows, significance=self.sig_for_split, est_input=val_rows_est_treat_S, significance_flag = 'S_sig')

    cal_scores_treat = [cal_scores_treat_S, cal_scores_treat_W]
    
    ######################################## Control ########################################

    # do conformal prediction, get all CIs
    nc_control = RegressorNc_quantile(self.estimator_control_dict[i], QuantileRegErrFunc())

    # Fit with training data, calibrate with validation
    icp_control = IcpRegressor_quantile(nc_control)

    icp_control.fit(train_rows_control, train_outcome_control)
    
    # calculate mu for full group (estimator predictions on validation set)
    val_est_treat_control = [self.estimator_control_dict[i].predict(val_rows_treat,significance_flag= 'S_sig'),self.estimator_control_dict[i].predict(val_rows_treat,significance_flag= 'W_sig'),self.estimator_control_dict[i].predict(val_rows_treat,significance_flag= 'mse')]
    val_est_control_control = [self.estimator_control_dict[i].predict(val_rows_control,significance_flag= 'S_sig'),self.estimator_control_dict[i].predict(val_rows_control,significance_flag= 'W_sig'),self.estimator_control_dict[i].predict(val_rows_control,significance_flag= 'mse')]
    val_rows_est_control_S = np.concatenate([val_est_treat_control[0], val_est_control_control[0]])
    val_rows_est_control_W = np.concatenate([val_est_treat_control[1], val_est_control_control[1]])
    
    # W significance
    icp_control.calibrate_quantiles(val_rows_control, val_labels_control,significance_flag = 'W_sig')
    cal_scores_control_W = icp_control.cal_scores
    intv_control = icp_control.predict(val_rows, significance=self.significance, est_input=val_rows_est_control_W,significance_flag = 'W_sig')
    
    # S significance
    icp_control.calibrate_quantiles(val_rows_control, val_labels_control,significance_flag = 'S_sig')
    cal_scores_control_S = icp_control.cal_scores
    intv_control_split = icp_control.predict(val_rows, significance=self.sig_for_split, est_input=val_rows_est_control_S,significance_flag = 'S_sig')

    cal_scores_control = [cal_scores_control_S,cal_scores_control_W]

    ######################################## Mu ########################################

    # Get CATES
    val_est_treat_CATE = val_est_treat_treat[2] - val_est_treat_control[2]
    val_est_control_CATE = val_est_control_treat[2] - val_est_control_control[2]
    
    # calculate mu mean
    val_est = np.concatenate([val_est_treat_CATE, val_est_control_CATE])
    est_mean = float(np.mean(val_est))

    ######################################## Upper/lower CI bounds ########################################

    intv = self.get_TE_CI(intv_treat, intv_control)
    intv_len = np.mean(intv[:, 1] - intv[:, 0])

    intv_split = self.get_TE_CI(intv_treat_split, intv_control_split)

    # return results
    intv_info = intv, intv_len, intv_split
    icp = icp_treat, icp_control
    cal_scores = cal_scores_treat, cal_scores_control
    val_est_res = val_est_treat_treat, val_est_treat_control, val_est_control_treat, val_est_control_control
    
    return est_mean, intv_info, icp, cal_scores, val_est_res

def SCQR_r (self, node, i, val_set_treat, val_set_control, val_idx_treat, val_idx_control):

    # calculate mu for branch (estimator predictions on validation set)
    # quantiles 
    val_set_est_treat_treat = [node.est_treat_treat_dict[i][0][val_idx_treat],node.est_treat_treat_dict[i][1][val_idx_treat],node.est_treat_treat_dict[i][2][val_idx_treat]]
    val_set_est_control_treat = [node.est_control_treat_dict[i][0][val_idx_control],node.est_control_treat_dict[i][1][val_idx_control],node.est_control_treat_dict[i][2][val_idx_control]]
    
    val_set_est_treat_control = [node.est_treat_control_dict[i][0][val_idx_treat],node.est_treat_control_dict[i][1][val_idx_treat],node.est_treat_control_dict[i][2][val_idx_treat]]
    val_set_est_control_control = [node.est_control_control_dict[i][0][val_idx_control],node.est_control_control_dict[i][1][val_idx_control],node.est_control_control_dict[i][2][val_idx_control]]
    
    # S
    val_set_est_treat_S = np.concatenate([val_set_est_treat_treat[0], val_set_est_control_treat[0]])
    val_set_est_control_S = np.concatenate([val_set_est_treat_control[0], val_set_est_control_control[0]])

    # W
    val_set_est_treat_W = np.concatenate([val_set_est_treat_treat[1], val_set_est_control_treat[1]])
    val_set_est_control_W = np.concatenate([val_set_est_treat_control[1], val_set_est_control_control[1]])
    
    val_set_est_treat = [val_set_est_treat_S,val_set_est_treat_W]
    val_set_est_control =[val_set_est_control_S,val_set_est_control_W]
    
    # mean
    val_set_est_treat_m = np.concatenate([val_set_est_treat_treat[2], val_set_est_control_treat[2]])
    val_set_est_control_m = np.concatenate([val_set_est_treat_control[2], val_set_est_control_control[2]])

    # calculate mu mean for branch
    val_est = val_set_est_treat_m - val_set_est_control_m
    est_mean = float(np.mean(val_est))

    # combine validation set
    val_set = np.concatenate([val_set_treat, val_set_control])

    ######################################## Treatment ########################################

    # do conformal prediction, get all CIs
    cal_scores_treat_W = {0: np.sort(node.cal_scores_treat_dict[i][1][1][val_idx_treat],0)[::-1],1:node.cal_scores_treat_dict[i][1][1][val_idx_treat]}
    cal_scores_treat_S ={0: np.sort(node.cal_scores_treat_dict[i][0][1][val_idx_treat],0)[::-1],1:node.cal_scores_treat_dict[i][0][1][val_idx_treat]}
    cal_scores_treat =  [cal_scores_treat_S ,cal_scores_treat_W]
    
    # calculate partition measure, get upper and lower CI bounds for branch
    intv_treat = node.conf_pred_treat_dict[i].predict_given_scores(val_set,
                                                            significance=self.significance,
                                                            cal_scores=cal_scores_treat_W,
                                                            est_input=val_set_est_treat_W,significance_flag = 'W_sig')
    
    intv_treat_split = node.conf_pred_treat_dict[i].predict_given_scores(val_set,
                                                                    significance=self.sig_for_split,
                                                                    cal_scores=cal_scores_treat_S,
                                                                    est_input=val_set_est_treat_S,significance_flag = 'S_sig')
    

    ######################################## Control ########################################

    # do conformal prediction, get all CIs
    cal_scores_control_W = {0: np.sort(node.cal_scores_control_dict[i][1][1][val_idx_control],0)[::-1],1:node.cal_scores_control_dict[i][1][1][val_idx_control]}
    
    cal_scores_control_S = {0: np.sort(node.cal_scores_control_dict[i][0][1][val_idx_control],0)[::-1],1:node.cal_scores_control_dict[i][0][1][val_idx_control]}
    
    
    cal_scores_control =[cal_scores_control_S,cal_scores_control_W]

    # calculate partition measure, get upper and lower CI bounds for branch
    intv_control = node.conf_pred_control_dict[i].predict_given_scores(val_set,
                                                                significance=self.significance,
                                                                cal_scores=cal_scores_control_W,
                                                                est_input=val_set_est_control_W,
                                                                significance_flag = 'W_sig')
    
    intv_control_split = node.conf_pred_control_dict[i].predict_given_scores(val_set,
                                                                        significance=self.sig_for_split,
                                                                        cal_scores=cal_scores_control_S,
                                                                        est_input=val_set_est_control_S,
                                                                        significance_flag = 'S_sig')
    
    ######################################## Upper/lower CI bounds ########################################

    intv = self.get_TE_CI(intv_treat, intv_control)
    intv_len = np.mean(intv[:, 1] - intv[:, 0])
    
    intv_split = self.get_TE_CI(intv_treat_split, intv_control_split)

    # return results
    intv_info = intv, intv_len, intv_split
    cal_scores = cal_scores_treat, cal_scores_control
    val_set_est_res = val_set_est_treat_treat, val_set_est_treat_control, val_set_est_control_treat, val_set_est_control_control

    return est_mean, intv_info, cal_scores, val_set_est_res

def SCQR_short (self, i, treat_data, control_data):
    
    train_rows_treat, val_rows_treat, train_outcome_treat, val_labels_treat = treat_data
    train_rows_control, val_rows_control, train_outcome_control, val_labels_control = control_data

    val_rows = np.concatenate([val_rows_treat, val_rows_control])
    
    ######################################## Treatment ########################################
    
    # do conformal prediction, get all CIs
    nc_treat = RegressorNc_quantile(self.estimator_treat_dict[i], QuantileRegErrFunc())
    
    # mu = A({ (X_i, Y_i) : i in I_1 })
    # Fit with training data, calibrate with validation
    icp_treat = IcpRegressor_quantile(nc_treat)

    #icp_treat.fit(train_rows_treat, train_outcome_treat.reshape((train_outcome_treat.shape[0], 1)))
    icp_treat.fit(train_rows_treat, train_outcome_treat)
    
    ### Get estimates
    val_est_treat_treat = [self.estimator_treat_dict[i].predict(val_rows_treat,significance_flag= 'S_sig'),self.estimator_treat_dict[i].predict(val_rows_treat,significance_flag= 'W_sig'),self.estimator_treat_dict[i].predict(val_rows_treat,significance_flag= 'mse')]
    val_est_control_treat = [self.estimator_treat_dict[i].predict(val_rows_control,significance_flag= 'S_sig'),self.estimator_treat_dict[i].predict(val_rows_control,significance_flag= 'W_sig'),self.estimator_treat_dict[i].predict(val_rows_control,significance_flag= 'mse')]
    val_rows_est_treat_S = np.concatenate([val_est_treat_treat[0], val_est_control_treat[0]])
    val_rows_est_treat_W = np.concatenate([val_est_treat_treat[1], val_est_control_treat[1]])

    # W
    icp_treat.calibrate_quantiles(val_rows_treat, val_labels_treat,significance_flag = 'W_sig')
    cal_scores_treat_W = icp_treat.cal_scores
    intv_treat = icp_treat.predict(val_rows, significance=self.significance, est_input=val_rows_est_treat_W,significance_flag = 'W_sig')
    
    #S
    icp_treat.calibrate_quantiles(val_rows_treat, val_labels_treat,significance_flag = 'S_sig')
    cal_scores_treat_S = icp_treat.cal_scores
    intv_treat_split = icp_treat.predict(val_rows, significance=self.sig_for_split, est_input=val_rows_est_treat_S,significance_flag = 'S_sig')

    cal_scores_treat = [cal_scores_treat_S,cal_scores_treat_W]
    
    ######################################## Control ########################################

    # do conformal prediction, get all CIs
    nc_control = RegressorNc_quantile(self.estimator_control_dict[i], QuantileRegErrFunc())

    # Fit with training data, calibrate with validation
    icp_control = IcpRegressor_quantile(nc_control)
    icp_control.fit(train_rows_control, train_outcome_control)
    
    # Get estimates
    val_est_treat_control = [self.estimator_control_dict[i].predict(val_rows_treat,significance_flag= 'S_sig'),self.estimator_control_dict[i].predict(val_rows_treat,significance_flag= 'W_sig'),self.estimator_control_dict[i].predict(val_rows_treat,significance_flag= 'mse')]
    val_est_control_control = [self.estimator_control_dict[i].predict(val_rows_control,significance_flag= 'S_sig'),self.estimator_control_dict[i].predict(val_rows_control,significance_flag= 'W_sig'),self.estimator_control_dict[i].predict(val_rows_control,significance_flag= 'mse')]
    val_rows_est_control_S = np.concatenate([val_est_treat_control[0], val_est_control_control[0]])
    val_rows_est_control_W = np.concatenate([val_est_treat_control[1], val_est_control_control[1]])
    
    # W
    icp_control.calibrate_quantiles(val_rows_control, val_labels_control,significance_flag = 'W_sig')
    cal_scores_control_W = icp_control.cal_scores
    intv_control = icp_control.predict(val_rows, significance=self.significance, est_input=val_rows_est_control_W,significance_flag = 'W_sig')
    
    # S 
    icp_control.calibrate_quantiles(val_rows_control, val_labels_control,significance_flag = 'S_sig')
    cal_scores_control_S = icp_control.cal_scores
    intv_control_split = icp_control.predict(val_rows, significance=self.sig_for_split, est_input=val_rows_est_control_S,significance_flag = 'S_sig')

    cal_scores_control = [cal_scores_control_S,cal_scores_control_W]

    ######################################## Mu ########################################

    # Get CATES
    val_est_treat_CATE = val_est_treat_treat[2] - val_est_treat_control[2]
    val_est_control_CATE = val_est_control_treat[2] - val_est_control_control[2]
    
    # calculate mu mean
    val_est = np.concatenate([val_est_treat_CATE, val_est_control_CATE])
    est_mean = float(np.mean(val_est))

    ######################################## Upper/lower CI bounds ########################################

    intv = self.get_TE_CI(intv_treat, intv_control)
    intv_len = np.mean(intv[:, 1] - intv[:, 0])

    intv_split = self.get_TE_CI(intv_treat_split, intv_control_split)

    # return results
    intv_info = intv, intv_len, intv_split
    icp = icp_treat, icp_control
    
    intv_both = [[intv_treat,intv_treat_split],[intv_control,intv_control_split]]
    
    cal_scores = cal_scores_treat, cal_scores_control
    val_est_res = val_est_treat_treat, val_est_treat_control, val_est_control_treat, val_est_control_control
    
    return est_mean, intv_info, icp, cal_scores, val_est_res, intv_both


## Condensed recursive partitioning
def SCQR_r_short (self, node, i, val_set_treat, val_set_control, val_idx_treat, val_idx_control):

    # calculate mu for branch (estimator predictions on validation set)
    # quantiles 
    val_set_est_treat_treat = [node.est_treat_treat_dict[i][0][val_idx_treat],node.est_treat_treat_dict[i][1][val_idx_treat],node.est_treat_treat_dict[i][2][val_idx_treat]]
    val_set_est_control_treat = [node.est_control_treat_dict[i][0][val_idx_control],node.est_control_treat_dict[i][1][val_idx_control],node.est_control_treat_dict[i][2][val_idx_control]]
    
    val_set_est_treat_control = [node.est_treat_control_dict[i][0][val_idx_treat],node.est_treat_control_dict[i][1][val_idx_treat],node.est_treat_control_dict[i][2][val_idx_treat]]
    val_set_est_control_control = [node.est_control_control_dict[i][0][val_idx_control],node.est_control_control_dict[i][1][val_idx_control],node.est_control_control_dict[i][2][val_idx_control]]
    
    # S
    val_set_est_treat_S = np.concatenate([val_set_est_treat_treat[0], val_set_est_control_treat[0]])
    val_set_est_control_S = np.concatenate([val_set_est_treat_control[0], val_set_est_control_control[0]])

    # W
    val_set_est_treat_W = np.concatenate([val_set_est_treat_treat[1], val_set_est_control_treat[1]])
    val_set_est_control_W = np.concatenate([val_set_est_treat_control[1], val_set_est_control_control[1]])
    
    # mean
    val_set_est_treat_m = np.concatenate([val_set_est_treat_treat[2], val_set_est_control_treat[2]])
    val_set_est_control_m = np.concatenate([val_set_est_treat_control[2], val_set_est_control_control[2]])

    # calculate mu mean for branch
    val_est = val_set_est_treat_m - val_set_est_control_m
    est_mean = float(np.mean(val_est))
    
    ######################################## Intervals ########################################
    intv_info_treat  = node.intv_both_dict[i][0]
    intv_info_control  = node.intv_both_dict[i][1]
    
    intv_treat=intv_info_treat[0][np.concatenate([val_idx_treat, val_idx_control])]
    intv_treat_split=intv_info_treat[1][np.concatenate([val_idx_treat, val_idx_control])]
    
    intv_control=intv_info_control[0][np.concatenate([val_idx_treat, val_idx_control])]
    intv_control_split=intv_info_control[1][np.concatenate([val_idx_treat, val_idx_control])]
    
    cal_scores_control_W = {0: np.sort(node.cal_scores_control_dict[i][1][1][val_idx_control],0)[::-1],1:node.cal_scores_control_dict[i][1][1][val_idx_control]}
    
    cal_scores_control_S = {0: np.sort(node.cal_scores_control_dict[i][0][1][val_idx_control],0)[::-1],1:node.cal_scores_control_dict[i][0][1][val_idx_control]}
    
    cal_scores_control =[cal_scores_control_S,cal_scores_control_W]
    
    ######################################## Upper/lower CI bounds ########################################

    # calculate partition measure, get upper and lower CI bounds for branch
    intv_both = [[intv_treat,intv_treat_split],[intv_control,intv_control_split]]
    intv = self.get_TE_CI(intv_treat, intv_control)
    intv_len = np.mean(intv[:, 1] - intv[:, 0])
    
    intv_split = self.get_TE_CI(intv_treat_split, intv_control_split)

    # return results
    intv_info = intv, intv_len, intv_split
    cal_scores = cal_scores_control, cal_scores_control
    val_set_est_res = val_set_est_treat_treat, val_set_est_treat_control, val_set_est_control_treat, val_set_est_control_control

    return est_mean, intv_info, cal_scores, val_set_est_res, intv_both


