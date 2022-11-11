from sklearn.model_selection import train_test_split
from utils import *
from conf_prediction import *

# This method was extended from the Robust recursive partitioning algorithm: https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/LICENSE.md

# Multi outcome partitioning (MOP) using a quantile estimator
class MOP_quantile:
    def __init__(self, estimator_treat_dict=None, estimator_control_dict=None,
                 max_depth=-1, min_size=10,
                 conformal_mode="SCR_CMGP", params_qf=None,
                 significance=0.05, weight=0.5, gamma=0.05,
                 sig_for_split=0.8,
                 seed=None,normalize_S_W_regions=0, sum_var = 1, beta = .5):
        print('Condensed Quantile Regression')
        self.root = None
        self.max = -np.inf
        self.min = np.inf
        self.num_leaves = 0
        self.curr_leaves = 0
        self.estimator_treat_dict = estimator_treat_dict
        self.estimator_control_dict = estimator_control_dict

        self.max_depth = max_depth
        self.min_size = min_size

        self.conformal_mode = conformal_mode
        self.params_qf = params_qf
        self.significance = significance
        self.sig_for_split =sig_for_split
        
        self.weight = weight
        self.gamma = gamma
        self.seed = seed
        self.eval_func = self.conf_homo

        self.tree_depth = 0
        self.obj = 0.0
        self.start = 0.0
        self.time = 0.0
        
        self.normalize_S_W_regions = normalize_S_W_regions
        self.beta = beta
        self.sum = sum_var

    class Node:
        def __init__(self, col=-1, value=None, true_branch=None, false_branch=None, leaf=False, leaf_num=None,
                     obj=0.0, homogeneity=0.0, intv_len_dict={0:0.0},
                     est_treat_treat_dict=None, est_treat_control_dict=None,
                     est_control_treat_dict=None, est_control_control_dict=None,
                     conf_pred_treat_dict=None, conf_pred_control_dict=None,
                     cal_scores_treat_dict=None, cal_scores_control_dict=None,
                     node_depth=0,intv_both_dict = None):
            self.col = col  # the column of the feature used for splitting
            self.value = value  # the value that splits the data

            self.est_treat_treat_dict= est_treat_treat_dict
            self.est_treat_control_dict = est_treat_control_dict
            self.est_control_treat_dict = est_control_treat_dict
            self.est_control_control_dict = est_control_control_dict

            self.conf_pred_treat_dict = conf_pred_treat_dict
            self.conf_pred_control_dict = conf_pred_control_dict

            self.cal_scores_treat_dict = cal_scores_treat_dict
            self.cal_scores_control_dict = cal_scores_control_dict

            self.obj = obj
            self.intv_len_dict = intv_len_dict
            self.homogeneity = homogeneity

            self.true_branch = true_branch  # pointer to node for true branch
            self.false_branch = false_branch  # pointer to node for false branch
            self.leaf = leaf  # true/false if leaf or not
            self.leaf_num = leaf_num  # the leaf label
            
            self.intv_both_dict = intv_both_dict
            self.node_depth = node_depth

    def fit(self, num_outcomes, rows_treat, labels_treat, rows_control, labels_control):
        if rows_treat.shape[0] == 0:
            return self.Node()

        if self.seed is not None:
            np.random.seed(self.seed)

        # Keep track of all values per outcome
        intv_dict = {}
        intv_split_dict = {}
        est_mean_dict = {}
        total_val_no_treat_dict = {}
        total_val_no_control_dict = {}

        val_rows_treat_dict = {}
        val_labels_treat_dict = {}
        val_rows_control_dict = {}
        val_labels_control_dict = {}
        val_est_treat_treat_dict = {}
        val_est_treat_control_dict = {}
        val_est_control_treat_dict = {}
        val_est_control_control_dict = {}

        intv_len_dict = {}
        icp_treat_dict = {}
        icp_control_dict = {}
        cal_scores_treat_dict = {}
        cal_scores_control_dict = {}
        
        self.normalization_dict = {}
        intv_both_dict = {}

        for i in range(num_outcomes):

            # split for conformal regression
            train_rows_treat, val_rows_treat, train_outcome_treat, val_labels_treat = \
                train_test_split(rows_treat, labels_treat[i], shuffle=True, test_size=0.5, random_state=0)
            train_rows_control, val_rows_control, train_outcome_control, val_labels_control = \
                train_test_split(rows_control, labels_control[i], shuffle=True, test_size=0.5, random_state=0)

            # check estimator internal error
            error_no_tmp = 0
            FIT_FLAG = True
            while (FIT_FLAG):
                x_train = np.concatenate([train_rows_treat, train_rows_control])
                y_train = np.concatenate([train_outcome_treat, train_outcome_control])
                w_train = np.zeros(x_train.shape[0])
                w_train[0:train_rows_treat.shape[0]] = 1
                if self.conformal_mode == "SCR_CMGP":
                    FIT_FLAG = self.estimator_treat_dict[i].model.fit(x_train, y_train, w_train)
                if self.conformal_mode == "SCQR_RF":
                    FIT_FLAG = self.estimator_treat_dict[i].fit(train_rows_treat, train_outcome_treat, tune = True)
                    FIT_FLAG = self.estimator_control_dict[i].fit(train_rows_control, train_outcome_control, tune = True)
                if self.conformal_mode == "SCR_RF":
                    FIT_FLAG = self.estimator_treat_dict[i].fit(train_rows_treat, train_outcome_treat, tune = True)
                    FIT_FLAG = self.estimator_control_dict[i].fit(train_rows_control, train_outcome_control, tune = True)
                  
                error_no_tmp = error_no_tmp + 1
                if error_no_tmp > 2:
                    # error occur request new datasets
                    raise Exception('Too many errors occur in internal estimator.')

            # N^l_2 : # of samples in the validation set I^l_2
            total_val_no_treat = val_rows_treat.shape[0]
            total_val_no_control = val_rows_control.shape[0]

            # do conformal prediction, get all CIs
            if self.conformal_mode == "SCQR_RF":
                est_mean, intv_info, icp, cal_scores, val_est_res,intv_both = SCQR_short (self, i, 
                    [train_rows_treat, val_rows_treat, train_outcome_treat, val_labels_treat],
                    [train_rows_control, val_rows_control, train_outcome_control, val_labels_control])
            else:
                est_mean, intv_info, icp, cal_scores, val_est_res  = SCR (self, i, 
                    [train_rows_treat, val_rows_treat, train_outcome_treat, val_labels_treat],
                    [train_rows_control, val_rows_control, train_outcome_control, val_labels_control])
            
            intv, intv_len, intv_split = intv_info
            icp_treat, icp_control = icp
            cal_scores_treat, cal_scores_control = cal_scores
            
            val_est_treat_treat, val_est_treat_control, \
                val_est_control_treat, val_est_control_control = val_est_res

            # Add values to dict
            intv_dict[i] = intv
            intv_split_dict[i] = intv_split
            est_mean_dict[i] = est_mean
            total_val_no_treat_dict[i] = total_val_no_treat
            total_val_no_control_dict[i] = total_val_no_control

            val_rows_treat_dict[i] = val_rows_treat
            val_labels_treat_dict[i] = val_labels_treat
            val_rows_control_dict[i] = val_rows_control
            val_labels_control_dict[i] = val_labels_control
            val_est_treat_treat_dict[i] = val_est_treat_treat
            val_est_treat_control_dict[i] = val_est_treat_control
            val_est_control_treat_dict[i] = val_est_control_treat
            val_est_control_control_dict[i] = val_est_control_control
            intv_len_dict[i] = intv_len
            icp_treat_dict[i] = icp_treat
            icp_control_dict[i] = icp_control
            cal_scores_treat_dict[i] = cal_scores_treat
            cal_scores_control_dict[i] = cal_scores_control
            
            intv_both_dict[i] = intv_both
            self.normalization_dict[i]=[[np.min(intv[:,0]), np.max(intv[:,1])],[np.min(intv[:,0]), np.max(intv[:,1])]]      

        # (lambda*)*w + (1-lambda)*S
        obj, intv_measure, homogeneity, obj_real = \
            self.eval_func(num_outcomes, intv_dict, intv_split_dict, est_mean_dict, total_val_no_treat_dict, total_val_no_control_dict,self.normalization_dict)
        
        if self.seed is not None:
            np.random.seed(self.seed)

        self.obj = obj
        self.curr_leaves = 1

        self.root = self.Node(col=-1, value=None, obj=obj, homogeneity=homogeneity, intv_len_dict=intv_len_dict,
                              est_treat_treat_dict=val_est_treat_treat_dict, est_treat_control_dict=val_est_treat_control_dict,
                              est_control_treat_dict=val_est_control_treat_dict, est_control_control_dict=val_est_control_control_dict,
                              conf_pred_treat_dict=icp_treat_dict, conf_pred_control_dict=icp_control_dict,
                              cal_scores_treat_dict=cal_scores_treat_dict, cal_scores_control_dict=cal_scores_control_dict, 
                              node_depth=0,intv_both_dict=intv_both_dict)

        self.root = self.fit_r(num_outcomes, rows_treat, labels_treat, rows_control, labels_control, 
                               curr_depth=0, node=self.root,
                               val_rows_treat_dict=val_rows_treat_dict, val_labels_treat_dict=val_labels_treat_dict,
                               val_rows_control_dict=val_rows_control_dict, val_labels_control_dict=val_labels_control_dict,
                               total_val_no_treat_dict=total_val_no_treat_dict, total_val_no_control_dict=total_val_no_control_dict)

    def fit_r(self, num_outcomes, rows_treat, labels_treat, rows_control, labels_control, curr_depth=0, node=None,
              val_rows_treat_dict=None, val_labels_treat_dict=None, val_rows_control_dict=None, val_labels_control_dict=None,
              total_val_no_treat_dict=None, total_val_no_control_dict=None):
        if rows_treat.shape[0] == 0:
            return node

        if curr_depth > self.tree_depth:
            self.tree_depth = curr_depth

        if self.max_depth == curr_depth:
            # node leaf number
            self.num_leaves += 1
            # add node leaf number to node class
            node.leaf_num = self.num_leaves
            node.leaf = True
            return node
        
        best_gain = 0.0
        best_attribute = None

        best_tb_obj = 0.0
        best_fb_obj = 0.0

        best_tb_intv_len = 0.0
        best_fb_intv_len = 0.0

        best_tb_homo = 0.0
        best_fb_homo = 0.0

        curr_depth += 1

        column_count = rows_treat.shape[1]
        rows = np.concatenate([rows_treat, rows_control])
        
        for col in range(0, column_count):
            # unique values
            unique_vals = np.unique(rows[:, col])

            for value in unique_vals:
                # Keep track of all values per outcome
                tb_intv_dict = {}
                tb_intv_split_dict = {}
                tb_est_mean_dict = {}
                fb_intv_dict = {}
                fb_intv_split_dict = {}
                fb_est_mean_dict = {}

                tb_val_set_treat_dict = {}
                fb_val_set_treat_dict = {}
                tb_val_y_treat_dict = {}
                fb_val_y_treat_dict = {}
                tb_val_set_control_dict = {}
                fb_val_set_control_dict = {}
                tb_val_y_control_dict = {}
                fb_val_y_control_dict = {}

                tb_val_set_est_treat_treat_dict = {}
                fb_val_set_est_treat_treat_dict = {}
                tb_val_set_est_treat_control_dict = {}
                fb_val_set_est_treat_control_dict = {}

                tb_val_set_est_control_treat_dict = {}
                fb_val_set_est_control_treat_dict = {}
                tb_val_set_est_control_control_dict = {}
                fb_val_set_est_control_control_dict = {}

                tb_intv_len_dict = {}
                fb_intv_len_dict = {}
                tb_cal_scores_treat_dict = {}
                fb_cal_scores_treat_dict = {}
                tb_cal_scores_control_dict = {}
                fb_cal_scores_control_dict = {}
                
                tb_intv_both_dict = {}
                fb_intv_both_dict = {}
                
                # keep track if there are enough samples in each split
                enough_samples_max = True
                enough_samples_min = True
            
                for i in range(num_outcomes):
                    # check for enough samples in tb, fb for all outcomes
                    if (not enough_samples_max) or (not enough_samples_min):
                        continue

                    val_rows_treat = val_rows_treat_dict[i]
                    val_labels_treat = val_labels_treat_dict[i]
                    val_rows_control = val_rows_control_dict[i]
                    val_labels_control = val_labels_control_dict[i]

                    # binary treatment splitting, create fb/tb branches
                    (tb_val_set_treat, fb_val_set_treat,
                    tb_val_y_treat, fb_val_y_treat,
                    tb_val_idx_treat, fb_val_idx_treat) = \
                        divide_set(val_rows_treat, val_labels_treat, col, value)
                    (tb_val_set_control, fb_val_set_control,
                    tb_val_y_control, fb_val_y_control,
                    tb_val_idx_control, fb_val_idx_control) = \
                        divide_set(val_rows_control, val_labels_control, col, value)

                    # check for enough samples in tb, fb
                    if tb_val_set_treat.shape[0] < self.min_size or tb_val_set_control.shape[0] < self.min_size or \
                            fb_val_set_treat.shape[0] < self.min_size or fb_val_set_control.shape[0] < self.min_size:
                        enough_samples_max = False
                        continue
                    if tb_val_set_treat.shape[0] == 0 or tb_val_set_control.shape[0] == 0 \
                            or fb_val_set_treat.shape[0] == 0 or fb_val_set_control.shape[0] == 0:
                        enough_samples_min = False
                        continue
                    
                    ######################################################
                    # tb
                    ######################################################

                    # do conformal prediction, get all CIs
                    if self.conformal_mode == "SCQR_RF":
                        tb_est_mean, tb_intv_info, tb_cal_scores, tb_val_set_est_res, tb_intv_both = SCQR_r_short(self, node, i, 
                            tb_val_set_treat, tb_val_set_control,
                            tb_val_idx_treat, tb_val_idx_control)     
                        
                        
                    else:
                        tb_est_mean, tb_intv_info, tb_cal_scores, tb_val_set_est_res = SCR_r (self, node, i, 
                            tb_val_set_treat, tb_val_set_control,
                            tb_val_idx_treat, tb_val_idx_control)
                    
                    tb_intv, tb_intv_len, tb_intv_split = tb_intv_info
                    
                    tb_cal_scores_treat, tb_cal_scores_control = tb_cal_scores
                    tb_val_set_est_treat_treat, tb_val_set_est_treat_control, \
                        tb_val_set_est_control_treat, tb_val_set_est_control_control = tb_val_set_est_res

                    ######################################################
                    # fb
                    ######################################################

                    # do conformal prediction, get all CIs
                    if self.conformal_mode == "SCQR_RF":
                        fb_est_mean, fb_intv_info, fb_cal_scores, fb_val_set_est_res, fb_intv_both = SCQR_r_short (self, node, i, 
                            fb_val_set_treat, fb_val_set_control,
                            fb_val_idx_treat, fb_val_idx_control)
                    else:
                        fb_est_mean, fb_intv_info, fb_cal_scores, fb_val_set_est_res = SCR_r (self, node, i, 
                            fb_val_set_treat, fb_val_set_control,
                            fb_val_idx_treat, fb_val_idx_control)
                    
                    fb_intv, fb_intv_len, fb_intv_split = fb_intv_info
                    fb_cal_scores_treat, fb_cal_scores_control = fb_cal_scores
                    fb_val_set_est_treat_treat, fb_val_set_est_treat_control, \
                        fb_val_set_est_control_treat, fb_val_set_est_control_control = fb_val_set_est_res

                    # Add values to dict
                    fb_intv_both_dict[i] = fb_intv_both
                    tb_intv_both_dict[i] = tb_intv_both
                    
                    tb_intv_dict[i] = tb_intv
                    tb_intv_split_dict[i] = tb_intv_split
                    tb_est_mean_dict[i] = tb_est_mean
                    fb_intv_dict[i] = fb_intv
                    fb_intv_split_dict[i] = fb_intv_split
                    fb_est_mean_dict[i] = fb_est_mean

                    tb_val_set_treat_dict[i] = tb_val_set_treat
                    fb_val_set_treat_dict[i] = fb_val_set_treat
                    tb_val_y_treat_dict[i] = tb_val_y_treat
                    fb_val_y_treat_dict[i] = fb_val_y_treat
                    tb_val_set_control_dict[i] = tb_val_set_control
                    fb_val_set_control_dict[i] = fb_val_set_control
                    tb_val_y_control_dict[i] = tb_val_y_control
                    fb_val_y_control_dict[i] = fb_val_y_control

                    tb_val_set_est_treat_treat_dict[i] = tb_val_set_est_treat_treat
                    fb_val_set_est_treat_treat_dict[i] = fb_val_set_est_treat_treat
                    tb_val_set_est_treat_control_dict[i] = tb_val_set_est_treat_control
                    fb_val_set_est_treat_control_dict[i] = fb_val_set_est_treat_control

                    tb_val_set_est_control_treat_dict[i] = tb_val_set_est_control_treat
                    fb_val_set_est_control_treat_dict[i] = fb_val_set_est_control_treat
                    tb_val_set_est_control_control_dict[i] = tb_val_set_est_control_control
                    fb_val_set_est_control_control_dict[i] = fb_val_set_est_control_control

                    tb_intv_len_dict[i] = tb_intv_len
                    fb_intv_len_dict[i] = fb_intv_len
                    tb_cal_scores_treat_dict[i] = tb_cal_scores_treat
                    fb_cal_scores_treat_dict[i] = fb_cal_scores_treat
                    tb_cal_scores_control_dict[i] = tb_cal_scores_control
                    fb_cal_scores_control_dict[i] = fb_cal_scores_control
                    
                # check for enough samples in tb, fb in outer for loop
                if (not enough_samples_max) or (not enough_samples_min):
                    continue
                
                # (lambda*)*w + (1-lambda)*S
                tb_obj, tb_intv_measure, tb_homogeneity, tb_obj_real = \
                    self.eval_func(num_outcomes, tb_intv_dict, tb_intv_split_dict, tb_est_mean_dict, total_val_no_treat_dict, total_val_no_control_dict,self.normalization_dict)
                
                fb_obj, fb_intv_measure, fb_homogeneity, fb_obj_real = \
                    self.eval_func(num_outcomes, fb_intv_dict, fb_intv_split_dict, fb_est_mean_dict, total_val_no_treat_dict, total_val_no_control_dict,self.normalization_dict)
               
                gain = node.obj - tb_obj - fb_obj

                if gain > best_gain:
                    best_gain = gain
                    best_attribute = [col, value]
                    best_tb_obj, best_fb_obj = tb_obj, fb_obj
                    best_tb_obj_real, best_fb_obj_real = tb_obj_real, fb_obj_real
                    best_tb_homo, best_fb_homo = tb_homogeneity, fb_homogeneity
                    best_tb_intv_len, best_fb_intv_len = tb_intv_len_dict, fb_intv_len_dict
                    best_tb_set_treat, best_fb_set_treat = tb_val_set_treat_dict, fb_val_set_treat_dict
                    best_tb_y_treat, best_fb_y_treat = tb_val_y_treat_dict, fb_val_y_treat_dict
                    best_tb_set_control, best_fb_set_control = tb_val_set_control_dict, fb_val_set_control_dict
                    best_tb_y_control, best_fb_y_control = tb_val_y_control_dict, fb_val_y_control_dict
                    best_tb_val_est_treat_treat, best_fb_val_est_treat_treat = tb_val_set_est_treat_treat_dict, fb_val_set_est_treat_treat_dict
                    best_tb_val_est_treat_control, best_fb_val_est_treat_control = tb_val_set_est_treat_control_dict, fb_val_set_est_treat_control_dict
                    best_tb_val_est_control_treat, best_fb_val_est_control_treat = tb_val_set_est_control_treat_dict, fb_val_set_est_control_treat_dict
                    best_tb_val_est_control_control, best_fb_val_est_control_control = tb_val_set_est_control_control_dict, fb_val_set_est_control_control_dict
                    best_tb_cal_scores_treat, best_fb_cal_scores_treat = tb_cal_scores_treat_dict, fb_cal_scores_treat_dict
                    best_tb_cal_scores_control, best_fb_cal_scores_control = tb_cal_scores_control_dict, fb_cal_scores_control_dict
                    best_tb_intv_both_dict = tb_intv_both_dict
                    best_fb_intv_both_dict = fb_intv_both_dict
        if best_gain > self.gamma * node.obj:
            node.col = best_attribute[0]
            node.value = best_attribute[1]

            self.curr_leaves = self.curr_leaves + 1

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            tb = self.Node(obj=best_tb_obj_real, homogeneity=best_tb_homo, intv_len_dict=best_tb_intv_len,
                           est_treat_treat_dict=best_tb_val_est_treat_treat, est_treat_control_dict=best_tb_val_est_treat_control,
                           est_control_treat_dict=best_tb_val_est_control_treat,
                           est_control_control_dict=best_tb_val_est_control_control,
                           conf_pred_treat_dict=node.conf_pred_treat_dict,
                           conf_pred_control_dict=node.conf_pred_control_dict,
                           cal_scores_treat_dict=best_tb_cal_scores_treat, cal_scores_control_dict=best_tb_cal_scores_control,
                           node_depth=curr_depth,intv_both_dict=best_tb_intv_both_dict)
            
            fb = self.Node(obj=best_fb_obj_real, homogeneity=best_fb_homo, intv_len_dict=best_fb_intv_len,
                           est_treat_treat_dict=best_fb_val_est_treat_treat, est_treat_control_dict=best_fb_val_est_treat_control,
                           est_control_treat_dict=best_fb_val_est_control_treat,
                           est_control_control_dict=best_fb_val_est_control_control,
                           conf_pred_treat_dict=node.conf_pred_treat_dict,
                           conf_pred_control_dict=node.conf_pred_control_dict,
                           cal_scores_treat_dict=best_fb_cal_scores_treat, cal_scores_control_dict=best_fb_cal_scores_control,
                           node_depth=curr_depth,intv_both_dict=best_fb_intv_both_dict)
            
            node.true_branch = self.fit_r(num_outcomes, rows_treat, labels_treat, rows_control, labels_control,
                                          curr_depth=curr_depth, node=tb,
                                          val_rows_treat_dict=best_tb_set_treat, val_labels_treat_dict=best_tb_y_treat,
                                          val_rows_control_dict=best_tb_set_control, val_labels_control_dict=best_tb_y_control,
                                          total_val_no_treat_dict=total_val_no_treat_dict,
                                          total_val_no_control_dict=total_val_no_control_dict)
            node.false_branch = self.fit_r(num_outcomes, rows_treat, labels_treat, rows_control, labels_control,
                                           curr_depth=curr_depth, node=fb,
                                           val_rows_treat_dict=best_fb_set_treat, val_labels_treat_dict=best_fb_y_treat,
                                           val_rows_control_dict=best_fb_set_control, val_labels_control_dict=best_fb_y_control,
                                           total_val_no_treat_dict=total_val_no_treat_dict,
                                           total_val_no_control_dict=total_val_no_control_dict)
            
            return node
        else:
            # node leaf number
            self.num_leaves += 1
            # add node leaf number to node class
            node.leaf_num = self.num_leaves
            node.leaf = True
            return node

    # (lambda*)*w + (1-lambda)*S
    def conf_homo(self, num_outcomes, intv_dict, intv_homo_dict, est_mean_dict, total_val_no_treat_dict, total_val_no_control_dict,normalization_dict):
        homogeneity = []
        intv_measure = []
        
        for i in range(num_outcomes):
            intv = intv_dict[i]
            intv_homo = intv_homo_dict[i]
            est_mean = est_mean_dict[i]
            
            total_val_no_treat = total_val_no_treat_dict[i]
            total_val_no_control = total_val_no_control_dict[i]

            num_samples = intv.shape[0]
            y_lower = intv_homo[:, 0] - est_mean
            y_upper = est_mean - intv_homo[:, 1]

            homogeneity.append((np.sum(y_lower.clip(min=0)) + np.sum(y_upper.clip(min=0))) / \
                        (total_val_no_treat + total_val_no_control))
    
            intv_measure.append(np.sum(intv[:, 1] - intv[:, 0]) / (total_val_no_treat + total_val_no_control))

        # (lambda*)*w + (1-lambda)*S    
        if self.sum ==0:   
            homogeneity_total = self.beta* homogeneity[0] * (1-self.beta) *homogeneity[1]  
            intv_measure_total = self.beta* intv_measure[0] * (1-self.beta) *intv_measure[1]
        if self.sum == 1:
            homogeneity_total = self.beta* homogeneity[0] + (1-self.beta) *homogeneity[1]  
            intv_measure_total = self.beta* intv_measure[0] + (1-self.beta) *intv_measure[1]
        obj = self.weight * intv_measure_total + (1 - self.weight) * homogeneity_total
        obj_real = self.weight * np.mean(intv[:, 1] - intv[:, 0]) + (1 - self.weight) * homogeneity_total
        
        return obj, intv_measure_total, homogeneity_total, obj_real

    def get_TE_CI(self, intv_treat, intv_control):
        intv = np.zeros(np.shape(intv_treat))
        if len(np.shape(intv_treat)) > 1:
            intv[:, 0] = intv_treat[:, 0] - intv_control[:, 1]
            intv[:, 1] = intv_treat[:, 1] - intv_control[:, 0]
        else:
            intv[0] = intv_treat[0] - intv_control[1]
            intv[1] = intv_treat[1] - intv_control[0]

        return intv

    def predict(self, num_outcomes, test_data, test_y1=None, test_y0=None, test_tau=None, root=False):
        
        def classify_r(node, observation, which_outcome, conformal_mode):
            if node.leaf:
                if conformal_mode == "SCQR_RF": 
                    conf_intv_treat = node.conf_pred_treat_dict[which_outcome].predict_given_scores(observation,
                                                                                significance=self.significance,
                                                                                cal_scores=node.cal_scores_treat_dict[which_outcome][1])
                    conf_intv_control = node.conf_pred_control_dict[which_outcome].predict_given_scores(observation,
                                                                                    significance=self.significance,
                                                                                    cal_scores=node.cal_scores_control_dict[which_outcome][1])
                    conf_intv = self.get_TE_CI(conf_intv_treat, conf_intv_control)
                else:
                    conf_intv_treat = node.conf_pred_treat_dict[which_outcome].predict_given_scores(observation,
                                                                                significance=self.significance,
                                                                                cal_scores=node.cal_scores_treat_dict[which_outcome])
                    conf_intv_control = node.conf_pred_control_dict[which_outcome].predict_given_scores(observation,
                                                                                    significance=self.significance,
                                                                                    cal_scores=node.cal_scores_control_dict[which_outcome])

                    conf_intv = self.get_TE_CI(conf_intv_treat, conf_intv_control)
                return node.leaf_num, conf_intv_treat, conf_intv_control, conf_intv
            else:
                v = observation[:, node.col]
                if v >= node.value:
                    branch = node.true_branch
                else:
                    branch = node.false_branch

            return classify_r(branch, observation, which_outcome,conformal_mode)                

        if len(test_data.shape) == 1:
            leaf_results = []
            for j in range(num_outcomes):
                leaf_results.append(classify_r(self.root, test_data, j,self.conformal_mode))
            return leaf_results

        num_test = test_data.shape[0]
        
        predict_list = []
        leaf_results_list = []
        conf_intv_list = []
        CATE_intv_cov_list = []
        pehe_list = []
        num_leaves_list = []
        within_var_list = []
        across_var_list = []
    
        joint_cov_count = 0
        
        joint_ci_list = []
        for j in range(num_outcomes):
            predict = np.zeros(num_test)
            leaf_results = np.zeros(num_test)
            conf_intv = np.zeros([num_test, 2])
            predict_treat = np.zeros(num_test)
            predict_control = np.zeros(num_test)
            conf_intv_treat = np.zeros([num_test, 2])
            conf_intv_control = np.zeros([num_test, 2])
            
            # JointCI
            joint_ci = []
            
            if not root:
                CATE_intv_cov = 0
                pehe = 0
                for i in range(num_test):
                    test_example = test_data[i, :]
                    test_example = test_example.reshape(1, -1)
                    
                    leaf_results[i], conf_intv_treat[i, :], conf_intv_control[i, :], conf_intv[i, :] = \
                        classify_r(self.root, test_example, j, self.conformal_mode)
                    
                    if self.conformal_mode == "SCQR_RF":
                        predict_treat[i] = self.estimator_treat_dict[j].predict(test_example,significance_flag = 'mse')
                        predict_control[i] = self.estimator_control_dict[j].predict(test_example,significance_flag = 'mse')
                    else:
                        predict_treat[i] = self.estimator_treat_dict[j].predict(test_example)
                        predict_control[i] = self.estimator_control_dict[j].predict(test_example)
                    predict[i] = predict_treat[i] - predict_control[i]
                    pehe = pehe + (predict[i] - test_tau[j][i]) ** 2
                    
                    if conf_intv[i, 1] >= test_tau[j][i] > conf_intv[i, 0]:
                        CATE_intv_cov = CATE_intv_cov + 1
                         #joint coverate
                        joint_ci.append(True)
                    else:
                        joint_ci.append(False)
                num_leaves = self.num_leaves
                within_var = get_within_var(self.num_leaves, leaf_results, test_tau[j])
                across_var = get_across_var(self.num_leaves, leaf_results, test_tau[j])
                joint_ci_list.append(joint_ci)

            else:
                CATE_intv_cov = 0
                pehe = 0
                for i in range(num_test):
                    test_example = test_data[i, :]
                    test_example = test_example.reshape(1, -1)
                    leaf_results[i] = 0
                   
                    if self.conformal_mode=="SCQR_RF":
                        conf_intv_treat[i, :] = \
                            self.root.conf_pred_treat_dict[j].predict_given_scores(test_example,
                                                                        significance=self.significance,
                                                                        cal_scores=self.root.cal_scores_treat_dict[j][1])
                        conf_intv_control[i, :] = \
                            self.root.conf_pred_control_dict[j].predict_given_scores(test_example,
                                                                            significance=self.significance,
                                                                            cal_scores=self.root.cal_scores_control_dict[j][1])
                        
                        conf_intv[i, :] = self.get_TE_CI(conf_intv_treat[i, :], conf_intv_control[i, :])
                        predict_treat[i] = self.estimator_treat_dict[j].predict(test_example,significance_flag = 'mse')
                        predict_control[i] = self.estimator_control_dict[j].predict(test_example,significance_flag = 'mse')
                    else:
                        conf_intv_treat[i, :] = \
                            self.root.conf_pred_treat_dict[j].predict_given_scores(test_example,
                                                                        significance=self.significance,
                                                                        cal_scores=self.root.cal_scores_treat_dict[j])
                        conf_intv_control[i, :] = \
                            self.root.conf_pred_control_dict[j].predict_given_scores(test_example,
                                                                            significance=self.significance,
                                                                            cal_scores=self.root.cal_scores_control_dict[j])
                        conf_intv[i, :] = self.get_TE_CI(conf_intv_treat[i, :], conf_intv_control[i, :])
                        predict_treat[i] = self.estimator_treat_dict[j].predict(test_example)
                        predict_control[i] = self.estimator_control_dict[j].predict(test_example)
                    
                    predict[i] = predict_treat[i] - predict_control[i]
                    pehe = pehe + (predict[i] - test_tau[j][i]) ** 2

                    if conf_intv[i, 1] >= test_tau[j][i] > conf_intv[i, 0]:
                        CATE_intv_cov = CATE_intv_cov + 1
                        joint_ci.append(True)
                    else:
                        joint_ci.append(False)
                num_leaves = 1
                within_var = np.var(test_tau[j])
                across_var = 0
                joint_ci_list.append(joint_ci)
            
            # check if joint ci
            CATE_intv_cov = CATE_intv_cov / num_test
            pehe = pehe / num_test
            pehe = np.sqrt(pehe)
           
            if pehe > 1:
                print('Warning: PEHE is greater than 1 \n PEHE = {}'.format(pehe))
            # Add values to list
            predict_list.append(predict)
            leaf_results_list.append(leaf_results)
            conf_intv_list.append(conf_intv)
            CATE_intv_cov_list.append(CATE_intv_cov)
            pehe_list.append(pehe)
            num_leaves_list.append(num_leaves)
            within_var_list.append(within_var)
            across_var_list.append(across_var)
            
        for i in range(num_test):
            if joint_ci_list[0][i]==True and joint_ci_list[1][i] ==True:
                joint_cov_count+=1
                
        joint_cov_count =joint_cov_count/ num_test
        return predict_list, leaf_results_list, conf_intv_list, CATE_intv_cov_list, pehe_list, num_leaves_list, within_var_list, across_var_list, joint_cov_count
