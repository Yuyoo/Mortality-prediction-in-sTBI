import pandas as pd
import numpy as np
from sklearn.svm import SVC
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost
import pickle

import warnings
warnings.filterwarnings('ignore')

feature_cols = ['Age', 'PreInjuryASAPSClassification', 'HeadNeckISS', 'BrainInjuryISS', 'ThoracicSpineISS', 
               'ThoraxChestISS', 'AbdomenPelvicContentsISS', 'TotalISS', 'InjType','CTAbnormalBasalCisterns', 
               'CTShift', 'CTtSAH', 'CTIntracranialLesions', 'PresentEDH', 'PresentASDH', 'Contusions', 
               'PresEmergencyCare', 'PresEmergencyCareIntubation', 'PresEmergencyCareVentilation', 'PresTBIRef', 
               'ICUAdm', 'GcsEDArrScore', 'GcsEDArrEyes', 'GcsEDArrMotor', 'GcsEDArrVerbal', 'GcsEDArrPupils1', 
               'EDIntubation', 'EDArrivedIntubated', 'DecompressionCraniectomy', 'CraniotomyForHaematoma', 
               'ExternalVentricularCSFDrainage', 'ExternalFixationLimb', 'EDArrSpO2', 'EDArrSBP', 'Anticoagulants', 
               'PlateletAggregInhibitors']
target_cols='Die'

def cal_confidence_interval95(lst):
    mean0=np.mean(lst)
    std0=np.std(lst)
    a=mean0-1.96*std0
    b=mean0+1.96*std0
    return mean0, a, b

def main():
    train_cv=[]
    test_cv=[]
    for i in range(10):
        tr=pd.read_csv('../data/cv3/train_ {}.csv'.format(i+1), index_col=0)
        te=pd.read_csv('../data/cv3/test_ {}.csv'.format(i+1), index_col=0)
        tr.replace([88, 99], [np.nan, np.nan], inplace=True)
        te.replace([88, 99], [np.nan, np.nan], inplace=True)
        train_cv.append(tr)
        test_cv.append(te)

    data_ex=pd.read_csv('../data/cv3/EXTERNAL Validation.csv', index_col=0)
    data_ex.replace([88, 99], [np.nan, np.nan], inplace=True)

    params = {
        "eta": 0.005,
        "max_depth": 2, 
        "objective": "binary:logistic",
        "min_child_weight":8.863 ,
        "gamma":2.653,
        "lambda":3.787,
        "alpha":0.7044 ,
        "subsample":0.708,
        "colsample_bytree":0.5868    
    }

    train_auc_lst, valid_auc_lst, test_auc_lst=[], [], []

    for i in range(10):
        X_train=train_cv[i][feature_cols]
        y_train=train_cv[i]['Die']
        X_valid=test_cv[i][feature_cols]
        y_valid=test_cv[i]['Die']   
        xgb_train = xgboost.DMatrix(X_train, label=y_train)
        xgb_valid = xgboost.DMatrix(X_valid, label=y_valid)

        model = xgboost.train(params, 
                            xgb_train, 
                            num_boost_round=2046, 
                            evals = [(xgb_train, "train"), (xgb_valid, "valid")], 
                            verbose_eval=0)
        train_auc=roc_auc_score(y_train.values, model.predict(xgb_train))
        valid_auc=roc_auc_score( y_valid.values, model.predict(xgb_valid))

        test=data_ex
        X_test=test[feature_cols]
        y_test=test['Die']
        xgb_test = xgboost.DMatrix(X_test, label=y_test)
        test_auc=roc_auc_score(y_test.values, model.predict(xgb_test))

        train_auc_lst.append(train_auc)
        valid_auc_lst.append(valid_auc)
        test_auc_lst.append(test_auc)

    train_m, train_a, train_b=cal_confidence_interval95(train_auc_lst)
    valid_m, valid_a, valid_b=cal_confidence_interval95(train_auc_lst)
    test_m, test_a, test_b=cal_confidence_interval95(train_auc_lst)

    print("train auc:", train_m, " confidence interval: [{}, {}]".format(train_a,train_b))
    print("valid auc:", valid_m, " confidence interval: [{}, {}]".format(valid_a,valid_b))
    print("test auc:", test_m, " confidence interval: [{}, {}]".format(test_a,test_b))

if __name__ == '__main__':
    main()