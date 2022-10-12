from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost

train_data=pd.read_csv('../data/train_data.csv', index_col=0)
valid_data=pd.read_csv('../data/valid_data.csv', index_col=0)
data_ex=pd.read_csv('../data/EXTERNAL Validation for XGBoost.csv', index_col=0)
data_ex.replace([88, 99], [np.nan, np.nan], inplace=True)
train_data.replace([88, 99], [np.nan, np.nan], inplace=True)
valid_data.replace([88, 99], [np.nan, np.nan], inplace=True)

feature_cols = ['Age', 'PresentASDH', 'Contusions', 'Anticoagulants',
       'PresEmergencyCareIntubation', 'PresEmergencyCareVentilation',
       'PresTBIRef', 'ICUAdm', 'GcsEDArrScore', 'GcsEDArrMotor',
       'GcsEDArrPupils1', 'EDIntubation', 'KeyEmergencyInterventions',
       'DecompressionCraniectomy', 'ExternalVentricularCSFDrainage',
       'EDArrSpO2', 'EDArrSBP', 'EDArrDBP', 'PresentEDH', 'CTShift',
       'PresEmergencyCare', 'BrainInjuryISS',
       'PreInjuryASAPSClassification', 'TotalISS', 'UpperExtremitiesISS',
       'CTAbnormalBasalCisterns']
target_cols='Die'

x=train_data[feature_cols]
y=train_data[target_cols]
def xgb_evaluate(                
                l1,
                l2,
                nGamma,
                minChildWeight,
#                 maxDepth,
                nEstimators,
                subsample,
                colSam
                ):
    
    clf = xgboost.XGBClassifier(
        objective = "binary:logistic",
        learning_rate=0.005,
        reg_alpha= l1,
        reg_lambda= l2,
        min_child_weight= minChildWeight,
        gamma=nGamma,
#         subsample_for_bin= 50000,
        n_estimators= int(nEstimators),
        max_depth= 2,
        subsample= subsample,
        colsample_bytree= colSam,
        verbose =-1
    )
    
    score = cross_val_score(clf,x, y, scoring='roc_auc', cv=10).mean()

    return score


def bayesOpt():
    lgbBO = BayesianOptimization(xgb_evaluate, {
                                              'l1':  (0, 1),
                                                'l2': (1, 10),
                                                'nGamma':(1, 10),
                                                'minChildWeight': (0.1, 10),
                                                'nEstimators':(100, 5000),
                                                'subsample': (0.4, 1),                                                
                                                'colSam': (0.4, 1)
                                                 })


    lgbBO.maximize(init_points=5, n_iter=5)

    print("Final result:", lgbBO.max)

bayesOpt()