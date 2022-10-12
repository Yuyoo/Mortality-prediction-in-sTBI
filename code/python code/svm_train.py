from sys import maxunicode
from unicodedata import name
import pandas as pd
import numpy as np
from sklearn.svm import SVC
# import statsmodels.api as sm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost
import pickle

import warnings
warnings.filterwarnings('ignore')

category=['Anticoagulants', 'CTAbnormalBasalCisterns', 'CTShift', 'Contusions', 'DecompressionCraniectomy',
            'EDIntubation', 'ExternalVentricularCSFDrainage', 'GcsEDArrPupils1', 'ICUAdm', 'KeyEmergencyInterventions',
            'PreInjuryASAPSClassification', 'PresEmergencyCare', 'PresEmergencyCareIntubation',
            'PresEmergencyCareVentilation', 'PresTBIRef', 'PresentASDH', 'PresentEDH']

continous=['Age', 'BrainInjuryISS', 'EDArrDBP', 'EDArrSBP', 'EDArrSpO2', 'GcsEDArrMotor', 'GcsEDArrScore',
            'TotalISS', 'UpperExtremitiesISS']

target_cols='Die'

def cal_confidence_interval95(lst):
    mean0=np.mean(lst)
    std0=np.std(lst)
    a=mean0-1.96*std0
    b=mean0+1.96*std0
    return mean0, a, b


class svm_preprocess:
    def __init__(self, sample_dataset, category, continous, target):
        self.category=category
        self.continous=continous
        self.y=target
        self.sample_dataset=sample_dataset
        self.mean_cont=sample_dataset[continous].mean()
        self.std_cont=sample_dataset[continous].std()
        self.std_scaler=StandardScaler().fit(self.sample_dataset[continous])
        self.onehot_enc=OneHotEncoder(handle_unknown='ignore').fit(self.sample_dataset[category])
        
    def z_score(self, dataset):
        con_arr=self.std_scaler.transform(dataset[self.continous])
        con_arr[np.isnan(con_arr)]=0
        return con_arr
    
    def one_hot(self, dataset):
        cat_arr=self.onehot_enc.transform(dataset[self.category]).toarray()
        return cat_arr
    
    def preprocess(self, dataset):
        df=np.concatenate((self.z_score(dataset), self.one_hot(dataset)), axis=1)
        return df

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

    data_ex=pd.read_csv('/data3/syy/TBI/data/cv3/EXTERNAL Validation.csv', index_col=0)
    data_ex.replace([88, 99], [np.nan, np.nan], inplace=True)

    

    cv_tr_auc=[]
    cv_va_auc=[]
    cv_te_auc=[]

    for i in range(10):
        train_data=train_cv[i]
        valid_data=test_cv[i]
        clf=SVC(C=0.4147, kernel='rbf', gamma='auto', probability=True)
        tbi_d=svm_preprocess(train_data, category, continous, 'Die')

        X_train=tbi_d.preprocess(train_data)
        X_valid=tbi_d.preprocess(valid_data)
        X_test=tbi_d.preprocess(data_ex)

        y_train = train_data[target_cols]
        y_valid = valid_data[target_cols]
        y_test=data_ex[target_cols]
        clf.fit(X_train, y_train)
        train_auc=roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
        valid_auc=roc_auc_score(y_valid, clf.predict_proba(X_valid)[:,1])
        test_auc=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

        cv_tr_auc.append(train_auc)
        cv_va_auc.append(valid_auc)
        cv_te_auc.append(test_auc)


    train_m, train_a, train_b=cal_confidence_interval95(cv_tr_auc)
    valid_m, valid_a, valid_b=cal_confidence_interval95(cv_va_auc)
    test_m, test_a, test_b=cal_confidence_interval95(cv_te_auc)

    print("SVM train auc:", train_m, " confidence interval: [{}, {}]".format(train_a,train_b))
    print("SVM valid auc:", valid_m, " confidence interval: [{}, {}]".format(valid_a,valid_b))
    print("SVM test auc:", test_m, " confidence interval: [{}, {}]".format(test_a,test_b))

if __name__ == '__main__':
    main()