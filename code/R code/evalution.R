setwd("~/Project/pingan/renji/code")
source("~/Project/pingan/renji/code/jce_9040_mmc3.R")
library(pROC)

train_xgb=read.csv('../svm_res/train_svm_54.csv')
valid_xgb=read.csv('../svm_res/valid_svm_54.csv')
test_xgb=read.csv('../svm_res/test_svm_54.csv')

roc(train_xgb$label,train_xgb$pred) #0.8793
roc(valid_xgb$label,valid_xgb$pred) #0.8572
roc(test_xgb$label,test_xgb$pred) #0.8572

###CI
ci.auc(train_xgb$label,train_xgb$pred)  #95% CI: 0.8396-0.8749 
ci.auc(valid_xgb$label,valid_xgb$pred)  #95% CI: 0.8396-0.8749 
ci.auc(test_xgb$label,test_xgb$pred) 

val.prob.ci.2(train_xgb$pred,train_xgb$label)
val.prob.ci.2(valid_xgb$pred,valid_xgb$label)
val.prob.ci.2(test_xgb$pred,test_xgb$label)