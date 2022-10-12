source("./jce_9040_mmc3.R")
library(pROC)

train.data=read.csv('../data/train_data.csv')
valid.data=read.csv('../data/valid_data.csv')
test.data=read.csv('../data/EXTERNAL Validation.csv')

library(Hmisc)
train.data=apply(train.data,2, function(x){impute(x,mean)})
valid.data=apply(valid.data,2, function(x){impute(x,mean)})
test.data=apply(test.data,2, function(x){impute(x,mean)})


# 26
oh.feature=c('Anticoagulants', 'CTAbnormalBasalCisterns', 'CTShift', 'Contusions', 'DecompressionCraniectomy',
'EDIntubation', 'ExternalVentricularCSFDrainage', 'GcsEDArrPupils1', 'ICUAdm', 'KeyEmergencyInterventions',
'PreInjuryASAPSClassification', 'PresEmergencyCare', 'PresEmergencyCareIntubation',
'PresEmergencyCareVentilation', 'PresTBIRef', 'PresentASDH', 'PresentEDH')

ct.feature=c('Age', 'BrainInjuryISS', 'EDArrDBP', 'EDArrSBP', 'EDArrSpO2', 'GcsEDArrMotor', 'GcsEDArrScore',
               'TotalISS', 'UpperExtremitiesISS', 'Die')

# 8
# oh.feature=c('GcsEDArrPupils1', 'InjType', 'PresTBIRef', 'DecompressionCraniectomy')
#
# ct.feature=c('Age', 'BrainInjuryISS', 'GcsEDArrScore', 'GcsEDArrMotor','Die')

# 6
# oh.feature=c('GcsEDArrPupils1', 'InjType', 'PresTBIRef')
#
# ct.feature=c('Age', 'BrainInjuryISS', 'GcsEDArrScore','Die')

library(caret)
train.data=as.data.frame(train.data)
train.data[,"source"]=0
valid.data=as.data.frame(valid.data)
valid.data$source=1
test.data=as.data.frame(test.data)
test.data$source=2

train.data.dummy=train.data[,oh.feature]
train.data.dummy=apply(train.data.dummy,2,as.factor)
dmy <- dummyVars(" ~ .", data = train.data.dummy,fullRank=T)
train.data.dummy.2 <- data.frame(predict(dmy, newdata = train.data.dummy))
head(train.data.dummy.2)
dim(train.data.dummy.2)

valid.data.dummy=valid.data[,oh.feature]
valid.data.dummy=apply(valid.data.dummy,2,as.factor)
dmy <- dummyVars(" ~ .", data = valid.data.dummy,fullRank=T)
valid.data.dummy.2 <- data.frame(predict(dmy, newdata = valid.data.dummy))
head(valid.data.dummy.2)
dim(valid.data.dummy.2)

test.data.dummy=test.data[,oh.feature]
test.data.dummy=apply(test.data.dummy,2,as.factor)
dmy <- dummyVars(" ~ .", data = test.data.dummy,fullRank=T)
test.data.dummy.2 <- data.frame(predict(dmy, newdata = test.data.dummy))
head(test.data.dummy.2)
dim(test.data.dummy.2)

train.data.all=cbind(train.data.dummy.2,train.data[,ct.feature])
valid.data.all=cbind(valid.data.dummy.2,valid.data[,ct.feature])
test.data.all=cbind(test.data.dummy.2,test.data[,ct.feature])

a = apply(train.data.all, 2, as.numeric)
r=rcorr(a,type = 'spearman')
# View(r$r)
# View(r$r[r$r>0.9])
write.csv(r$r,file='./corr.r.csv')

# library('car')

lr=glm(Die~., data=train.data.all, family=binomial(link="logit"))
# vif(lr)

train.res=predict.glm(lr,type="response",newdata=train.data.all)
valid.res=predict.glm(lr,type="response",newdata=valid.data.all)
test.res=predict.glm(lr,type="response",newdata=test.data.all)

roc(train.data.all$Die,train.res) 
roc(valid.data.all$Die,valid.res) 
roc(test.data.all$Die,test.res) 

ci.auc(train.data.all$Die,train.res) 
ci.auc(valid.data.all$Die,valid.res) 
ci.auc(test.data.all$Die,test.res)

val.prob.ci.2(train.res,train.data.all$Die)
val.prob.ci.2(valid.res,valid.data.all$Die)
val.prob.ci.2(test.res,test.data.all$Die)

write.csv(cbind(train.data.all$Die,train.res),file='../rawlr_res/rawlr_54_train.csv', row.names = F)
write.csv(cbind(valid.data.all$Die,valid.res),file='../rawlr_res/rawlr_54_valid.csv', row.names = F)
write.csv(cbind(test.data.all$Die,test.res),file='../rawlr_res/rawlr_54_test.csv', row.names = F)
# 


