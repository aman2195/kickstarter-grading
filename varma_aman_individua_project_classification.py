# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:49:52 2019

@author: 15145
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:59:02 2019

@author: varma_Aman
"""

#------------------------
#--Import libraries
#------------------------
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import pandas as pd
import numpy as np
#------------------------
#--Check data quality
#------------------------
test1=pd.read_excel(r'C:\Users\15145\Downloads\Kickstarter-Grading-Sample.xlsx')
project_pred=pd.read_excel(r"C:\Users\15145\Documents\data_mining\individual project 1\Kickstarter.xlsx")
frames=[test1,project_pred]
combined=pd.concat(frames)#applying all the preprocessing on the combined dataset
project_pred = project_pred[(project_pred.state=="successful") | (project_pred.state=="failed")]
project_pred.columns
project_pred=project_pred.drop(['launch_to_state_change_days'],axis=1)
project_pred.columns

project_pred=project_pred[['state','goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
#Drop irrelevant columns
project_pred = project_pred.dropna(how='any',axis=0) 
#------------------------
#--Data manipulation
#------------------------
##################for combined data set doing the preprocessing################################

#Bucketing
bins_day = [0,7,14,21,32]
project_pred['deadline_day'] = pd.cut(project_pred['deadline_day'], bins_day)
#project_pred[['deadline_day','deadline_day_buck']]
bins_hour = [0,8,16,25]
project_pred['deadline_hr'] = pd.cut(project_pred['deadline_hr'], bins_hour)
project_pred['created_at_day'] = pd.cut(project_pred['created_at_day'], bins_day)
project_pred['created_at_hr'] = pd.cut(project_pred['created_at_hr'], bins_hour)
project_pred['launched_at_day'] = pd.cut(project_pred['launched_at_day'], bins_day)
project_pred['launched_at_hr'] = pd.cut(project_pred['launched_at_hr'], bins_hour)
project_pred["goal"] = project_pred["goal"]*project_pred["static_usd_rate"] #keeping us currency #Select relevant columns X=project_pred[['goal','disable_communication','country','staff_pick','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]

#Select relevant columns
project_pred=project_pred[['state','goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
################################################################
####for combined data set doing the preprocessing###########################################
combined = combined[(combined.state=="successful") | (combined.state=="failed")]
combined.columns
combined=combined.drop(['launch_to_state_change_days'],axis=1)
combined.columns

combined=combined[['state','goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
#Drop irrelevant columns
combined = combined.dropna(how='any',axis=0) 
#------------------------
#--Data manipulation
#------------------------
##################combined data set preprocessing################################

#Bucketing
bins_day = [0,7,14,21,32]
combined['deadline_day'] = pd.cut(combined['deadline_day'], bins_day)
#combined[['deadline_day','deadline_day_buck']]
bins_hour = [0,8,16,25]
combined['deadline_hr'] = pd.cut(combined['deadline_hr'], bins_hour)
combined['created_at_day'] = pd.cut(combined['created_at_day'], bins_day)
combined['created_at_hr'] = pd.cut(combined['created_at_hr'], bins_hour)
combined['launched_at_day'] = pd.cut(combined['launched_at_day'], bins_day)
combined['launched_at_hr'] = pd.cut(combined['launched_at_hr'], bins_hour)
combined["goal"] = combined["goal"]*combined["static_usd_rate"] #keeping us currency #Select relevant columns X=combined[['goal','disable_communication','country','staff_pick','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]

#Select relevant columns
combined=combined[['state','goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]



################################################################
################################################################
#####predictor and target variable from project_pred that is the dataframe that we used for training
predictor=project_pred[['goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
target=project_pred['state']
#Dummify the features with more than 2 categories
predictor = pd.get_dummies(predictor, columns = ['disable_communication','country','staff_pick','category','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])
###############################################################
##############################################################
#combined data set contains the combination of kick starter grading sample(testing) and kick starter(trainig) data set 
combined_predictor=combined[['goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
combined_target=combined['state']
#Dummify the features with more than 2 categories
combined_predictor = pd.get_dummies(combined, columns = ['disable_communication','country','staff_pick','category','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])
y=combined_predictor['state']
x1=combined_predictor[['staff_pick_True','category_Plays','name_len_clean','category_Festivals','category_Musical','category_Experimental','category_Shorts','launched_at_yr_2013','category_Immersive','launched_at_yr_2012','category_Spaces','launched_at_hr_(0, 8]','launched_at_weekday_Tuesday','deadline_yr_2016','category_Sound','country_US','category_Hardware','category_Blues','category_Gadgets','category_Robots','launched_at_yr_2011','category_Wearables','created_at_yr_2016','deadline_yr_2013','created_at_month_5','country_GB']]


#-----------------------------
#---------Random Forest-------
#-----------------------------
###random forest selection

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
y_dummy_train = label_encoder.fit_transform(y)
X_train=x1[:14213][:]
X_test=x1[14214:][:]
Y_train=y_dummy_train[:14213][:]
Y_test=y_dummy_train[14214:][:]
model = randomforest.fit(X_train,Y_train)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.04)
sfm.fit(X_train,Y_train)
for feature_list_index in sfm.get_support(indices=True):
    print(X_train.columns[feature_list_index])
random_forest_output = pd.DataFrame(list(zip(X_train.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
random_forest_output.to_csv("Random_Forest_Feature_Select11.csv")
x_train_rf=X_train[["staff_pick_True","category_Plays","name_len_clean","category_Festivals","category_Musical","category_Experimental","category_Shorts","launched_at_yr_2013","category_Immersive","launched_at_yr_2012","category_Spaces","launched_at_hr_(0, 8]","launched_at_weekday_Tuesday","deadline_yr_2016","category_Sound","country_US","category_Hardware","category_Blues","category_Gadgets","category_Robots","launched_at_yr_2011","category_Wearables","created_at_yr_2016","deadline_yr_2013","created_at_month_5","country_GB"]]
x_test_rf=X_test[["staff_pick_True","category_Plays","name_len_clean","category_Festivals","category_Musical","category_Experimental","category_Shorts","launched_at_yr_2013","category_Immersive","launched_at_yr_2012","category_Spaces","launched_at_hr_(0, 8]","launched_at_weekday_Tuesday","deadline_yr_2016","category_Sound","country_US","category_Hardware","category_Blues","category_Gadgets","category_Robots","launched_at_yr_2011","category_Wearables","created_at_yr_2016","deadline_yr_2013","created_at_month_5","country_GB"]]
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model_rf = randomforest.fit(x_train_rf,Y_train)
prediction=model_rf.predict(x_test_rf)
random_forest_accuracy=accuracy_score(Y_test,prediction)#accuracy 75.9

randomforest = RandomForestClassifier(random_state=0,max_depth=17)
model_rf = randomforest.fit(x_train_rf,Y_train)
prediction=model_rf.predict(x_test_rf)
random_forest_accuracy=accuracy_score(Y_test,prediction)   
random_forest_accuracy
'''
############################
in the code below i have tried different feature selection approaches and ran different model on the feature selected.
the model submitted above is the final as i have got the best accuracy for the above model.
below you will find the following approaches :
    RFE
    Lasso with different values of alpha
    i have ran different models for these feature selection approaches like support vector classifier, logistic regression ,decision tree and random forest.
'''

'''
#-----------------------------
#-------------RFE-------------
#-----------------------------

#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
LR = LogisticRegression()
rfe = RFE(LR,1)             #Used feature = 1 to get ranking
my_model= rfe.fit(predictor,target)
ranking=pd.DataFrame(list(zip(predictor.columns,my_model.ranking_)), columns = ['variable','order'])

#Decision Tree

from sklearn.tree import DecisionTreeClassifier
decisiontree= DecisionTreeClassifier(max_depth=3)
rfe1 = RFE(decisiontree,1)
my_model2 = rfe1.fit(predictor,target)
dt_ranking = pd.DataFrame(list(zip(predictor.columns,my_model2.ranking_)), columns = ['variable','order'])

#Random Forest

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
rfe2 = RFE(randomforest,1)      #Used feature = 1 to get ranking
my_model3 = rfe2.fit(predictor,target)
rf_ranking = pd.DataFrame(list(zip(predictor.columns,my_model3.ranking_)), columns = ['variable','order'])

#-------------LASSO-----------
from sklearn.linear_model import Lasso
#alpha = 0.01
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_std= scaler.fit_transform(predictor)
y_dummy = label_encoder.fit_transform(target)
alpha=[.01,.035,.0005,.7,.5,.00005]
lasso_alpha=["lasso_alpha_1.csv","lasso_alpha_2.csv","lasso_alpha_3.csv","lasso_alpha_4.csv","lasso_alpha_5.csv","lasso_alpha_6.csv"]
j=0
#lasso1 = lasso1.fillna(0)
for i in alpha:
    model = Lasso(alpha=i, positive=True)
    model.fit(X_std,y_dummy)
    ranking_i= pd.DataFrame(list(zip(predictor.columns,model.coef_)), columns = ['predictor','coefficient'])
    ranking_i.to_csv(lasso_alpha[j])
    j=j+1
    

####accuracy= 74.8

#select features
#model building
#alpha=.01 #accuracy=75.411
y=project_pred['state']
x1=predictor[['staff_pick_True','category_Plays','name_len_clean','category_Festivals','category_Musical','category_Experimental','category_Shorts','launched_at_yr_2013','category_Immersive','launched_at_yr_2012','category_Spaces','launched_at_hr_(0, 8]','launched_at_weekday_Tuesday','deadline_yr_2016','category_Sound','country_US','category_Hardware','category_Blues','category_Gadgets','category_Robots','launched_at_yr_2011','category_Wearables','created_at_yr_2016','deadline_yr_2013','created_at_month_5','country_GB']]
from sklearn.linear_model import LogisticRegression
lr3=LogisticRegression()
model3 = lr3.fit(x1,y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=model3, X=x1, y=y, cv=5)
scores
np.average(scores)

# Calculate the accuracy score 75.09
alpha=.00005
x2=predictor[['staff_pick_True','category_Plays','created_at_yr_2016','created_at_yr_2015','deadline_yr_2014','category_Musical','category_Festivals','name_len_clean','launched_at_yr_2013','launched_at_yr_2012','created_at_yr_2014','country_US','category_Experimental','launched_at_yr_2015','deadline_yr_2013','deadline_yr_2015','launched_at_yr_2016','deadline_yr_2010','deadline_yr_2011','deadline_yr_2016','category_Immersive','category_Shorts','category_Gadgets','deadline_month_2','category_Hardware','deadline_month_3','launched_at_hr_(0, 8]','launched_at_weekday_Tuesday','category_Apps','category_Spaces','country_GB','deadline_month_5','deadline_yr_2012','category_Wearables','category_Sound','launched_at_yr_2011','deadline_month_4','category_Robots','created_at_hr_(8, 16]','launched_at_hr_(8, 16]','deadline_month_6','country_CA','deadline_month_1','deadline_hr_(16, 25]','created_at_yr_2013','created_at_month_5','country_DE','created_at_hr_(16, 25]','launched_at_month_8','created_at_hr_(0, 8]','deadline_month_11','launched_at_yr_2014','country_FR','category_Blues','created_at_month_6','country_NL','deadline_yr_2009','created_at_month_11','deadline_month_12','category_Makerspaces','created_at_month_9','created_at_month_10','created_at_month_8','launched_at_yr_2017','launched_at_month_7','deadline_weekday_Tuesday','deadline_weekday_Monday','created_at_month_3','launched_at_weekday_Thursday','created_at_month_7','deadline_weekday_Wednesday','created_at_month_12','country_IE','created_at_weekday_Tuesday','deadline_month_7','launched_at_month_6','country_AU','create_to_launch_days','launched_at_weekday_Monday','created_at_yr_2017','launched_at_weekday_Wednesday','created_at_yr_2012','deadline_month_10','country_CH','country_NZ','launched_at_weekday_Sunday','launched_at_month_11','deadline_weekday_Friday','deadline_hr_(0, 8]','country_HK','launched_at_month_9','created_at_weekday_Monday','created_at_weekday_Wednesday','category_Comedy','deadline_month_8','created_at_month_4','created_at_weekday_Thursday','country_ES','deadline_day_(0, 7]','launched_at_month_5','deadline_weekday_Thursday','country_LU','created_at_weekday_Friday','launched_at_month_10','deadline_day_(14, 21]','launched_at_day_(0, 7]','created_at_yr_2009','created_at_weekday_Sunday','launched_at_weekday_Saturday','country_AT','country_DK','country_SG','country_SE','deadline_weekday_Sunday','launched_at_month_3','created_at_day_(7, 14]','country_NO','country_MX','deadline_day_(7, 14]','launched_at_month_12','created_at_day_(14, 21]','launched_at_day_(14, 21]','launched_at_day_(21, 32]','created_at_month_1','blurb_len_clean','created_at_day_(21, 32]']]
lr4=LogisticRegression()
model4 = lr4.fit(x2,y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=model4, X=x2, y=y, cv=5)
scores
np.average(scores)

#random forest with alpha = .035

x3=predictor[['staff_pick_True','category_Plays','name_len_clean','backers_count','category_Festivals','category_Musical']]
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model5 = randomforest.fit(x3,y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=model4, X=x3, y=y, cv=5)
scores
np.average(scores)

### SVC
#### Sigmoid acc 0.69
from sklearn.svm import SVC
scaler = StandardScaler()
X_std = scaler.fit_transform(x1)
svc = SVC(kernel="sigmoid", random_state=0)
model = svc.fit(X_std, y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=model, X=X_std, y=y, cv=5)
scores
np.average(scores)

## Kernel rbf = acc 0.7519
svc = SVC(kernel="rbf", random_state=0)
model = svc.fit(X_std, y)
scores = cross_val_score(estimator=model, X=X_std, y=y, cv=5)
scores
np.average(scores)

#### Kernel Linear 0.7483

svc = SVC(kernel="linear", random_state=0)
model = svc.fit(X_std, y)
scores = cross_val_score(estimator=model, X=X_std, y=y, cv=5)
scores
np.average(scores)

#kernel poly acc 0.76
X_std_train=x1[:14213][:]
X_std_test=x1[14214:][:]
Y_train=y[:14213][:]
Y_test=y[14214:][:]

svc = SVC(kernel="poly", random_state=0)
model = svc.fit(X_std_train, Y_train)
y_pred=model.predict(X_std_test)
x=accuracy_score(Y_test,y_pred)
#prediction accuracy 73.78
scores = cross_val_score(estimator=model, X=X_std, y=y, cv=5)
scores
np.average(scores)
'''