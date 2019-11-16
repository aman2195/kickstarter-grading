# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:59:02 2019
@author: varma_Aman
"""
#------------------------
#--Import libraries
#------------------------
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
#------------------------
#--Check data quality
#------------------------
test1=pd.read_excel(r'C:\Users\15145\Downloads\Kickstarter-Grading-Sample.xlsx')
project_pred=pd.read_excel(r"C:\Users\15145\Documents\data_mining\individual project 1\Kickstarter.xlsx")
frames=[test1,project_pred]
combined=pd.concat(frames)
project_pred = project_pred[(project_pred.state=="successful") | (project_pred.state=="failed")]
project_pred.columns
project_pred=project_pred.drop(['launch_to_state_change_days'],axis=1)
project_pred.columns
#Drop irrelevant columns
project_pred = project_pred.dropna(how='any',axis=0) 
null_columns=project_pred.columns[project_pred.isnull().any()]#checking the null values
project_pred[null_columns].isnull().sum()
################################################################
####for combined data set doing the preprocessing###########################################
combined = combined[(combined.state=="successful") | (combined.state=="failed")]
combined.columns
combined=combined.drop(['launch_to_state_change_days'],axis=1)
combined.columns

combined=combined[['state','usd_pledged','goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
#Drop irrelevant columns
combined = combined.dropna(how='any',axis=0) 
#------------------------
#--Data manipulation
#------------------------
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
#project_pred["goal"] = project_pred["goal"]*project_pred["static_usd_rate"] #keeping us currency #Select relevant columns X=project_pred[['goal','disable_communication','country','staff_pick','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
#Select relevant columns
project_pred=project_pred[['usd_pledged','goal','disable_communication','state','country','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
predictor=project_pred[['disable_communication','goal','country','state','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
target=project_pred['usd_pledged']
#Dummify the features with more than 2 categories
predictor = pd.get_dummies(predictor, columns = ['disable_communication','country','state','category','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])

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
combined=combined[['state','usd_pledged','goal','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]


##############################################################
#combined data set contains the combination of kick starter grading sample(testing) and kick starter(trainig) data set 
combined_predictor=combined[['goal','state','disable_communication','country','staff_pick','static_usd_rate','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]
combined_target=combined['usd_pledged']
#Dummify the features with more than 2 categories
combined_predictor = pd.get_dummies(combined, columns = ['disable_communication','state','country','staff_pick','category','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])
randomforest = RandomForestRegressor()
rfe2 = RFE(randomforest,1)#Used feature = 1 to get ranking
my_model3 = rfe2.fit(combined_predictor,combined_target)
rf_regression_ranking = pd.DataFrame(list(zip(predictor.columns,my_model3.ranking_)), columns = ['variable','order'])
rf_regression_ranking.to_csv("rf_regression_ranking.csv")
y=combined_predictor['usd_pledged']
x1=combined_predictor[["create_to_launch_days","blurb_len_clean","goal","launch_to_deadline_days","state_failed","name_len_clean","static_usd_rate","launched_at_weekday_Wednesday","created_at_day_(0, 7]","deadline_hr_(8, 16]","category_Sound","deadline_month_8","created_at_weekday_Monday","deadline_weekday_Saturday","deadline_day_(0, 7]","category_Hardware","deadline_hr_(0, 8]","created_at_day_(7, 14]","category_Wearables","deadline_weekday_Thursday","launched_at_month_7","deadline_month_10","created_at_month_4","created_at_hr_(16, 25]","launched_at_weekday_Thursday","launched_at_weekday_Tuesday","deadline_month_4","created_at_hr_(8, 16]","launched_at_month_1","deadline_month_7","category_Gadgets","launched_at_day_(21, 32]","launched_at_month_5","deadline_day_(7, 14]","launched_at_hr_(16, 25]","created_at_month_6","created_at_month_10","launched_at_day_(14, 21]","category_Flight","created_at_month_7","deadline_day_(14, 21]","launched_at_hr_(8, 16]","state_successful","launched_at_day_(7, 14]","created_at_month_8","deadline_weekday_Friday","created_at_weekday_Thursday","created_at_day_(14, 21]","launched_at_hr_(0, 8]","launched_at_month_2","created_at_weekday_Saturday","launched_at_month_6","launched_at_yr_2016","launched_at_month_9","category_Robots","created_at_weekday_Tuesday","launched_at_weekday_Monday","created_at_yr_2016","launched_at_day_(0, 7]","created_at_month_12","created_at_weekday_Wednesday","launched_at_month_11","launched_at_yr_2014","country_DE","deadline_day_(21, 32]","created_at_yr_2013","created_at_yr_2015","deadline_month_12","created_at_month_2","created_at_day_(21, 32]","launched_at_yr_2015","deadline_month_9","deadline_weekday_Wednesday","created_at_month_9","launched_at_weekday_Sunday","deadline_weekday_Monday","deadline_yr_2015","launched_at_yr_2013","created_at_month_3","deadline_month_11","deadline_weekday_Sunday","created_at_weekday_Sunday","deadline_month_6","deadline_yr_2016","deadline_weekday_Tuesday","deadline_yr_2017","launched_at_yr_2012","created_at_weekday_Friday","created_at_hr_(0, 8]","country_AU","created_at_month_1","deadline_yr_2011","deadline_month_2","created_at_month_5","deadline_yr_2014","category_Software","launched_at_month_3","deadline_yr_2013","deadline_hr_(16, 25]","deadline_month_3","launched_at_month_12","deadline_month_5","created_at_month_11","launched_at_month_10","launched_at_month_8","launched_at_weekday_Friday","country_NL","created_at_yr_2014","country_AT","deadline_month_1","launched_at_weekday_Saturday","country_US","launched_at_month_4","deadline_yr_2012","country_CA","country_IT","country_FR","category_Musical","country_GB","created_at_yr_2010","created_at_yr_2012","category_Web","country_CH","country_ES","launched_at_yr_2017","category_Apps","country_IE","category_Spaces","created_at_yr_2011","country_DK","category_Immersive","country_SG","category_Makerspaces","launched_at_yr_2010","launched_at_yr_2011","country_MX","deadline_yr_2010","country_HK","category_Experimental","country_BE","country_NO","country_SE","category_Festivals","category_Plays","country_NZ","category_Blues","category_Thrillers","category_Places","category_Webseries","launched_at_yr_2009","category_Academic","created_at_yr_2009","created_at_yr_2017","deadline_yr_2009","category_Shorts","country_LU","disable_communication_False","category_Comedy"]]

###################################
#########--------final model-------
##################################
rfc = RandomForestRegressor(random_state=5,max_features =157,max_depth=6,min_samples_leaf=2,min_samples_split=5,n_estimators=17)
X_i = x1.iloc[:,:157]  
X_train=X_i[:14213][:]
X_test=X_i[14214:][:]
Y_train=y[:14213][:]
Y_test=y[14214:][:]
#X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.33, random_state=5)
model3 = rfc.fit(X_train, Y_train)
y_test_pred = model3.predict(X_test)
mse2 = mean_squared_error(Y_test,y_test_pred)
#best mse is 11.20
'''
#-----------------------------
#-------------RFE-------------
#-----------------------------
check=[25,50,75,100,125,157]
randomforest = RandomForestRegressor()
rfe2 = RFE(randomforest,1)#Used feature = 1 to get ranking
my_model3 = rfe2.fit(predictor,target)
rf_regression_ranking = pd.DataFrame(list(zip(predictor.columns,my_model3.ranking_)), columns = ['variable','order'])
rf_regression_ranking.to_csv("rf_regression_ranking.csv")
predictor2=predictor[["create_to_launch_days","blurb_len_clean","goal","launch_to_deadline_days","state_failed","name_len_clean","static_usd_rate","launched_at_weekday_Wednesday","created_at_day_(0, 7]","deadline_hr_(8, 16]","category_Sound","deadline_month_8","created_at_weekday_Monday","deadline_weekday_Saturday","deadline_day_(0, 7]","category_Hardware","deadline_hr_(0, 8]","created_at_day_(7, 14]","category_Wearables","deadline_weekday_Thursday","launched_at_month_7","deadline_month_10","created_at_month_4","created_at_hr_(16, 25]","launched_at_weekday_Thursday","launched_at_weekday_Tuesday","deadline_month_4","created_at_hr_(8, 16]","launched_at_month_1","deadline_month_7","category_Gadgets","launched_at_day_(21, 32]","launched_at_month_5","deadline_day_(7, 14]","launched_at_hr_(16, 25]","created_at_month_6","created_at_month_10","launched_at_day_(14, 21]","category_Flight","created_at_month_7","deadline_day_(14, 21]","launched_at_hr_(8, 16]","state_successful","launched_at_day_(7, 14]","created_at_month_8","deadline_weekday_Friday","created_at_weekday_Thursday","created_at_day_(14, 21]","launched_at_hr_(0, 8]","launched_at_month_2","created_at_weekday_Saturday","launched_at_month_6","launched_at_yr_2016","launched_at_month_9","category_Robots","created_at_weekday_Tuesday","launched_at_weekday_Monday","created_at_yr_2016","launched_at_day_(0, 7]","created_at_month_12","created_at_weekday_Wednesday","launched_at_month_11","launched_at_yr_2014","country_DE","deadline_day_(21, 32]","created_at_yr_2013","created_at_yr_2015","deadline_month_12","created_at_month_2","created_at_day_(21, 32]","launched_at_yr_2015","deadline_month_9","deadline_weekday_Wednesday","created_at_month_9","launched_at_weekday_Sunday","deadline_weekday_Monday","deadline_yr_2015","launched_at_yr_2013","created_at_month_3","deadline_month_11","deadline_weekday_Sunday","created_at_weekday_Sunday","deadline_month_6","deadline_yr_2016","deadline_weekday_Tuesday","deadline_yr_2017","launched_at_yr_2012","created_at_weekday_Friday","created_at_hr_(0, 8]","country_AU","created_at_month_1","deadline_yr_2011","deadline_month_2","created_at_month_5","deadline_yr_2014","category_Software","launched_at_month_3","deadline_yr_2013","deadline_hr_(16, 25]","deadline_month_3","launched_at_month_12","deadline_month_5","created_at_month_11","launched_at_month_10","launched_at_month_8","launched_at_weekday_Friday","country_NL","created_at_yr_2014","country_AT","deadline_month_1","launched_at_weekday_Saturday","country_US","launched_at_month_4","deadline_yr_2012","country_CA","country_IT","country_FR","category_Musical","country_GB","created_at_yr_2010","created_at_yr_2012","category_Web","country_CH","country_ES","launched_at_yr_2017","category_Apps","country_IE","category_Spaces","created_at_yr_2011","country_DK","category_Immersive","country_SG","category_Makerspaces","launched_at_yr_2010","launched_at_yr_2011","country_MX","deadline_yr_2010","country_HK","category_Experimental","country_BE","country_NO","country_SE","category_Festivals","category_Plays","country_NZ","category_Blues","category_Thrillers","category_Places","category_Webseries","launched_at_yr_2009","category_Academic","created_at_yr_2009","created_at_yr_2017","deadline_yr_2009","category_Shorts","country_LU","disable_communication_False","category_Comedy"]]
lg_predictor = [] 
lg_score=[]
for i in check:
    rfc = RandomForestRegressor(random_state=5, max_features =i)
    X_i = x1.iloc[:,:i]  
    X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.33, random_state=5)
    model3 = rfc.fit(X_train, y_train)
    y_test_pred = model3.predict(X_test)
    mse2 = mean_squared_error(y_test,y_test_pred)
    if(min>mse2):
        min=mse2
        lg_predictor.append(i)
        lg_score.append(min)
#mse 12.8
#playing with hyperparameters for the best model and finding the optimum values
for  i in range(1,15):
    rfc = RandomForestRegressor(random_state=5, max_features =157,max_depth=6)
    X_i = x1.iloc[:,:157]  
    X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.33, random_state=5)
    model3 = rfc.fit(X_train, y_train)
    y_test_pred = model3.predict(X_test)
    lg_predictor.append(i)
    lg_score.append(mean_squared_error(y_test,y_test_pred) )
#best depth at 6
for  i in range(1,15):   
    rfc = RandomForestRegressor(random_state=5,max_features =157,max_depth=6,min_samples_leaf=i)
    X_i = x1.iloc[:,:157]  
    X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.33, random_state=5)
    model3 = rfc.fit(X_train, y_train)
    y_test_pred = model3.predict(X_test)
    mse2 = mean_squared_error(y_test,y_test_pred)
    lg_predictor.append(i)
    lg_score.append(mean_squared_error(y_test,y_test_pred) )
#min sample leaf = 2
for  i in range(2,15):   
    rfc = RandomForestRegressor(random_state=5,max_features =157,max_depth=6,min_samples_leaf=2,min_samples_split=5)
    X_i = x1.iloc[:,:157]  
    X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.33, random_state=5)
    model3 = rfc.fit(X_train, y_train)
    y_test_pred = model3.predict(X_test)
    mse2 = mean_squared_error(y_test,y_test_pred)
    lg_predictor.append(i)
    lg_score.append(mean_squared_error(y_test,y_test_pred) )
#min split =5
for  i in range(2,20):   
    rfc = RandomForestRegressor(random_state=5,max_features =157,max_depth=6,min_samples_leaf=2,min_samples_split=5,n_estimators=i)
    X_i = x1.iloc[:,:157]  
    X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.33, random_state=5)
    model3 = rfc.fit(X_train, y_train)
    y_test_pred = model3.predict(X_test)
    mse2 = mean_squared_error(y_test,y_test_pred)
    lg_predictor.append(i)
    lg_score.append(mean_squared_error(y_test,y_test_pred) )    
    


#-------------LASSO-----------
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_std= scaler.fit_transform(predictor)
y_dummy = label_encoder.fit_transform(target)
alpha=[.0001,.001,.01,.7,.5]
lasso_alpha=["lasso_alpharegression_1.csv","lasso_alpharegression_2.csv","lasso_alpharegression_3.csv","lasso_alpharegression_4.csv","lasso_alpha_5regression.csv"]
j=0
for i in alpha:
    model = Lasso(alpha=i, positive=True)
    model.fit(X_std,y_dummy)
    ranking_i= pd.DataFrame(list(zip(predictor.columns,model.coef_)), columns = ['predictor','coefficient'])
    ranking_i.to_csv(lasso_alpha[j])
    j=j+1
 #.0001  #13.5(mse) 
predictor3=predictor[["goal","static_usd_rate","name_len_clean","create_to_launch_days","launch_to_deadline_days","country_AT","country_AU","country_BE","country_CA","country_CH","country_DE","country_DK","country_ES","country_FR","country_HK","country_IE","country_IT","country_LU","country_MX","country_NL","country_NO","country_NZ","country_SE","country_SG","country_US","state_successful","category_Academic","category_Apps","category_Blues","category_Experimental","category_Festivals","category_Flight","category_Gadgets","category_Hardware","category_Immersive","category_Makerspaces","category_Musical","category_Places","category_Plays","category_Robots","category_Shorts","category_Software","category_Sound","category_Spaces","category_Thrillers","category_Wearables","category_Web","category_Webseries","deadline_weekday_Friday","deadline_weekday_Monday","deadline_weekday_Saturday","deadline_weekday_Sunday","deadline_weekday_Thursday","deadline_weekday_Tuesday","deadline_weekday_Wednesday","created_at_weekday_Friday","created_at_weekday_Monday","created_at_weekday_Saturday","created_at_weekday_Sunday","created_at_weekday_Thursday","created_at_weekday_Tuesday","created_at_weekday_Wednesday","launched_at_weekday_Friday","launched_at_weekday_Monday","launched_at_weekday_Sunday","launched_at_weekday_Thursday","launched_at_weekday_Tuesday","launched_at_weekday_Wednesday","deadline_month_1","deadline_month_3","deadline_month_4","deadline_month_5","deadline_month_6","deadline_month_7","deadline_month_8","deadline_month_9","deadline_month_10","deadline_month_11","deadline_month_12","deadline_day_(0, 7]","deadline_day_(7, 14]","deadline_day_(14, 21]","deadline_day_(21, 32]","deadline_yr_2009","deadline_yr_2011","deadline_yr_2012","deadline_yr_2013","deadline_yr_2014","deadline_yr_2015","deadline_yr_2016","deadline_yr_2017","deadline_hr_(0, 8]","deadline_hr_(16, 25]","created_at_month_1","created_at_month_2","created_at_month_3","created_at_month_4","created_at_month_5","created_at_month_6","created_at_month_7","created_at_month_8","created_at_month_10","created_at_month_11","created_at_month_12","created_at_day_(0, 7]","created_at_day_(7, 14]","created_at_day_(14, 21]","created_at_day_(21, 32]","created_at_yr_2009","created_at_yr_2010","created_at_yr_2011","created_at_yr_2012","created_at_yr_2013","created_at_yr_2014","created_at_yr_2015","created_at_yr_2016","created_at_hr_(0, 8]","created_at_hr_(8, 16]","launched_at_month_1","launched_at_month_2","launched_at_month_3","launched_at_month_4","launched_at_month_5","launched_at_month_6","launched_at_month_8","launched_at_month_9","launched_at_month_10","launched_at_month_11","launched_at_month_12","launched_at_day_(0, 7]","launched_at_day_(7, 14]","launched_at_day_(14, 21]","launched_at_day_(21, 32]","launched_at_yr_2010","launched_at_yr_2011","launched_at_yr_2012","launched_at_yr_2013","launched_at_yr_2014","launched_at_yr_2015","launched_at_yr_2016","launched_at_yr_2017","launched_at_hr_(0, 8]","launched_at_hr_(8, 16]"]]  
predictor4=predictor[["goal","static_usd_rate","name_len_clean","create_to_launch_days","launch_to_deadline_days","country_AT","country_AU","country_BE","country_CA","country_CH","country_DE","country_DK","country_ES","country_FR","country_HK","country_IE","country_IT","country_LU","country_MX","country_NL","country_NO","country_NZ","country_SE","country_SG","country_US","state_successful","category_Academic","category_Apps","category_Blues","category_Experimental","category_Festivals","category_Flight","category_Gadgets","category_Hardware","category_Immersive","category_Makerspaces","category_Musical","category_Places","category_Plays","category_Robots","category_Shorts","category_Software","category_Sound","category_Spaces","category_Thrillers","category_Wearables","category_Web","category_Webseries","deadline_weekday_Friday","deadline_weekday_Monday","deadline_weekday_Sunday","deadline_weekday_Thursday","deadline_weekday_Tuesday","deadline_weekday_Wednesday","created_at_weekday_Friday","created_at_weekday_Monday","created_at_weekday_Saturday","created_at_weekday_Thursday","created_at_weekday_Tuesday","created_at_weekday_Wednesday","launched_at_weekday_Friday","launched_at_weekday_Monday","launched_at_weekday_Sunday","launched_at_weekday_Thursday","launched_at_weekday_Tuesday","launched_at_weekday_Wednesday","deadline_month_1","deadline_month_3","deadline_month_4","deadline_month_5","deadline_month_6","deadline_month_7","deadline_month_8","deadline_month_9","deadline_month_10","deadline_month_11","deadline_month_12","deadline_day_(0, 7]","deadline_day_(7, 14]","deadline_day_(14, 21]","deadline_day_(21, 32]","deadline_yr_2009","deadline_yr_2011","deadline_yr_2012","deadline_yr_2013","deadline_yr_2014","deadline_yr_2015","deadline_yr_2016","deadline_yr_2017","deadline_hr_(0, 8]","deadline_hr_(16, 25]","created_at_month_1","created_at_month_2","created_at_month_3","created_at_month_4","created_at_month_5","created_at_month_6","created_at_month_7","created_at_month_8","created_at_month_10","created_at_month_11","created_at_month_12","created_at_day_(7, 14]","created_at_day_(14, 21]","created_at_day_(21, 32]","created_at_yr_2009","created_at_yr_2010","created_at_yr_2011","created_at_yr_2012","created_at_yr_2013","created_at_yr_2014","created_at_yr_2015","created_at_yr_2016","created_at_hr_(0, 8]","created_at_hr_(8, 16]","launched_at_month_1","launched_at_month_2","launched_at_month_3","launched_at_month_4","launched_at_month_5","launched_at_month_6","launched_at_month_8","launched_at_month_9","launched_at_month_10","launched_at_month_11","launched_at_month_12","launched_at_day_(0, 7]","launched_at_day_(14, 21]","launched_at_day_(21, 32]","launched_at_yr_2010","launched_at_yr_2011","launched_at_yr_2012","launched_at_yr_2013","launched_at_yr_2014","launched_at_yr_2015","launched_at_yr_2016","launched_at_yr_2017","launched_at_hr_(0, 8]","launched_at_hr_(8, 16]"]]
predictor5=predictor[["goal","static_usd_rate","name_len_clean","create_to_launch_days","launch_to_deadline_days","country_AT","country_AU","country_BE","country_CA","country_CH","country_DE","country_DK","country_FR","country_HK","country_IE","country_LU","country_MX","country_NL","country_NO","country_SE","country_SG","country_US","state_successful","category_Academic","category_Apps","category_Blues","category_Experimental","category_Flight","category_Gadgets","category_Hardware","category_Immersive","category_Makerspaces","category_Musical","category_Places","category_Plays","category_Robots","category_Software","category_Sound","category_Spaces","category_Thrillers","category_Wearables","category_Web","category_Webseries","deadline_weekday_Friday","deadline_weekday_Monday","deadline_weekday_Sunday","deadline_weekday_Thursday","deadline_weekday_Tuesday","deadline_weekday_Wednesday","created_at_weekday_Friday","created_at_weekday_Monday","created_at_weekday_Thursday","created_at_weekday_Tuesday","created_at_weekday_Wednesday","launched_at_weekday_Monday","launched_at_weekday_Sunday","launched_at_weekday_Thursday","launched_at_weekday_Tuesday","launched_at_weekday_Wednesday","deadline_month_1","deadline_month_4","deadline_month_5","deadline_month_6","deadline_month_7","deadline_month_8","deadline_month_9","deadline_month_10","deadline_month_12","deadline_day_(7, 14]","deadline_day_(14, 21]","deadline_day_(21, 32]","deadline_yr_2011","deadline_yr_2012","deadline_yr_2013","deadline_yr_2014","deadline_yr_2015","deadline_yr_2016","deadline_hr_(0, 8]","deadline_hr_(16, 25]","created_at_month_1","created_at_month_3","created_at_month_4","created_at_month_5","created_at_month_6","created_at_month_8","created_at_month_10","created_at_month_11","created_at_month_12","created_at_day_(7, 14]","created_at_day_(14, 21]","created_at_day_(21, 32]","created_at_yr_2010","created_at_yr_2013","created_at_yr_2014","created_at_yr_2015","created_at_yr_2016","created_at_hr_(0, 8]","created_at_hr_(8, 16]","launched_at_month_1","launched_at_month_2","launched_at_month_3","launched_at_month_5","launched_at_month_6","launched_at_month_8","launched_at_month_9","launched_at_month_10","launched_at_month_11","launched_at_month_12","launched_at_day_(0, 7]","launched_at_day_(14, 21]","launched_at_day_(21, 32]","launched_at_yr_2011","launched_at_yr_2012","launched_at_yr_2013","launched_at_yr_2014","launched_at_yr_2016","launched_at_yr_2017","launched_at_hr_(0, 8]","launched_at_hr_(8, 16]"]]
rfc = RandomForestRegressor(random_state=5)
X_i = predictor3
X_i1=predictor4
X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.33, random_state=5)
model3 = rfc.fit(X_train, y_train)
y_test_pred = model3.predict(X_test)
mse2 = mean_squared_error(y_test,y_test_pred)
#alpha .001 mse 15.9
rfc = RandomForestRegressor(random_state=5)
X_i1=predictor4
X_train, X_test, y_train, y_test = train_test_split(X_i1,target, test_size=0.33, random_state=5)
model3 = rfc.fit(X_train, y_train)
y_test_pred = model3.predict(X_test)
mse2 = mean_squared_error(y_test,y_test_pred)

#alpha .7 mse 13.6
rfc = RandomForestRegressor(random_state=5)
X_i1=predictor5
X_train, X_test, y_train, y_test = train_test_split(X_i1,target, test_size=0.33, random_state=5)
model3 = rfc.fit(X_train, y_train)
y_test_pred = model3.predict(X_test)
mse2 = mean_squared_error(y_test,y_test_pred)
###############################################
#-----------------------------
#---------Random Forest-------
#-----------------------------
###random forest selection
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state=0)
model = randomforest.fit(predictor,target)
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(predictor,target)
for feature_list_index in sfm.get_support(indices=True):
    print(predictor.columns[feature_list_index])
random_forest_output = pd.DataFrame(list(zip(predictor.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
random_forest_output.to_csv("Random_Forest_Feature_Select_regression.csv")
#############################################################
#Decision Tree 
from sklearn.tree import DecisionTreeRegressor
decisiontree= DecisionTreeRegressor(max_depth=3)
rfe1 = RFE(decisiontree,1)
my_model2 = rfe1.fit(predictor,target)
regression_ranking = pd.DataFrame(list(zip(predictor.columns,my_model2.ranking_)), columns = ['variable','order'])
regression_ranking.to_csv("regression_ranking.csv")
predictor1=predictor[["goal","state_successful","launched_at_hr_(16, 25]","launched_at_hr_(8, 16]","launched_at_hr_(0, 8]","launched_at_yr_2017","launched_at_yr_2016","launched_at_yr_2015","launched_at_yr_2014","launched_at_yr_2013","country_AT","disable_communication_False","launch_to_deadline_days","create_to_launch_days","name_len_clean","blurb_len_clean","launched_at_yr_2012","static_usd_rate","launched_at_yr_2011","country_AU","launched_at_yr_2010","country_BE","launched_at_yr_2009","country_CA","launched_at_day_(21, 32]","country_CH","launched_at_day_(14, 21]","country_DE","launched_at_day_(7, 14]","country_DK","launched_at_day_(0, 7]","country_ES","launched_at_month_12","country_FR","launched_at_month_11","country_GB","launched_at_month_10","country_HK","launched_at_month_9","country_IE","launched_at_month_8","country_IT","launched_at_month_7","country_LU","launched_at_month_6","country_MX","launched_at_month_5","country_NL","launched_at_month_4","country_NO","launched_at_month_3","country_NZ","launched_at_month_2","country_SE","launched_at_month_1","country_SG","created_at_hr_(16, 25]","country_US","created_at_hr_(8, 16]","state_failed","created_at_hr_(0, 8]","category_Academic","created_at_yr_2017","category_Apps","created_at_yr_2016","category_Blues","created_at_yr_2015","category_Comedy","created_at_yr_2014","category_Experimental","created_at_yr_2013","category_Festivals","created_at_yr_2012","category_Flight","created_at_yr_2011","category_Gadgets","created_at_yr_2010","category_Hardware","created_at_yr_2009","category_Immersive","created_at_day_(21, 32]","category_Makerspaces","created_at_day_(14, 21]","category_Musical","created_at_day_(7, 14]","category_Places","created_at_day_(0, 7]","category_Plays","created_at_month_12","category_Robots","created_at_month_11","category_Shorts","created_at_month_10","category_Software","created_at_month_9","category_Sound","created_at_month_8","category_Spaces","created_at_month_7","category_Thrillers","created_at_month_6","category_Wearables","created_at_month_5","category_Web","created_at_month_4","category_Webseries","created_at_month_3","deadline_weekday_Friday","created_at_month_2","deadline_weekday_Monday","created_at_month_1","deadline_weekday_Saturday","deadline_hr_(16, 25]","deadline_weekday_Sunday","deadline_hr_(8, 16]","deadline_weekday_Thursday","deadline_hr_(0, 8]","deadline_weekday_Tuesday","deadline_yr_2017","deadline_weekday_Wednesday","deadline_yr_2016","created_at_weekday_Friday","deadline_yr_2015","created_at_weekday_Monday","deadline_yr_2014","created_at_weekday_Saturday","deadline_yr_2013","created_at_weekday_Sunday","deadline_yr_2012","created_at_weekday_Thursday","deadline_yr_2011","created_at_weekday_Tuesday","deadline_yr_2010","created_at_weekday_Wednesday","deadline_yr_2009","launched_at_weekday_Friday","deadline_day_(21, 32]","launched_at_weekday_Monday","deadline_day_(14, 21]","launched_at_weekday_Saturday","deadline_day_(7, 14]","launched_at_weekday_Sunday","deadline_day_(0, 7]","launched_at_weekday_Thursday","deadline_month_12","launched_at_weekday_Tuesday","deadline_month_11","launched_at_weekday_Wednesday","deadline_month_10","deadline_month_1","deadline_month_9","deadline_month_2","deadline_month_8","deadline_month_3","deadline_month_7","deadline_month_4","deadline_month_6","deadline_month_5"]]
min=100000000000000000000000
for i in check:
    rfc = DecisionTreeRegressor(random_state=5, max_features =i)
    X_i = predictor1.iloc[:,:i]  
    X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.30, random_state=0)
    model3 = rfc.fit(X_train, y_train)
    y_test_pred = model3.predict(X_test)
    mse2 = mean_squared_error(y_test,y_test_pred)
    if(min>mse2):
        min=mse2
#mse 19.6 billion
y=0
from sklearn.ensemble import RandomForestRegressor
for i in check:
    rfc = RandomForestRegressor(random_state=5, max_features =i)
    X_i = predictor1.iloc[:,:i]  
    X_train, X_test, y_train, y_test = train_test_split(X_i,target, test_size=0.30, random_state=0)
    model3 = rfc.fit(X_train, y_train)
    y_test_pred = model3.predict(X_test)
    mse2 = mean_squared_error(y_test,y_test_pred)
    if(min>mse2):
        min=mse2
        y=i
#mse 13.69 billion
#Random Forest RFE implementation
'''