
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[9]:

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
def blight_model():
    
    train=pd.read_csv("train.csv",encoding="latin-1",na_values=-1)
    test=pd.read_csv("test.csv",encoding="latin-1",na_values=-1)
    train["state"]=train.state.astype("category")
    train=pd.get_dummies(train,columns=["state"])
    train["agency_name"]=train["agency_name"].astype("category")
    train=pd.get_dummies(train,columns=["agency_name"])
    
    train["violation_code"]=train["violation_code"].astype("category")
    train=pd.get_dummies(train,columns=["violation_code"])
    #train["discount_amount"]=train["discount_amount"].astype("category")
    #train["late_fee"]=train["late_fee"].astype("category")
    #train=pd.get_dummies(train,columns=["discount_amount"])
    #train=pd.get_dummies(train,columns=["late_fee"])
    
    test["state"]=test.state.astype("category")
    test=pd.get_dummies(test,columns=["state"])
    test["agency_name"]=test["agency_name"].astype("category")
    test=pd.get_dummies(test,columns=["agency_name"])
    
    test["violation_code"]=test["violation_code"].astype("category")
    test=pd.get_dummies(test,columns=["violation_code"])
    #test["discount_amount"]=test["discount_amount"].astype("category")
    #test["late_fee"]=test["late_fee"].astype("category")
    #test=pd.get_dummies(test,columns=["discount_amount"])
    #test=pd.get_dummies(test,columns=["late_fee"])
    test.drop(["admin_fee","state_fee","violation_zip_code",'clean_up_cost',"violation_street_number","mailing_address_str_number"],axis=1,inplace=True)
    train.drop(["admin_fee","state_fee","violation_zip_code",'clean_up_cost',"violation_street_number","mailing_address_str_number","payment_amount","balance_due"],axis=1,inplace=True)
    
    
    m=[train[i].name for i in train.columns if (train[i].dtype=="float64" or train[i].dtype=="uint8")]
    f=[test[i].name for i in test.columns if (test[i].dtype=="float64" or test[i].dtype=="uint8")]
    data_train=pd.DataFrame(train[m])
    
    data_test=pd.DataFrame(test[f])
    common=[data_train[i].name for i in data_train.columns if i in data_test.columns ]
    data_train=pd.DataFrame(data_train[common])
    data_test=pd.DataFrame(data_test[common])
    data_train["compliance"]=train.compliance
    result=data_train[data_train.compliance>-1]
    #result.drop(["admin_fee","state_fee","violation_zip_code",'clean_up_cost',"violation_street_number","mailing_address_str_number","payment_amount","balance_due"],axis=1,inplace=True)
    
    #result.drop(['violation_zip_code'],axis=1,inplace=True)
    
    for i in result.columns:
        result=result[result[i]>-1]
    labels=result.compliance
    result.drop("compliance",axis=1,inplace=True)
    x_train=np.asmatrix(result)
    y_train=np.asarray(labels)
    x_test_f=np.asmatrix(data_test)
    #for i in range(x_train.shape[1]):
        #x_train[:,i]=MinMaxScaler().fit_transform(x_train[:,i])
    #k=PCA(n_components=20).fit(x_train)
    #x_train=k.transform(x_train)
    #x_test_f=k.transform(x_test_f)
    x_train_1,x_test,y_train_1,y_test=train_test_split(x_train,y_train)
    model=DecisionTreeClassifier().fit(x_train_1,y_train_1)
    #k=cross_val_score(model,x_train,y_train,cv=3)
    #importance=model.tree_.compute_feature_importances(normalize=False)
    y_pred_1=model.predict_proba(x_test)
    y_pred=model.predict_proba(x_test_f)[:,1]
    #y_pred_=model.predict(x_train)
    #s=f1_score(y_train,y_pred_)
    #y_pred_final=np.max(y_pred,axis=1)
    #fpr,tpr,_=roc_curve(y_train,y_pred)
    final=pd.DataFrame()
    
    final["prob"]=pd.Series(y_pred)
    final.index=test["ticket_id"]
    
    
       
    
    
    
    return final



# In[10]:

blight_model()


# In[ ]:



