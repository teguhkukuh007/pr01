import json

import numpy as np
import pandas as pd

import uuid
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score 
from sklearn.externals import joblib


from pandas.io.json import json_normalize

from sklearn.preprocessing import StandardScaler
#import keras
#from keras.models import Sequential 
#from keras.layers import Dense
class Ml_class():
    savemodel=""
    
    def __init__(self,feature, goal): 
        self.jsonfeature = feature
        self.jsongoal=goal 
        self.initializedata()
        uniqeid=uuid.uuid4()
        self.id=str(uniqeid) 
        
    def setId(self,idunique):
        self.id=idunique
        
    def getId(self,idunique):
        return self.id
        
    def initializedata(self):
        #change jsin to pandas 
        self.feature=pd.DataFrame(self.jsonfeature)
        self.goal=pd.DataFrame(self.jsongoal) 
         
    def modelnaivebayes(self):
        gnb_churn=GaussianNB() 
        gnb_churn.fit(self.feature,self.goal.values)
       
        # Output a pickle file for the model
        joblib.dump(gnb_churn, self.id+'nby.pkl') 
        
        
    def modellogistic(self):
        lr_churn=LogisticRegression(solver="liblinear")
        lr_churn.fit(self.feature,self.goal.values)
        joblib.dump(lr_churn, self.id+'reg.pkl') 
        
    
    def modelnn(self):
        sc = StandardScaler()
        X_train = sc.fit_transform(self.feature)
        X_test = sc.transform(self.goal)
        classifier = Sequential()
        classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = self.feature.shape[1])) 
        classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu')) 
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) 
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        joblib.dump(classifier, self.id+'nn.pkl') 
       
    ##train function

        
    def describe(self):
        print(self.feature.describe())
        
        
    def trainNaivebayes(self):
        model = joblib.load(self.id+'nby.pkl')
        y_pred=model.predict(self.feature)
        self.ypred=y_pred
        self.evaluasi()
        return y_pred
    
    def trainLogistic(self):
        model = joblib.load(self.id+'reg.pkl')
        y_pred=model.predict(self.feature)
        self.ypred=y_pred
        self.evaluasi()
        return y_pred
    
    def trainNn(self):
        model = joblib.load(self.id+'nn.pkl')
        y_pred=model.predict(self.feature)
        self.ypred=y_pred
        self.evaluasi()
        return y_pred
    
    def evaluasi(self):
        self.precision =precision_score(self.goal.values,self.ypred)
        self.recall =recall_score(self.goal.values,self.ypred)
        self.fscore =f1_score(self.goal.values,self.ypred)
       # , self.recall, self.fscore, self.support = precision_recall_fscore_support(self.goal.values, self.ypred,average='micro')
        self.accuracy=accuracy_score(self.goal.values,self.ypred)
        
        
    def getPrecision(self):
        return self.precision
    
    def getRecall(self):
        return self.recall
    
    def getFscore(self):
        return self.fscore
    
    def getAccuracy(self):
        return self.accuracy
