from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
def short_list(data,min_amount,column,len_limit=None):
    '''
    From the original data, returns a short list of a 
    selected categorical values
    '''
    temp_data = (data.groupby(column)[["income_adj"]]
                   .agg(amount=("income_adj","count"),
                        mean_income=("income_adj","mean")))
    temp_data["money_ratio"] = temp_data["mean_income"] / temp_data["amount"]
    short_list = temp_data[temp_data.amount>min_amount]
    if len_limit:
        return short_list.sort_values("amount",ascending=False)[:len_limit]
    else:
        return short_list
    

    

def cat_long_form(data,columns):
    '''
    Transform the given columns into longForm
    param: 
    - columns: list of columns
    '''
    for col in columns:
        data[col] = data[col].apply(
                    lambda x: x.split(",")[0:5] if isinstance(x,str) else None)
        data[col].dropna(inplace=True)
        data = data.explode(col)
        data[col] = data[col].apply(
                    lambda x: x.strip() if x != None else None)
    return data

class One_hot_encoder(BaseEstimator,TransformerMixin):
    def __init__(self,columns,short_list,min_amount=10):
        self.min_amount = min_amount
        self.columns = columns
        self.short_list = short_list
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):        
        X.loc[~X[self.columns].isin(self.short_list.index),[self.columns]] = 'OTHER'
        X = pd.get_dummies(X,columns=[self.columns])
        return X

class Label_encoder(BaseEstimator,TransformerMixin):       
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        encoder = LabelEncoder()
        result = encoder.fit_transform(X)
        return result
        

        
class EncodeCategoricals(BaseEstimator,TransformerMixin):    
    def __init__(self,cols,regularization= "expand_mean",
                 target="income_adj",drop_columns=True,drop_target=True):

        self.cols = cols
        self.target = target
        self.drop_columns = drop_columns
        self.drop_target = drop_target
        self.regularization = regularization
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = pd.DataFrame(X,columns=self.cols)
        prior = X[self.target].mean()
        X = X.copy()
        if self.regularization == "expand_mean":           
            for col in self.cols:
                cumsum = X.groupby(col)[self.target].cumsum() - X[self.target]
                cumcnt = X.groupby(col).cumcount()
                X[col+'_mean_'+self.target] = cumsum/cumcnt
            X = X.fillna(prior)
                
            if self.drop_columns:
                X.drop(columns=self.cols,inplace=True) 
                X.drop(columns=["income_adj_mean_income_adj"],inplace=True)
            return X   
        
        
class Dates_enginering(BaseEstimator,TransformerMixin):
    '''
    From date column object, creates day and month features (year is omitted)
    '''
    def __init__(self,date_column,drop_original=True):
        self.date_column = date_column
        self.drop_original = drop_original

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        # From date, create month and day columns
        if self.date_column:
            #assert(self.date_column in X.columns)
            X = pd.DataFrame(X,columns=["date_published"])
            X["day"] = pd.to_datetime(X[self.date_column]).dt.day
            X["month"] = pd.to_datetime(X[self.date_column]).dt.month
            X["week_day"] = pd.to_datetime(X[self.date_column]).dt.dayofweek
            if self.drop_original:
                X.drop(columns=[self.date_column],inplace=True)

        return X
    
class OutlierDetection(BaseEstimator,TransformerMixin):
    def __init__(self,detector):
        self.detector = detector        
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X_ = X.copy()
        outliers = self.detector.fit(X_).predict(X_)
        X_ = np.column_stack((X_,outliers)) 
        return X_
            