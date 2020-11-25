##importing a few general use case libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pickle 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
data = pd.read_csv("C:/Users/SRIRAM SRIPADA/Desktop/Task/cars_price.csv")

##preprocessing the ordinal  columns in data
def preprocessing_cols(df):
    #creating a copy of dataset in df1 
    df1 = data.copy()
    df1 = df1.drop(['make'],axis=1)
    #model feature is dropped as it contains around 1000 categorical variables and it is unwanted to determine Price of car 
    df1 = df1.drop(['model'],axis=1)

    

    # filling the null values
    #filling the null value of  volume(cm3) with median of volume
    df1['volume(cm3)'].fillna((df1['volume(cm3)'].median()), inplace=True)
    #filling the null value of  drive unit with mode of drive unit (27000) values (observed that front wheel drive is most repeated)
    df1.drive_unit.fillna(value='front-wheel drive',inplace=True)
    #filling the null value of segment with mode of segment (observed that D Segmnet is most repeated) 
    df1.segment.fillna(value='D',inplace=True)

    # let's obtain the counts for each one of the labels in variable 'make'
    # let's capture this in a dictionary that we can use to re-map the labels
    #df_frequency_map = df1.make.value_counts().to_dict()                       
    #df1['make'] = df1.make.map(df_frequency_map)

    #creating new column - 2020 and subtracting the year from it to know how many yrs taken when the car was 1st bought
    df1['Current Year'] = 2020
    df1['no_of_years']=df1['Current Year'] - df1['year']

    df1 = df1.drop(['year'],axis=1)
    df1 = df1.drop(['Current Year'],axis=1)

    df1.rename(columns = {'mileage(kilometers)':'mileage_km', 'volume(cm3)':'volume_cm3'}, inplace = True)

    df1['mileage_km'] = np.log1p(df1['mileage_km'])
    df1['volume_cm3'] = np.log1p(df1['volume_cm3'])


    #giving counts according to the count of colours (Count Encoding) 
    df1['color'] = df1['color'].map({'black':12,'silver':11,'blue':10,'gray':9,'white':8,'green':7,'other':6,'red':5,'burgundy':4,'brown':3,'purple':2,'yellow':1,'orange':0})
    #Here considering 'condition','fuel type','transmission','segment','drive unit' as ORDINAL CATEGORICAL VARIABLES , PERFORMED ORDINAL ENCODING
    #Here the  all the features(color,condition,fuel type,transmission,segment,drive unit) are considered ORDINAL
    df1["condition"] = df1["condition"].map({ 'with mileage':2,'for parts' :1,'with damage': 0})
    df1['fuel_type'] = df1['fuel_type'].map({ 'electrocar':0, 'petrol':1, 'diesel':2 })
    df1['transmission'] = df1['transmission'].map({ 'mechanics' : 0,'auto' : 1 })
    df1['segment'] = df1['segment'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'J':7,'M':8,'S':9})
    df1['drive_unit'] = df1['drive_unit'].map({'front-wheel drive':0,'all-wheel drive':1,'rear drive':2,'part-time four-wheel drive':3})

    return df1


data1 = preprocessing_cols(data)
 

def outlier(df):
    Q1 = df['priceUSD'].quantile(0.10)
    Q2 = df['priceUSD'].quantile(0.85)
    IQR = Q2 - Q1
    LL = Q1-IQR*1.5
    UL = Q2+IQR*1.5

    df  =  df.loc[(df['priceUSD']>LL) & (df['priceUSD']<UL)]

    return df  

data2 = outlier(data1)

X = data2.drop(['priceUSD'],axis = 'columns', inplace = False)
y = np.log1p(data2['priceUSD'])

#Setup train and validation cross validation datasets for model building
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size  = 0.2,random_state=42)


# random forest  regresiion
from sklearn.ensemble import RandomForestRegressor
params = {'max_depth': 20,'max_features': 'auto','min_samples_leaf': 1,'min_samples_split': 15,'n_estimators': 700}
rf = RandomForestRegressor(**params)

rf.fit(X_train,y_train)


# Creating a pickle file for the classifier
#filename = 'carprice_rfr_model.pkl'
#pickle.dump(rf, open(filename, 'wb'))

loaded_model = pickle.load(open('carprice_rfr_model.pkl', 'rb'))
result = loaded_model.score(X_test,y_test)
print(result)

##Acieved  Test accuracy of 87 %


#HyperParameterTuning Gradient boosting regressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import RandomizedSearchCV

# Randomized Search CV
# Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
#max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 5, 10]

#learning_rate=[0.15,0.1,0.05,0.01,0.005,0.001]

#loss=['ls', 'lad', 'huber', 'quantile']

# create random grid
#random_grid = {'n_estimators': n_estimators,
              # 'max_features': max_features,
              # 'max_depth': max_depth,
               #'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf,
               #'learning_rate' : learning_rate,
               #'loss' : loss }


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
#reg_gb = GradientBoostingRegressor(random_state=0)
#model = reg_gb.fit(X_train, y_train)


#gb_random = RandomizedSearchCV(estimator = reg_gb, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


#gb_random.fit(X_train,y_train)


# Creating a pickle file for the classifier
#filename = 'carprice_gbr_model.pkl'
#pickle.dump(gb_random, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test,y_test)
#print(result)






