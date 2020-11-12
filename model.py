# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle


# %%
df = pd.read_csv('Data\cars_price.csv', index_col=0)

# %% [markdown]
# Since the Make and model have very high Cardinality we'll not be consider for model training.

# %%
def details(df):
  print("Description of training set")
  print("Shape: \t\t\t", df.shape)
  print("#NaNs: \t\t\t", df.isna().sum().sum()) 
  
details(df)


# %%
df.dropna(inplace=True)
details(df)


# %%
df['year']=datetime.datetime.now().year-df['year']


# %%
def description(df):
  for cols in df.columns:
    print('Unique values for ',cols,' : ',len(df[cols].unique()))
description(df)


# %%
categorical = ['make', 'model',  'condition',
       'fuel_type', 'color', 'transmission', 'drive_unit','segment']
numerical = ['priceUSD','year','mileage(kilometers)','volume(cm3)']


# %%
anomaly_dict = {
    'year': 36,
    'mileage(kilometers)': 0.8e7,
    'volume(cm3)': 3700
}


# %%
def outlier_removal(df,dict):
  for key, value in dict.items():
    df = df[df[key] < value]
  return df


# %%
df = outlier_removal(df3,anomaly_dict)
df.dropna(inplace=True)
details(df)
df.info()


# %%
df.shape

# %% [markdown]
# ## Model

# %%
features =[ 'year', 'condition', 'mileage(kilometers)',
       'fuel_type', 'volume(cm3)', 'color', 'transmission', 'drive_unit',
       'segment']
categorical = ['condition',
       'fuel_type', 'color', 'transmission', 'drive_unit','segment']
numerical = ['priceUSD','year','mileage(kilometers)','volume(cm3)']


# %%
def mappings(df,categorical):
  label = LabelEncoder()
  df2= df
  dictionaries = []
  for feature in categorical:
    label.fit(df[feature])
    le_name_mapping = dict(zip(label.classes_, label.transform(label.classes_)))
    dictionaries.append(le_name_mapping)
    # print(le_name_mapping)
  return dictionaries
dictionaries = mappings(df,categorical)


# %%
def encoding(df,categorical):
  label = LabelEncoder()
  df2= df
  for feature in categorical:
    df2[feature] =label.fit_transform(df[feature])
  return df2


# %%
dataset = encoding(df,categorical)


# %%
target = df['priceUSD']
dataset = pd.DataFrame(data=dataset[features],columns=features)


# %%
print(dataset.shape)
dataset.head()


# %%
def normalize(df,features):
  mms = MinMaxScaler(feature_range=(0,10))
  df[features] = mms.fit_transform(df[features])
  return df


# %%
df2 = dataset


# %%
data = normalize(df2,['year','mileage(kilometers)','volume(cm3)'])


# %%
X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.10, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape,  y_test.shape)


# %%
XGB = XGBRegressor(max_depth=3,learning_rate=0.2,n_estimators=500,reg_alpha=0.001,reg_lambda=0.001,n_jobs=-1,min_child_weight=3)
XGB.fit(X_train,y_train)


# %%
print ("Training score:",XGB.score(X_train,y_train),"Test Score:",XGB.score(X_test,y_test))


# %%
y_test_pred = XGB.predict(X_test)


# %%
dictionary = {
    'year':[4.193548387096774],
     'condition':[2],
      'mileage(kilometers)':[0.9600150960015096],
       'fuel_type':[1],
        'volume(cm3)':[0.7692307692307693],
         'color':[0],
          'transmission':[1],
           'drive_unit':[1],
            'segment':[3]
}
test = pd.DataFrame(data=dictionary)
print(test)


# %%
print(XGB.predict(test)[0])


# %%
pickle.dump(XGB, open('model.pkl','wb'))


