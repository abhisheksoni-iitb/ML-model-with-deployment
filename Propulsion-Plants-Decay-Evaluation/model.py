#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Data\propulsion.csv', index_col=0)
df.head()


# In[3]:


# import pandas_profiling as pp


# In[4]:


# pp.ProfileReport(df)


# In[5]:


def details(df):
  print("Description of training set")
  print("Shape: \t\t\t", df.shape)
  print("#NaNs: \t\t\t", df.isna().sum().sum()) 
  
details(df)


# In[6]:


df.info()


# In[7]:


df.columns


# # Plots

# In[8]:


# for cols in ['year','mileage(kilometers)','volume(cm3)']:
#     fig, ax = plt.subplots()
#     ax.scatter(x = df[cols] ,y = df['priceUSD'])
#     plt.ylabel('priceUSD', fontsize=13)
#     plt.xlabel(str(cols), fontsize=13)
#     plt.show()


# ## Ouliers

# In[9]:


def anomaly_plot(df,anaomaly_cols):
  for cols in anaomaly_cols:
    plt.figure(figsize=(8, 8))
    sns.distplot(df[cols])
anomaly_plot(df,df.columns)


# ## columns to drop

# In[10]:


df.drop(['GT Compressor inlet air temperature (T1) [C]','GT Compressor inlet air pressure (P1) [bar]'],axis =1, inplace=True )
df.head()


# In[11]:


def bxplot(df):
    for cols in df.columns:
        sns.boxplot(y=df[cols])
        plt.show()
bxplot(df)


# # Feature Correlation

# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[13]:


columns = np.full((corrmat.shape[0],), True, dtype=bool)
for i in range(corrmat.shape[0]):
    for j in range(i+1, corrmat.shape[0]):
        if corrmat.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
temp_data = df[selected_columns]
temp_data.shape


# Using a threshold of any value above 0.1, we are left with only 1 feature.
# So we cann't remove any feature.
# We cannot proceed with this method by eliminating correlated features. So, we'll be going to use

# In[14]:


df.shape


# ## Model

# In[15]:


target1 = df['GT Compressor decay state coefficient.']
target2 = df['GT Turbine decay state coefficient.']
dataset = df.drop(['GT Compressor decay state coefficient.','GT Turbine decay state coefficient.'],axis=1)
target1.shape,target2.shape, dataset.shape


# In[16]:


features = ['Lever position (lp) [ ]', 'Ship speed (v) [knots]',
       'Gas Turbine shaft torque (GTT) [kN m]',
       'Gas Turbine rate of revolutions (GTn) [rpm]',
       'Gas Generator rate of revolutions (GGn) [rpm]',
       'Starboard Propeller Torque (Ts) [kN]',
       'Port Propeller Torque (Tp) [kN]',
       'HP Turbine exit temperature (T48) [C]',
       'GT Compressor outlet air temperature (T2) [C]',
       'HP Turbine exit pressure (P48) [bar]',
       'GT Compressor outlet air pressure (P2) [bar]',
       'Gas Turbine exhaust gas pressure (Pexh) [bar]',
       'Turbine Injecton Control (TIC) [%]', 'Fuel flow (mf) [kg/s]']


# In[17]:


print(dataset.shape)
dataset.head()


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


def normalize(df,features):
  mms = MinMaxScaler(feature_range=(0,1))
  df[features] = mms.fit_transform(df[features])
  return df


# In[20]:


data = normalize(dataset,features)
data


# In[21]:


from sklearn.model_selection import train_test_split
import re


# In[22]:


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
dataset.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in dataset.columns.values]


# In[23]:


X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(data,target1,target2, test_size=0.10, random_state=42)
X_train.shape, y1_train.shape,y2_train.shape, X_test.shape,  y1_test.shape,  y2_test.shape


# In[24]:


y2_train


# In[25]:


from xgboost import XGBRegressor


# In[26]:


XGB1 = XGBRegressor(max_depth=3,learning_rate=0.2,n_estimators=500,reg_alpha=0.001,reg_lambda=0.001,n_jobs=-1,min_child_weight=3)
XGB1.fit(X_train,y1_train)


# In[27]:


XGB2 = XGBRegressor(max_depth=3,learning_rate=0.2,n_estimators=500,reg_alpha=0.001,reg_lambda=0.001,n_jobs=-1,min_child_weight=3)
XGB2.fit(X_train,y2_train)


# In[28]:


print ("Training score for GT Compressor :",XGB1.score(X_train,y1_train),"Test Score for GT Compressor :",XGB1.score(X_test,y1_test))
print ("Training score for GT Turbine:",XGB2.score(X_train,y2_train),"Test Score for GT Turbine:",XGB2.score(X_test,y2_test))


# In[29]:


y1_test_pred = XGB1.predict(X_test)
y2_test_pred = XGB2.predict(X_test)
# y_test_pred


# In[32]:


import pickle


# In[34]:


pickle.dump(XGB1, open('model1.pkl','wb'))
pickle.dump(XGB2, open('model2.pkl','wb'))


# In[ ]:




