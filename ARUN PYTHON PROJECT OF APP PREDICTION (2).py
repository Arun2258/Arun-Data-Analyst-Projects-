#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('googleplaystore.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe(include='all')


# In[7]:


df.isnull().sum()


# In[8]:


round(df.isnull().sum()/df.shape[0] * 100,2)


# In[9]:


df.shape


# In[10]:


df.dropna(inplace=True)
df.isnull().sum()


# In[11]:


df.shape


# In[12]:


df.head(2)


# In[13]:


df['Size']


# In[14]:


def size_convert(y):
    if 'M' in y:
        x = y[:-1]
        x = float(x)*1000
        return x
    elif 'K' in y:
        x = y[:-1]
        return x
    else:
        return None
    


# In[15]:


df['Size'] = df["Size"].apply(size_convert)


# In[16]:


df.head()


# In[17]:


df['Reviews'] = df["Reviews"].astype(float)


# In[18]:


df.head()


# In[19]:


df['Size'].value_counts()


# In[20]:


df['Size'].isnull().sum()


# In[21]:


df.Size


# In[22]:


df['Size'] = df['Size'].fillna(0.0)


# In[23]:


df['Size']


# In[24]:


df.head(2)


# In[ ]:





# In[25]:


df.head(2)


# In[26]:


df.info()


# In[27]:


df['Reviews']= df['Reviews'].astype(float)


# In[28]:


df.head()


# In[29]:


df['Size'].value_counts()


# In[30]:


df['Size'].isnull().sum()


# In[31]:


df['Size']


# In[32]:


df['Size'] = df['Size'].fillna(0.0)


# In[33]:


df['Size']


# In[34]:


df.head(2)


# In[38]:


df['Installs'] = df['Installs'].str.replace('+','' , regex=True)
df['Installs'] = df['Installs'].str.replace(',','', regex=True)


# In[39]:


df.head(2)


# In[42]:


df['Installs'] = df["Installs"].astype(int)


# In[43]:


df.head()


# In[44]:


df.info()


# In[45]:


df['Reviews']= df['Reviews'].astype(float)


# In[46]:


df[df['Reviews']>df['Installs']].index


# In[47]:


df.drop(df[df['Reviews']>df['Installs']].index,inplace=True)


# In[48]:


df[df['Reviews']>df['Installs']].index


# In[49]:


df['Rating'].min(),df['Rating'].max()


# In[50]:


df['Rating'].mean()


# In[52]:


df[df['Reviews']>df['Installs']].index


# In[62]:


df[df['Reviews']>df['Installs']]


# In[65]:


df["Price"].dtype


# In[68]:


df.dtypes


# In[70]:


df['Price'][df['Type']=='Free'].min()


# In[71]:


df['Price'][df['Type']=='Free'].max()


# In[72]:


df.dtypes


# In[73]:


df[(df['Price']>0) & (df['Type']=='Free')]


# In[74]:


df['Price'].drop_duplicates()


# In[77]:


import matplotlib.pyplot as plt


# In[78]:


plt.show()


# In[79]:


sns.boxplot('Price',data=df)
plt.show()


# In[80]:


sns.boxplot('Reviews',data=df)
plt.show()


# In[81]:


df['Reviews'].max()


# In[82]:


df['Reviews'].mean()


# In[83]:


df['Reviews'].describe()


# In[84]:


sns.displot(df['Rating'])


# In[85]:


sns.displot(df['Size'])


# In[86]:


df[df['Price']>200].size


# In[87]:


df.drop(df[df['Price']>200].index,inplace=True)


# In[88]:


df[df['Price']>200].size


# In[89]:


df.shape


# In[94]:


df['Reviews'].min()


# In[95]:


df['Reviews'].max()


# In[96]:


df.shape


# In[97]:


9338 - 8885


# In[98]:


df.head(2)


# In[99]:


sns.boxplot('Installs',data=df)
plt.show()


# In[100]:


df['Installs'].quantile([0.1,.25,.5,.7,.9,.95,.99])


# In[101]:


df.drop(df[df['Installs']>=100000000.0].index,inplace=True)


# In[102]:


df.shape


# In[103]:


8885-8743


# In[104]:


sns.jointplot(x='Rating',y='Price',data=df)


# In[105]:


sns.jointplot(x='Rating',y='Size',data=df)


# In[106]:


sns.jointplot(x='Rating',y='Reviews',data=df)


# In[107]:


df.columns


# In[108]:


sns.boxplot(x='Rating',y='Content Rating',data=df)
plt.show()


# In[109]:


plt.figure(figsize=(10,10))
sns.boxplot(x='Rating',y='Category',data=df)
plt.show()


# In[110]:


#MACHINE LEARNING


# In[111]:


inp1 = df.copy()


# In[112]:


df['Installs'].hist()


# In[113]:


inp1['Installs']= inp1['Installs'].apply(np.log1p)


# In[114]:


inp1['Installs'].hist()


# In[115]:


inp1.columns


# In[116]:


inp1['Installs']= inp1['Installs'].apply(np.log1p)
inp1['Reviews']= inp1['Reviews'].apply(np.log1p)


# In[117]:


inp1['Reviews'].dtypes


# In[118]:


inp1.head()


# In[119]:


inp1['Content Rating'].drop_duplicates()


# In[122]:


inp1.drop(['App','Last Updated', 'Current Ver','Android Ver'],axis=1,inplace=True)


# In[123]:


inp2 = pd.get_dummies(inp1)


# In[124]:


inp2.shape


# In[125]:


inp2.head()


# In[126]:


x = inp2.drop(['Rating'],axis=1)
y = inp2[['Rating']]


# In[127]:


from sklearn.model_selection import train_test_split


# In[130]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)


# In[131]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[132]:


from sklearn.linear_model import LinearRegression


# In[133]:


linear_reg = LinearRegression()


# In[134]:


linear_reg.fit(x_train,y_train) 


# In[135]:


x_train.shape,y_train.shape


# In[147]:


y = mx+c
y_train = m1x1+m2x2+......m160x160 + e


# In[141]:


y_train = ()


# In[148]:


y_pred = linear_reg.predict(x_test)


# In[149]:


y_pred


# In[150]:


y_test


# In[151]:


from sklearn.metrics import mean_squared_error


# In[152]:


print('MSE =', mean_squared_error(y_test,y_pred))


# In[153]:


print('RMSE =', np.sqrt(mean_squared_error(y_test,y_pred)))


# In[154]:


from sklearn.metrics import r2_score
print("R2 Score:", r2_score(y_test,y_pred) )
print("R2 Score:", r2_score(y_test,y_pred) )


# In[155]:


import joblib
joblib.dump(linear_reg,'linear_model.sav')


# In[156]:


lin1 = joblib.load('linear_model.sav')


# In[ ]:




