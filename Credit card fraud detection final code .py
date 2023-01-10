#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD FRAUD ANALYSIS AND MODELING

# # Importing Necessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
plt.style.use('classic')


# # Loding The Dataset

# In[2]:


credit_dataset=pd.read_csv("C:\\Users\\Manu\\Desktop\\data\\fraudTrain.csv")


# In[3]:


credit_dataset.head()


# In[4]:


credit_dataset.shape


# In[5]:


credit_dataset.info()


# In[6]:


msno.bar(credit_dataset)


# # Duplicates

# In[7]:


credit_dataset.duplicated().sum()


# # Extract Date Time Componet

# In[8]:


credit_dataset['trans_date_trans_time'].dtypes


# In[9]:


credit_dataset['trans_date_trans_time']=pd.to_datetime(credit_dataset['trans_date_trans_time'])
credit_dataset['trans_date_trans_time']


# In[10]:


credit_dataset['year']=credit_dataset['trans_date_trans_time'].dt.year
credit_dataset['month']=credit_dataset['trans_date_trans_time'].dt.month
credit_dataset['day']=credit_dataset['trans_date_trans_time'].dt.day
credit_dataset['hour']=credit_dataset['trans_date_trans_time'].dt.hour
credit_dataset['minute']=credit_dataset['trans_date_trans_time'].dt.minute
credit_dataset['second']=credit_dataset['trans_date_trans_time'].dt.second


# # Calculating Age

# In[11]:


credit_dataset['age'] = (pd.to_datetime('today').year - pd.DatetimeIndex(credit_dataset['dob']).year )


# # Extracting Numbers From Street Column

# In[12]:


credit_dataset.head()


# In[13]:


street_num=[]
for i in credit_dataset['street']:
    i=str(i)
    a=i.split(" ")
    street_num.append(a[0])


# In[14]:


credit_dataset['street_num']=street_num


# In[15]:


credit_dataset.street_num.dtypes


# In[16]:


credit_dataset.head()


# In[17]:


credit_dataset['street_num']=street_num


# In[18]:


print(credit_dataset.street_num.dtypes)


# In[19]:


credit_dataset['street_num']=credit_dataset['street_num'].astype('int64')


# In[20]:


fig = plt.figure(figsize=(10,8))
plt.subplot(3,4,1)
plt.title('cc_num')
sns.boxplot(data = credit_dataset['cc_num'])

plt.subplot(3,4,2)
plt.title('amt')
sns.boxplot(data = credit_dataset['amt'])

plt.subplot(3,4,3)
plt.title('zip')
sns.boxplot(data = credit_dataset['zip'])

plt.subplot(3,4,4)
plt.title('lat')
sns.boxplot(data = credit_dataset['lat'])

plt.subplot(3,4,5)
plt.title('long')
sns.boxplot(data = credit_dataset['long'])

plt.subplot(3,4,6)
plt.title('city_pop')
sns.boxplot(data = credit_dataset['city_pop'])

plt.subplot(3,4,7)
plt.title('unix_time')
sns.boxplot(data = credit_dataset['unix_time'])

plt.subplot(3,4,8)
plt.title('merch_lat')
sns.boxplot(data = credit_dataset['merch_lat'])

plt.subplot(3,4,9)
plt.title('merch_long')
sns.boxplot(data = credit_dataset['merch_long'])

plt.subplot(3,4,10)
plt.title('age')
sns.boxplot(data = credit_dataset['age'])

plt.subplot(3,4,11)
plt.title('street_num')
sns.boxplot(data = credit_dataset['street_num'])


# # Drop Unnecessary Columns

# In[21]:


credit_dataset.corr()


# In[22]:


credit_dataset=credit_dataset.drop(['Unnamed: 0','cc_num','unix_time','street','city','job','zip','merchant','trans_date_trans_time','dob','first','last','trans_num','street_num'], axis=1)


# # Treating Outliers

# In[23]:


credit_dataset.amt.describe()


# In[24]:


Q1=credit_dataset.amt.quantile(0.25)
Q3=credit_dataset.amt.quantile(0.75)
Q1,Q3
IQR=Q3-Q1
IQR
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[25]:


credit_dataset=credit_dataset[(credit_dataset['amt']>=-100) & (credit_dataset['amt']<=194)]


# In[26]:


Q1=credit_dataset.age.quantile(0.25)
Q3=credit_dataset.age.quantile(0.75)
Q1,Q3
IQR=Q3-Q1
IQR
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[27]:


credit_dataset=credit_dataset[(credit_dataset['age']>=0) & (credit_dataset['age']<=98)]


# In[28]:


Q1=credit_dataset.lat.quantile(0.25)
Q3=credit_dataset.lat.quantile(0.75)
Q1,Q3
IQR=Q3-Q1
IQR
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[29]:


credit_dataset=credit_dataset[(credit_dataset['lat']>=26) & (credit_dataset['lat']<=53)]


# In[30]:


Q1=credit_dataset.long.quantile(0.25)
Q3=credit_dataset.long.quantile(0.75)
Q1,Q3
IQR=Q3-Q1
IQR
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[31]:


credit_dataset=credit_dataset[(credit_dataset['long']>=-119) & (credit_dataset['long']<=-55)]


# In[32]:


Q1=credit_dataset.city_pop.quantile(0.25)
Q3=credit_dataset.city_pop.quantile(0.75)
Q1,Q3
IQR=Q3-Q1
IQR
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
lower_limit,upper_limit


# In[33]:


credit_dataset=credit_dataset[(credit_dataset['city_pop']>=0) & (credit_dataset['city_pop']<=4000)]


# # EDA

# # Transaction Amount

# In[34]:


sns.histplot(credit_dataset.amt)


# # Credit Card Holder Age

# In[35]:


sns.distplot(credit_dataset.age)


# # Month Vs Fraud

# In[36]:


ax=sns.histplot(data=credit_dataset, x="month", hue="is_fraud", common_norm=False,stat='percent',multiple='dodge')
ax.set_ylabel('Percentage')
ax.set_xlabel('Month')
plt.xticks(np.arange(1,13,1))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul",'Aug','Sep','Oct','Nov','Dec'])
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
plt.xticks(rotation=90)
plt.show()


# # Category Vs Fraud

# In[37]:


ax=sns.histplot(data=credit_dataset, x="category", hue="is_fraud", stat='percent', multiple='dodge', common_norm=False)
ax.set_ylabel('Percentage')
ax.set_xlabel('category')
plt.xticks(np.arange(1,14,1))
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
plt.xticks(rotation=90)
plt.show()


# # MODEL SELECTION

# # Logistic Regression

# One Hot Encoding

# In[38]:


credit_dataset["gender"]=pd.get_dummies(credit_dataset["gender"],drop_first=True)


# Label Encodeing

# In[39]:


from sklearn.preprocessing import LabelEncoder


# In[40]:


le=LabelEncoder()


# In[41]:


for i in credit_dataset.select_dtypes(include='object'):
    credit_dataset[i]=le.fit_transform(credit_dataset[i])


# In[42]:


credit_dataset.head()


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


scaler=StandardScaler()


# In[45]:


scaler.fit(credit_dataset.drop(columns=['is_fraud']))


# In[46]:


scaled_features=scaler.transform(credit_dataset.drop(columns=['is_fraud'],axis=1))


# In[47]:


x=credit_dataset.drop(['is_fraud'],axis=1)
y=credit_dataset.is_fraud


# In[48]:


df=pd.DataFrame(scaled_features,columns=x.columns)


# In[49]:


df


# In[50]:


from imblearn import over_sampling


# In[51]:


from imblearn.over_sampling import RandomOverSampler


# In[52]:


rs=RandomOverSampler(random_state=10)


# In[53]:


x,y=rs.fit_resample(scaled_features,y)


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[56]:


from sklearn.linear_model import LogisticRegression 


# In[57]:


lr=LogisticRegression()


# In[58]:


lr.fit(x_train,y_train)


# In[59]:


pred=lr.predict(x_test)


# In[60]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[61]:


print(accuracy_score(pred,y_test)*100)


# In[62]:


print(classification_report(pred,y_test))


# In[63]:


from sklearn.model_selection import KFold,cross_val_score


# In[64]:


kf=KFold(n_splits=5)


# In[65]:


cv=cross_val_score(lr,x,y,cv=5)


# In[66]:


cv


# In[67]:


print(np.mean(cv)*100)


# # K Nearest Neighbor

# In[68]:


from sklearn.neighbors import KNeighborsClassifier


# In[69]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[70]:


knn


# In[71]:


knn.fit(x_train,y_train)


# In[76]:


pred1=knn.predict(x_test)


# In[77]:


print(classification_report(pred1,y_test))


# In[78]:


print(accuracy_score(pred1,y_test)*100)


# In[79]:


print(confusion_matrix(pred1,y_test))


# # Random Forest Classifier

# In[80]:


from sklearn.ensemble import RandomForestClassifier


# In[81]:


rf=RandomForestClassifier(n_estimators=150,criterion='entropy')


# In[82]:


rf.fit(x_train,y_train)


# In[83]:


pred2=rf.predict(x_test)


# In[84]:


print(confusion_matrix(pred2,y_test))


# In[85]:


print(classification_report(pred2,y_test))


# In[86]:


print(accuracy_score(pred2,y_test)*100)


# In[ ]:




