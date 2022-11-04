#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# 
# The client/business deals with used cars sales.
# 
# The customers in this sector give strong preference to less-aged cars and popular brands with good resale value. This puts a very strong challenge as they only have a very limited range of vehicle options to showcase.

# #### No Pre-Set Standards
# 
# `How does one determine the value of a used car?`
# 
# The Market scenario is filled with a lot of malpractices. There is no defined standards exist to determine the appropriate price for the cars, the values are determined by arbitrary methods.
# 
# The unorganized and unstructured methods are disadvantageous to the both the parties trying to strike a deal. The look and feel can be altered in used cars, but the performance cannot be altered beyond a point.
# 

# #### Revolutionizing the Used Car Industry Through Machine Learning

# **Linear regression**
# Linear regression is a ML model that estimates the relationship between independent variables and a dependent variable using a linear equation (straight line equation) in a multidimensional space.

# **CRISP-ML(Q) process model describes six phases:**
# 
# - Business and Data Understanding
# - Data Preparation (Data Engineering)
# - Model Building (Machine Learning)
# - Model Evaluation and Tunning
# - Deployment
# - Monitoring and Maintenance
# 

# **Objective(s):** Maximize the profits
# 
# **Constraints:** Maximize the customer satisfaction

# **Success Criteria**
# 
# - **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%
# 
# - **ML Success Criteria**: RMSE should be less than 0.15
# 
# - **Economic Success Criteria**: Second/Used cars sales delars would see an increase in revenues by atleast 20%

# # Load the Data and perform EDA and Data Preprocessing

# In[ ]:


# Importing necessary libraries


# In[5]:


import pandas as pd

import seaborn as sb

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import pickle




# In[16]:


cars = pd.read_csv(r"C:/Users/kiran/Downloads/docker_test/Cars.csv")

cars


# In[17]:


# engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
#                        .format(user = "user1",# user
#                                pw = "user1", # passwrd
#                                db = "secondsale")) #database

# cars.to_sql('cars', con = engine, if_exists = 'append', chunksize = 50, index= False)


# #### Read the Table (data) from MySQL database

# In[18]:


# con = connector.connect(host = 'localhost',
#                   port='3306',
#                   user='user1',
#                   password='user1',
#                   database='secondsale',
#                   auth_plugin='mysql_native_password')

# cur = con.cursor()
# con.commit()


# In[19]:


# cur.execute('SELECT * FROM cars')
# df = cur.fetchall()


# In[21]:


# dataset = pd.DataFrame(df)

cars = cars.rename({0 : 'MPG'}, axis = 1)
cars = cars.rename({ 1 : 'Enginetype'}, axis = 1)
cars = cars.rename({2 : 'HP'}, axis = 1)
cars = cars.rename({3 : 'VOL'}, axis = 1)
cars = cars.rename({4 : 'SP'}, axis = 1)
cars = cars.rename({5 : 'WT'}, axis = 1)


# In[22]:


cars.isnull().any()


# In[23]:


#### Descriptive Statistics and Data Distribution
cars.describe()


# In[25]:


print(cars.corr())


# In[26]:


dataplot = sb.heatmap(cars.corr(), annot = True, cmap = "YlGnBu")


# In[27]:


# Seperating input and output variables 

X = cars.iloc[:, 1:6].values
y = cars.iloc[:, 0].values


# In[28]:


y


# In[29]:


X


# In[30]:


# checking unique values
cars["Enginetype"].unique()


# In[31]:


X.shape


# ### Define the steps for pipeline

# In[32]:


ct = ColumnTransformer([("ODC", OrdinalEncoder(), [0])], remainder = "passthrough")
abc=ct.fit(X)

joblib.dump(abc,'ordinalEnc')
final=abc.transform(X)
from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder().fit(X)

ab= ordinal.transform(X)

# In[33]:


columntranform1 = ct.fit(X)  
q =  columntranform1.transform([['hybrid',60,60,120,33]])

# In[34]:


X



# In[36]:


import os
os.getcwd()


# In[37]:


# splitting data into train and test

X_train, X_test, y_train, y_test = train_test_split(final, y, test_size = 0.2, random_state = 0)


# In[38]:


# import library to perform multilinear regression

multilinear = LinearRegression()

multilinear.fit(X_train, y_train)
pickle.dump(multilinear,open('mlr.pkl','wb'))
model=pickle.load(open('mlr.pkl','rb'))
X_train
# In[39]:


# Predicting upon X_test
y_pred = model.predict(X_test)
y_pred

# In[40]:


# checking the Accurarcy by using r2_score
accuracy = r2_score(y_test, y_pred)


# In[41]:


accuracy


# In[42]:


X.shape


# In[43]:


X[0]


# ## Saving the model into pickle file

# In[44]:



# In[45]:

od=joblib.load('ordinalEnc')
yp = model.predict(od.transform([['hybrid',60,60,120,33]]))

yp
# In[46]:


yp


# ### Demonstration: Prediction of Fuel Efficiency using Saved Model

# In[47]:





