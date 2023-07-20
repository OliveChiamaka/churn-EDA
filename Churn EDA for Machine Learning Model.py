#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Understanding Numpy

#NumPy stands for "Numerical Python." It is a powerful library in Python that helps us work with numbers and perform
#mathematical operations efficiently. It provides a collection of functions data structures that allow you to perform 
#mathematical operations such as addition, subtraction, multiplication, and division on large arrays or matrices of numbers. 
#and tools that make it easier to handle large sets of data, perform complex calculations, and analyze numerical information.

#Understanding Pandas

#Pandas, on the other hand, stands for "Python Data Analysis Library." It is built on top of numpy and provides higher-level
#data structures and tools specifically designed for data analysis and manipulation
#Imagine you have a table of data, like an Excel spreadsheet, with rows and columns. pandas is like a powerful tool that 
#helps you work with this data in a convenient and efficient way.With pandas, you can easily read data from various sources,
#such as CSV files or databases. pandas provides you with wide range of functions and methods to load, clean, transform,manipulate,
#analyze, and visualize data. It offers features like data alignment, missing data handling, merging and joining datasets, 
#and flexible data indexing.
# It provides functions to identify missing values in your dataset and offers options to either remove those values or
#fill them in with suitable replacements.


# In[14]:


#importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Understanding Seaborn and matplotlib


# In[15]:


# getting my working directory
import os


# In[16]:


cd = os.getcwd()


# In[17]:


print(cd)


# In[18]:


ChurnDataset = pd.read_excel(r"C:\Users\Ezetendu Olive C\Downloads\02 Churn-Dataset.xlsx")


# In[36]:


#views top 5 records of the dataset
ChurnDataset.head()


# In[9]:


# to view the datasets attribute such as shape(the number of shape and colum), the columns and datatypes
ChurnDataset.shape


# In[19]:


ChurnDataset.columns.values


# In[38]:


# to view the descriptive statistics of the numerical variable
ChurnDataset.describe()


# **75% of customers have tenures less than 55 months. Customers tenure is at an average of 32 months. The longest a customer has 
# stayed is 72 months.** 
# 
# **The average monthly charge is 64 USD, while 75% of customers spent 89 USD**

# In[21]:


# to check the number of churners and non-churners
ChurnDataset['Churn'].value_counts()


# In[23]:


# To check the percentage
100*ChurnDataset['Churn'].value_counts() / len(ChurnDataset['Churn'])


# **Data is imbalanced. Ratio of 73:27. Almost 3:1**

# In[24]:


# plotting the churners graph
# The figsize parameter takes a tuple of two values (width, height) as its argument.
# It determines the size of the output image or the space allocated for the plot.

ChurnDataset['Churn'].value_counts().plot(kind= 'barh', figsize =(6,3))
plt.xlabel('Count', labelpad=5)
plt.ylabel('Churners', labelpad=5)
plt.title('Count of Churners')


# To create a hyperlink, use square brackets ([text]) followed by parentheses with the URL ((url)). Example: [OpenAI] (https://openai.com).
# 
# There are over 140 colours and hexcodes supported in plotting. see more '[here](https://www.w3schools.com/colors/colors_hex.asp)'

# In[25]:


# It is typically used to enable more detailed or informative logging or debugging messages.
# It means that the code will produce additional output or messages that provide more information about the execution process.

ChurnDataset.info(verbose = True)


# In[26]:


# to see the beginning and end for missing values

pd.DataFrame(ChurnDataset.isnull())


# In[27]:


# to get the summary, of missing values
 
pd.DataFrame(ChurnDataset.isnull()).sum()


# In[28]:


# Getting the % of missing values. This is important so as to determine what your next step should be.
# such as will you drop the data if missing values is over 40% or 30%

100*pd.DataFrame(ChurnDataset.isnull()).sum() / ChurnDataset.shape[0]


# **Do not be in a haste to drop a column wit missing values especially if you dont have the industry or domain knowledge
# to ascertain if columns are related.**
# 
# **Using the IS_CAR and CAR_TYPE example**
# 
# | Is_car   | No_car   |
# |----------|----------|
# |   Yes    |   Benz   |
# |   No     |          |
# |   Yes    |  Toyota  |
# |   No     |          |
#   
# **Instead of deleting the column with the mising value which is related to the second column, you can just fill it up (find/replace)**
# 
# | Is_car   | No_car   |
# |----------|----------|
# |   Yes    |   Benz   |
# |   No     |  No_car  |
# |   Yes    |  Toyota  |
# |   No     |  No_car  |
# 
# **You can also import sklearn and use the impute method to fill the missing values with the mean, std of the whole value**
# 
# **When dealing with yearly data, we make use of ROLLING WINDOW FILLING where use fill in with the last year, cos you cant calculate for an average.**
# 
# **You can also use the previous values. Handling missing values differs from use case to use case.**

# **DATA CLEANING**

# In[43]:


# create a copy of data for manipulation and processing

churn = ChurnDataset.copy()


# In[73]:


churn.info


# In[80]:


# conversion to numerical datatype as machine model works with numerical and not categorical(total charges)

# errors='coerce' is an optional parameter that specifies how to handle errors during the conversion. In this case, 
#'coerce' is used, which means that if there are any values in the "TotalCharges" column that cannot be converted to numeric,
# those values will be set to NaN (missing value) instead of raising an error.

churn.TotalCharges = pd.to_numeric(churn.TotalCharges, errors = 'coerce')
churn.isnull().sum()


# In[83]:


churn['TotalCharges'].dtype 


# In[81]:


# loc function in pandas to select rows based on a specific condition
#  it selects rows where the 'TotalCharges' column is null (isnull() returns a boolean mask of True for null values)

churn.loc[churn['TotalCharges'].isnull() == True]


# In[ ]:


# if there were missing records, insinificant ones such as 10, we could easily drop/remove the column like this
# churn.dropna(how = 'any', inplace = true)

# if we don't want to remove them but fill them with 0 or the average number, we should do this
# churn.fillna(0). 
#The 0 argument pass means, the missing values should be filled with 0, so it can be any number.


# In[84]:


# Remove the missing record

churn.dropna(how = 'any', inplace = True)


# how='any': This parameter specifies that any row containing at least one missing value should be dropped. Alternatively, 
# you can use how='all' to drop rows only if all values are missing. 
# Inplace=True: This parameter determines whether the operation is applied directly to the DataFrame or if a new DataFrame is returned. When inplace=True, the operation is performed on the DataFrame itself, modifying it in-place. 
# If inplace=False (the default value), a new DataFrame with the dropped rows is returned

# In[86]:


# Dividing the tenures into bins, first get the maximum number

print (churn['tenure'].max())


# In[88]:


# Group the tenure in bins of 12 months

labels =["{0} - {1}".format (i, i + 11) for i in range (1, 72, 12)]
churn['tenure_group'] = pd.cut(churn.tenure, range (1, 80, 12), right = False, labels = labels)


# labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]: This line creates a list called labels using a list comprehension. Each label in the list represents a range of tenure values. 
# The format() function is used to replace the placeholders {'0'} and {'1'} with the appropriate values. 
# The range(1, 72, 12) generates a sequence of numbers starting from 1 and incrementing by 12 up to 72.
# 
# churn['tenure_group'] = pd.cut(churn.tenure, range(1, 80, 12), right=False, labels=labels): This line uses the pd.cut() function from the pandas library to categorize the tenure column into different groups based on the provided bins. 
# The range(1, 80, 12) generates a sequence of numbers representing the bin edges. 
# A left-closed interval means that the starting value is included in the interval, while a right-open interval means that the ending value is excluded from the interval(the ending value will be less than what was given in the range).
# The labels=labels parameter assigns the created labels to the corresponding intervals.

# In[89]:


churn['tenure_group'].value_counts()


# In[91]:


# remove columns not necessary for processing

churn.drop(columns = ['customerID', 'tenure'], axis=1, inplace= True)


# axis=1: This parameter indicates that we want to drop columns. Specifying axis=1 means we want to drop columns along the horizontal axis. If you wanted to drop rows, you would use axis=0.
# inplace=True is used to modify the original DataFrame churn instead of creating a new DataFrame.

# In[143]:


# to ascertain these columns really dropped

churn.head(10)


# **DATA EXPLORATION**
# 
# Plot distribution of individual predictors by churn
# 
# **Univariate Analysis**

# In[94]:


for i, predictor in enumerate(churn.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=churn, x=predictor, hue='Churn')


# This is using a for loop to iterate over each column (predictor) in the DataFrame (churn) after dropping the columns 'Churn', 'TotalCharges', and 'MonthlyCharges'. It then creates a count plot for each column to visualize the distribution of churned and non-churned customers.
# 
# The enumerate() function is used to assign an index (i) to each column.
# 
# plt.figure(i):
# This line creates a new figure for each column to plot the count plot.
# 
# sns.countplot(data=churn, x=predictor, hue='Churn'): This line creates a count plot using Seaborn's countplot() function
# 
# The data parameter specifies the DataFrame to use, which is churn.
# 
# The x parameter specifies the column (predictor) to plot on the x-axis.
# 
# The hue parameter specifies the column ('Churn') to differentiate the counts by different colors representing churned and non-churned customers.
# 
# The enumerate() function is a built-in Python function that allows you to iterate over a sequence (in this case, the columns of the churn) while also keeping track of an index for each element in the sequence.
# 
# i represents the index of the current column. It starts from 0 for the first column and increments by 1 in each iteration.
# predictor represents the current column being iterated.

# In[95]:


# convert target variable 'Churn' into a binary numeric variable (i.e) Yes=1, No=2

churn['Churn']= np.where(churn.Churn == 'Yes',1,0)


# We are using the NumPy where() function to assign a binary value to the 'Churn' column based on a condition.
# 
# np.where(churn.Churn == 'Yes', 1, 0) takes three arguments:
# 
# The first argument (churn.Churn == 'Yes') is the condition. It checks if each element in the 'Churn' column is equal to 'Yes'.
# 
# The second argument (1) is the value to assign when the condition is True. In this case, it assigns the value 1.
# 
# The third argument (0) is the value to assign when the condition is False. Here, it assigns the value 0.
# 
# The result of np.where() is a new array where each element is replaced with 1 if the condition is True, and 0 if the condition is False.

# In[96]:


# ascertaining the Churn col conversion

churn.head()


# In[99]:


#converting categorical variables into numeric

churn_dummies = pd.get_dummies(churn)


# The pd.get_dummies() function in pandas is used to perform one-hot encoding on categorical variables. It takes a DataFrame or Series containing categorical columns as input and returns a new DataFrame with binary (0 and 1) columns representing each unique category.
# 
# One-hot encoding is a technique used in data preprocessing to convert categorical variables into a binary representation that can be used for machine learning algorithms. It is commonly used when working with categorical data that cannot be directly used in models that expect numerical inputs.
# 
# It can be performed using various libraries, such as pandas' get_dummies() function or scikit-learn's OneHotEncoder class.

# In[100]:


# lets see the dummies

churn_dummies.head()


# **Relationship between Monthly charges and Total charges**

# In[101]:


sns.lmplot (data= churn_dummies, x= 'MonthlyCharges', y= 'TotalCharges', fit_reg= False)


# **Insight: TotalCharges increases as MonthlyCharges increases, as expected.**
# 
# sns.lmplot creates a scatter plot with a linear regression line to show the relationship between 2 variables in d dataset.
# 
# fit_reg parameter is used to control whether or not a regression line is plotted on a scatter plot.
# 
# When fit_reg is set to False, it means that no regression line will be fit to the data points, and therefore, no line will be displayed on the scatter plot. This is useful when you only want to visualize the data points without any additional regression analysis.

# In[102]:


# plotting with fit_reg as True

sns.lmplot (data= churn_dummies, x= 'MonthlyCharges', y= 'TotalCharges', fit_reg= True)


# **churn by monthly charges and total charges**

# In[106]:


Mth= sns.kdeplot(churn_dummies.MonthlyCharges[(churn_dummies['Churn'] == 0)],
                color= 'Red', shade= True)
Mth= sns.kdeplot(churn_dummies.MonthlyCharges[(churn_dummies['Churn'] == 1)],
                 ax= Mth, color= 'Blue', shade= True)
Mth.legend(['No Churn', 'Churn'], loc= 'upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly Charges by Churn')


# **Insight:Churn is high when monthly charges are high**

# This is used to plot Kernel Density Estimation (KDE) plots for the MonthlyCharges 
# 
# [(churn_dummies['Churn'] == 0)]: This selects the values from the MonthlyCharges column where the Churn column has a value of 0 (indicating non-churned customers).
# 
# sns.kdeplot() is used to create a KDE plot for the selected MonthlyCharges values. It takes the following parameters:
# 
# data: The data source, which is the selected MonthlyCharges values.
# color: The color of the KDE plot, which is 'Red' in this case.
# shade: A boolean value indicating whether to shade the area under the KDE curve.
# 
# [(churn_dummies['Churn'] == 1)]: This selects the values from the MonthlyCharges column where the Churn column has a value of 1 (indicating churned customers).
# 
# By plotting the KDE plots for MonthlyCharges separately for churned and non-churned customers, this code allows you to compare the distributions and visualize any potential differences or patterns in the MonthlyCharges between the two groups. The shading of the KDE curves helps to emphasize the density of the data.

# In[107]:


Mth= sns.kdeplot(churn_dummies.TotalCharges[(churn_dummies['Churn'] == 0)],
                color= 'Red', shade= True)
Mth= sns.kdeplot(churn_dummies.TotalCharges[(churn_dummies['Churn'] == 1)],
                 ax= Mth, color= 'Blue', shade= True)
Mth.legend(['No Churn', 'Churn'], loc= 'upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Total Charges')
Mth.set_title('Total Charges by Churn')


# **Insight: Churn is high when total charges are low**
# 
# This is quite a suprising one, but on a second thought, customers who have bought a year plan or longer plan duration would stick with the service, not withstanding their reservations, at least till their plan expires.
# 
# But customers who have more flexible plans such as the weekly or monthly can easily rate the service from the week or month activity and make a churn decision. There's nothing to lose on their end.
# 
# This is the reason why a lot of businesses encourage users to take a longer/yearly plan through a pricing strategy called 'tiered pricing' or 'pricing discrimination'. In this strategy, the shorter-term plans have higher daily rates or higher monthly costs, making them less appealing compared to the longer-term plans.
# 
# This pricing strategy aims to encourage customers to choose the longer-term plans by making them more attractive and cost-effective in the long run. It provides an incentive for customers to commit to a higher upfront payment or a longer contract duration, benefiting the company by ensuring customer loyalty and reducing the churn rate.

# **Build a correlation of all predictors with Churn**

# In[108]:


plt.figure(figsize=(20,8))
churn_dummies.corr()['Churn'].sort_values (ascending = False).plot(kind='bar')


# **Insights**: There was high correlation between **churn and month to month contract, no tech support, no online security, short tenure(1-12, first year of subscription)** while there were low correction between churn and **long term contracts, subscriptions without internet services, longer term tenure**

# **Bivariate Analysis**

# In[109]:


new_df1_target0= churn.loc[churn['Churn']==0]
new_df1_target1= churn.loc[churn['Churn']==1]


# This is selecting rows from churn where the value in the 'Churn' column is equal to 1 or 0. 
# 
# It creates a new DataFrame called new_df1_target1 containing only the rows where the 'Churn' column has a value of 1, or a new DF called new_df1_target0 where the 'Churn' column has a value of 0.

# In[128]:


def uniplot(df, col, title, hue = None):
    sns.set_style ('whitegrid')
    sns.set_context ('talk')
    plt.rcParams['axes.labelsize']= 10
    plt.rcParams['axes.titlesize']= 18
    plt.rcParams['axes.titlepad'] = 20
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 4 + 2 * len(temp.unique())
    fig.set_size_inches (width, 5)
    plt.xticks (rotation = 45)
    plt.yscale ('log')
    plt.title(title)
    ax = sns.countplot(data = df, x = col, order = df[col].value_counts().index, hue = hue, palette = 'bright')
    
    plt.show()


# The code above creates a customized countplot with various visual and formatting options.
# 
# The uniplot function is a custom function that generates a countplot using seaborn library to visualize the distribution of a categorical variable in a DF.
# 
# sns.set_style('whitegrid') and sns.set_context('talk') set the style and context for the plot, giving it a white grid background and adjusting the font size.
# 
# plt.rcParams['axes.labelsize'] = 20 and plt.rcParams['axes.titlesize'] = 22 set the label and title font sizes for the plot.
# plt.rcParams['axes.titlepad'] = 30 sets the padding for the plot title.
# 
# temp = pd.Series(data=hue) creates a pandas Series object from the hue parameter. A pandas Series object is a one-dimensional labeled array that can hold data of any type (integer, float, string, etc.). This is done for the purpose of manipulating and analyzing the data within it.
# 
# fig, ax = plt.subplots() creates a figure and an axes object for the plot.
# 
# width = len(df[col].unique()) + 7 + 4 * len(temp.unique()) calculates the width of the plot based on the number of unique values in the col column and the number of unique values in the hue parameter.
# 
# fig.set_size_inches(width, 8) sets the size of the figure.
# 
# plt.xticks(rotation=45) rotates the x-axis labels by 45 degrees for better readability.
# 
# plt.yscale('log') sets the y-axis scale to logarithmic.
# 
# When an axis is scaled to logarithmic,it uses the logarithemetic functions such as base 10 or base 20 logarithemetic to compresse values that span a wide range into a more compact representation. This is useful when dealing with data that has a large range of values or when there is a significant difference in magnitude between the data points. It allows for better visualization and comparison of data that varies over several orders of magnitude.
# 
# plt.title(title) sets the title of the plot.
# 
# ax = sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=hue, palette='bright') creates the countplot using seaborn's countplot function. 
# 
# It specifies df, the column col, the order of the x-axis categories based on the value counts of col, the hue parameter for grouping the data, and the 'bright' color palette.
# 
# plt.show() displays the plot.

# In[129]:


uniplot(new_df1_target1, col = 'Partner', title = 'Distribution of Gender for Churned Customers', hue = 'gender')


# In[130]:


uniplot(new_df1_target1, col = 'PaymentMethod', title = 'Distribution of Payment Method for Churned Customers', hue = 'gender')


# In[131]:


uniplot(new_df1_target1, col = 'Contract', title = 'Distribution of Contract for Churned Customers', hue = 'gender')


# In[132]:


uniplot(new_df1_target1, col = 'SeniorCitizen', title = 'Distribution of Senior Citizen for Churned Customers', hue = 'gender')


# In[138]:


uniplot(new_df1_target1, col = 'Dependents', title = 'Distribution of Dependents for Churned Customers', hue = 'gender')


# In[139]:


uniplot(new_df1_target1, col = 'PhoneService', title = 'Distribution of PhoneService for Churned Customers', hue = 'gender')


# In[140]:


uniplot(new_df1_target1, col = 'InternetService', title = 'Distribution of InternetService for Churned Customers', hue = 'gender')


# In[141]:


uniplot(new_df1_target1, col = 'MultipleLines', title = 'Distribution of MultipleLines for Churned Customers', hue = 'gender')


# In[142]:


uniplot(new_df1_target1, col = 'OnlineSecurity', title = 'Distribution of OnlineSecurity for Churned Customers', hue = 'gender')


# In[144]:


uniplot(new_df1_target1, col = 'DeviceProtection', title = 'Distribution of DeviceProtection for Churned Customers', hue = 'gender')


# In[145]:


uniplot(new_df1_target1, col = 'PaperlessBilling', title = 'Distribution of PaperlessBilling for Churned Customers', hue = 'gender')


# In[146]:


uniplot(new_df1_target1, col = 'tenure_group', title = 'Distribution of tenure_group for Churned Customers', hue = 'gender')


# In[147]:


uniplot(new_df1_target1, col = 'StreamingMovies', title = 'Distribution of StreamingMovies for Churned Customers', hue = 'gender')


# **CONCLUSION**
# 
# Some of the insights are:

# In[149]:


#converting our data into csv so we can use it for the machine learning model

churn_dummies.to_csv('churn.csv')


# In[ ]:


# sns.lmplot creates a scatter plot with a linear regression line to show the relationship between 2 variables in d dataset.

# The sns.axisgrid module provides a set of classes for creating and organizing grid-like structures of plots. These classes 
# are used to create multi-plot grids, where each plot represents a subset of the data or displays different aspects of the data.

# FacetGrid is used to create a grid of subplots based on unique values in one or more categorical variables. Each subplot
# represents a subset of the data, and you can apply different plotting functions to each subplot or control the appearance
# of the grid using various parameters.

# <Seaborn.axisgrid.FacetGrid at 0x1809d8e2e80> is the output generated when you create a FacetGrid object.
# <Seaborn.axisgrid.FacetGrid indicates the class type and module of the object.'at 0x1809d8e2e80' represents the memory
# location where the object is stored. This output serves as a confirmation that you have successfully created a FacetGrid object.


# In[ ]:




