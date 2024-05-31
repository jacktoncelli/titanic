import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os


df = pd.read_csv('test.csv')

missing_values = df.isna().sum()

# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2

# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

# extract title from name column (Mr, Miss, Dr, Rev, Mrs, Master, Countess)
# combine sipsp and parch to get family size
# get a new feature called "alone" if family size is 0
# group age and fare into categories (ex children, adults, elders, and different classes of fare costs)
# fill missing values in age column with mean
# scale all the columns down and normalize the data
# rearrange so that survived is the last column of the data set

df.drop(columns=['Cabin', 'PassengerId', 'Ticket'], inplace=True)

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
title_mapping = {
    "Mr": 0,
    "Miss": 1,
    "Mrs": 2,
    "Master": 3,
    "Dr": 4,
    "Rev": 5,
    "Countess": 6
}
df['Title'] = df['Title'].map(title_mapping)
df['Title'].fillna(0, inplace=True)

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Alone'] = 0
df.loc[df['FamilySize'] == 0, 'Alone'] = 1

df.drop(columns=['Name', 'SibSp', 'Parch'], inplace=True)

age_bins = [0, 12, 18, 65, df['Age'].max()]
age_labels = ['Children', 'Teenagers', 'Adults', 'Elders']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

fare_bins = [0, 10, 50, 100, df['Fare'].max()]
fare_labels = ['Low', 'Medium', 'High', 'Very High']
df['FareGroup'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels)

age_mean = df['Age'].mean()
df['Age'].fillna(age_mean, inplace=True)

label_encoder = LabelEncoder()

string_columns = df.select_dtypes(include=['object']).columns.tolist()
for col in string_columns:
    df[col] = label_encoder.fit_transform(df[col])
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


column_order = [
    'Pclass',
    'Sex',
    'Age',
    'Alone',
    'FamilySize',
    'Fare',
    'Embarked',
    'Title',
]

df = df[column_order]

# fix n/a data------------------
# Fill missing values with the mean of each column
for col in df.columns:
    if df[col].dtype != 'object':
        col_mean = df[col].mean()
        df[col].fillna(col_mean, inplace=True)


df.to_csv('cleaned_test.csv', index=False)



