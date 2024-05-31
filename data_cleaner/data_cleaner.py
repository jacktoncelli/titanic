import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os

data_raw_dir = os.path.join(os.getcwd(), "data/raw")

test_csv = os.path.join(data_raw_dir, "test.csv")
train_csv = os.path.join(data_raw_dir, "train.csv")

test_df = pd.read_csv(test_csv)
train_df = pd.read_csv(train_csv)

# test: PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# train: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

# extract the title from Name column (Mr, Miss, etc.)
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.')
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.')


# map each title to a number and encode the column
title_mapping = {
    "Mr": 0,
    "Miss": 1,
    "Mrs": 2,
    "Master": 3,
    "Dr": 4,
    "Rev": 5,
    "Countess": 6
}

train_df['Title'] = train_df['Title'].map(title_mapping)
train_df['Title'].fillna(0, inplace=True)

test_df['Title'] = test_df['Title'].map(title_mapping)
test_df['Title'].fillna(0, inplace=True)


# create a new column that is FamilySize by adding SibSp and Parch
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

# create a new column 'Alone' that is 1 if the family size is 0
train_df['Alone'] = 0
train_df.loc[train_df['FamilySize'] == 0, 'Alone'] = 1

test_df['Alone'] = 0
test_df.loc[test_df['FamilySize'] == 0, 'Alone'] = 1

# create age bins to classify the age group of the passenger
age_bins = [0, 12, 18, 65, train_df['Age'].max()]
age_labels = ['Children', 'Teenagers', 'Adults', 'Elders']

# add new column with age group
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels)
test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=age_bins, labels=age_labels)

# create new bins for the fare (based on quartiles of Fare column)
fare_bins = [0, 8, 15, 32, train_df['Fare'].max()]
fare_labels = ['Low', 'Medium', 'High', 'Very High']

# add new column with the fare group
train_df['FareGroup'] = pd.cut(train_df['Fare'], bins=fare_bins, labels=fare_labels)
test_df['FareGroup'] = pd.cut(test_df['Fare'], bins=fare_bins, labels=fare_labels)

# fill missing values of age column with the mean
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)


# Drop columns that are not categorical or have too many missing values
train_df.drop(columns=['Cabin', 'PassengerId', 'Ticket', 'Name', 'SibSp', 'Parch'], inplace=True)
test_df.drop(columns=['Cabin', 'PassengerId', 'Ticket', 'Name', 'SibSp', 'Parch'], inplace=True)


# create a label encoder to encode the string columns to numerics
label_encoder = LabelEncoder()

string_columns_train = train_df.select_dtypes(include=['object']).columns.tolist()
for col in string_columns_train:
    train_df[col] = label_encoder.fit_transform(train_df[col])
    
string_columns_test = test_df.select_dtypes(include=['object']).columns.tolist()
for col in string_columns_test:
    test_df[col] = label_encoder.fit_transform(test_df[col])
    
# rescale the data
scaler = StandardScaler()
train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = scaler.fit_transform(test_df[['Age', 'Fare']])

# rearrange the order of the columns to have survived as last feature
column_order_train = [
    'Pclass',
    'Sex',
    'Age',
    'Alone',
    'FamilySize',
    'Fare',
    'Embarked',
    'Title',
    'Survived'
]
train_df = train_df[column_order_train]

column_order_test = [
    'Pclass',
    'Sex',
    'Age',
    'Alone',
    'FamilySize',
    'Fare',
    'Embarked',
    'Title',
]
test_df = test_df[column_order_test]

# fix n/a data------------------
# Fill missing values with the mean of each column
for col in train_df.columns:
    if train_df[col].dtype != 'object':
        col_mean = train_df[col].mean()
        train_df[col].fillna(col_mean, inplace=True)
        
for col in test_df.columns:
    if test_df[col].dtype != 'object':
        col_mean = test_df[col].mean()
        test_df[col].fillna(col_mean, inplace=True)

data_clean_dir = os.path.join(os.getcwd(), "data/clean")
train_df.to_csv(os.path.join(data_clean_dir, "cleaned_train.csv"), index=False)
test_df.to_csv(os.path.join(data_clean_dir, "cleaned_test.csv"), index=False)