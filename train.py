import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv')

train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.')
title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Mlle': 1, 'Ms': 1, 'Mme': 2}
train_df['Title'] = train_df['Title'].map(title_map).fillna(4)

train_df['Age'] = train_df['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'))
train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
x_train = train_df[features]
y_train = train_df['Survived']

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(x_train, y_train)

joblib.dump(model, 'model.pkl')
print('saved')