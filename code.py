import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    df = pd.read_excel("bank-full.xlsx")
except FileNotFoundError:
    print("Error: 'bank-full.xlsx' file not found. Please check the file path.")
    exit()

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe())
print(df.dtypes)

sns.boxplot(y=df['age'], x=df['education'])
plt.show()

sns.histplot(df['job'], bins=10)
plt.show()

sns.violinplot(y=df['age'], x=df['education'])
plt.show()

sns.violinplot(y=df['age'], x=df['default'])
plt.show()

correlation = df.corr()
plt.figure(figsize=(7, 7))
sns.heatmap(correlation, cbar=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap="PuBuGn_r")
plt.show()

sns.pairplot(df, hue='default')
plt.show()

le = LabelEncoder()
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

df['y'] = le.fit_transform(df['y'])

X = df[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
         'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
         'previous', 'poutcome']]
y = df['y']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)

print(classification_report(ytest, ypred))
print(confusion_matrix(ytest, ypred))
print(accuracy_score(ytest, ypred))

client = pd.DataFrame(ypred, columns=["Prediction"])
print(client.value_counts())
