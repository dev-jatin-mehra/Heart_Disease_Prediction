# Heart Disease Prediction
# Pre Processing Data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
heart_df = pd.read_excel("D:\\Datasets\\heart_disease\\heart_disease_data.xlsx")
heart_df.head()
heart_df.tail()
heart_df.shape
heart_df.info()
#statistical values for the dataframe
heart_df.describe()
heart_df.isnull().sum()
#implies there are no missing values
heart_df['target'].value_counts()
#the distribution is even
#checking the distribution of target in the dataframe
import seaborn as sns
plt.figure(figsize=(8,6))
sns.histplot(data = heart_df,x='age',hue='target',multiple='stack',bins=10,palette='viridis')
plt.title('Distribution of targets on the basis of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(title='Target',labels=['Non-heart disease','heart disease'])
plt.grid(True)
plt.show()
#separating the data and labels
X = heart_df.drop(columns='target',axis = 1)
Y = heart_df['target']
X.head()
Y.head()
Y.tail()
#splitting the data into training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.2,stratify=Y,random_state=2)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
#Creating the model -> binary classification -> Ligistic
model = LogisticRegression()
model.fit(X_train,Y_train)
#model_evaluation
X_train_prediction = model.predict(X_train)
train_accuracy_score = accuracy_score(X_train_prediction,Y_train)
print(train_accuracy_score)
X_test_prediction = model.predict(X_test)
test_accuracy_score = accuracy_score(X_test_prediction,Y_test)
print(test_accuracy_score)
#making a predictive system 
input = (58,1,0,146,218,0,1,105,0,2,1,1,3)

#58,1,0,100,234,0,1,156,0,0.1,2,1,3 something wrong for this input..

convert_numpy = np.asarray(input)

input_reshape = convert_numpy.reshape(1,-1)

prediction = model.predict(input_reshape)

print(prediction[0])

if ((prediction[0])==1):
    print("Person have Heart Disease !")
else:
    print("Person is free from heart disease !")



