import pandas as pd
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn import metrics #accuracy measure
from sklearn import svm #support vector Machine
from sklearn.linear_model import LogisticRegression #logistic regression

df=pd.read_csv("Berlin_.csv",index_col="Date")
print(df.head())

df['Events'].value_counts()

#df.drop( df[ df['Events'] == 'Rain-Snow'].index , inplace=True)
#df.drop( df[ df['Events'] == 'Rain-Snow-Hail'].index , inplace=True)
df.drop( df[ df['Events'] == 'Fog-Rain-Snow'].index , inplace=True)
df.drop( df[ df['Events'] == 'Snow'].index , inplace=True)
df.drop( df[ df['Events'] == 'Fog-Rain-Hail-Thunderstorm'].index , inplace=True)
#df.drop( df[ df['Events'] == 'Fog-Snow'].index , inplace=True)
df.drop( df[ df['Events'] == 'Fog-Rain-Thunderstorm'].index , inplace=True)
df.drop( df[ df['Events'] == 'Fog-Rain'].index , inplace=True)
print(df.head())

print(df['Events'].value_counts())

df['Events']=df['Events'].replace('Rain',1)                          
#df['Events']=df['Events'].replace('Fog-Rain',2)
df['Events']=df['Events'].replace('Fog',2)
df['Events']=df['Events'].replace('Rain-Thunderstorm',3)
df['Events']=df['Events'].replace('Rain-Snow',1)
#df['Events']=df['Events'].replace('Snow',3)
#df['Events']=df['Events'].replace('Fog-Rain-Thunderstorm',7)
df['Events']=df['Events'].replace('Fog-Snow',2)
#df['Events']=df['Events'].replace('Fog-Rain-Snow',9)
df['Events']=df['Events'].replace('Thunderstorm',4)
df['Events']=df['Events'].replace('Rain-Snow-Hail',1)
df['Events']=df['Events'].replace('Rain-Hail-Thunderstorm',3)
#df['Events']=df['Events'].replace('Fog-Rain-Hail-Thunderstorm',13)
df['Events']=df['Events'].replace('Rain-Hail',1)
df['Events']=df['Events'].replace(' ',0)
print(df.head())

print(df.describe())

df.fillna(0, inplace = True)
print(df.info())

df.isnull().sum()
df = df.fillna(df.mean())
df.isnull().sum()

df.astype('float64')
df.info()

cor = df.corr()
cor_target = abs(cor['Events'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


Y =  df['Events']
X = df.drop(['Events'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
X.shape


Y.shape


model=RandomForestClassifier(n_estimators=50,max_depth=9,random_state=1)
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction1,y_test))


model=svm.SVC(kernel='linear',C=2,gamma=0.1)
model.fit(X_train,y_train)
prediction2=model.predict(X_test)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction2,y_test))


model = LogisticRegression(random_state=0, solver='saga',multi_class='multinomial')
model.fit(X_train,y_train)
prediction3=model.predict(X_test)
print('The accuracy of the LogisticRegression is',metrics.accuracy_score(prediction3,y_test))
