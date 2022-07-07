import numpy as  np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#from class.decision_tree_classification import X_test
df1=pd.read_csv('train_ctrUa4K.csv')
df2=pd.read_csv('test_lAUu6dG.csv')
print(df1.head())
print(df1.info())
df_all=pd.concat([df1,df2],sort=True).reset_index(drop=True)
print(df_all.head())
print(df_all.info())
print(df1.isnull().sum())
print(df2.isnull().sum())
#df_all['Gender']=df_all['Gender'].fillna(df_all['Gender'].mode(),inplace=True)
#print('\n\n\nAfter replace')
#print(df_all.info())
#df_all['Dependents']=df_all['Dependents'].map.apply(lambda x : 3 if(x == '3+') else x)
#df_all['Dependents']=df_all['Dependents'].fillna(df_all['Dependents'].mode(),inplace=True)
df_all['Dependents']=df_all['Dependents'].map(lambda x : '3' if(x == '3+') else x)
# df_all['Dependents']=df_all['Dependents'].astype('int64')
df_all['Dependents']=df_all['Dependents'].fillna('0')
df_all['Dependents']=df_all['Dependents'].astype('int64')
df_all['Credit_History']=df_all['Credit_History'].fillna(1.0)
print(df_all['Gender'].value_counts())
df_all['Gender']=df_all['Gender'].fillna('Male')
print(df_all.isnull().sum())
print(max(df_all['Married'].value_counts()))
df_all['Married']=df_all['Married'].fillna('Yes')
print(df_all['Self_Employed'].value_counts())
df_all['Self_Employed']=df_all['Self_Employed'].fillna('No')
print(df_all.isnull().sum())
print(df_all[['LoanAmount','Loan_Amount_Term']].describe())
sn.histplot(data=df_all,x=df_all['LoanAmount'],bins=20,hue='Property_Area')
plt.show()
df_all['LoanAmount']=df_all['LoanAmount'].fillna(df_all['LoanAmount'].median())
sn.barplot(data=df_all,x='Property_Area',y=df_all['LoanAmount'])
plt.show()
sn.histplot(data=df_all,x=df_all['Loan_Amount_Term'],bins=20)
plt.show()
print(df_all['Loan_Amount_Term'].mode())
df_all['Loan_Amount_Term']=df_all['Loan_Amount_Term'].fillna(df_all['Loan_Amount_Term'].median())
print(df_all[['LoanAmount','Loan_Amount_Term']].describe())
print(df_all.isnull().sum())
sn.boxplot(data=df_all,y=df_all['LoanAmount'])
plt.show()
print(df_all['Loan_ID'].head())
df_all['LI']=df_all['Loan_ID'].str.slice(start=2).astype('int64')
print(df_all['LI'])
print(df_all.info())
cat=['Education','Gender','Married','Self_Employed','Property_Area','Loan_Status']
df_all['Education']=df_all['Education'].map(lambda x : 1 if(x == 'Graduate') else 0)
df_all['Gender']=df_all['Gender'].map(lambda x : 1 if(x == 'Male') else 0)
df_all['Married']=df_all['Married'].map(lambda x : 1 if(x == 'Yes') else 0)
df_all['Self_Employed']=df_all['Self_Employed'].map(lambda x : 1 if(x == 'Yes') else 0)
df_all['Property_Area']=df_all['Property_Area'].map(lambda x : 3 if(x == 'Urban') else  (2 if(x == 'Semiurban')  else 1))
df_all['Loan_Status']=df_all['Loan_Status'].map(lambda x : 1 if(x == 'Y') else 0)
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# ordinal_encoder = OrdinalEncoder()
# for x in cat:
#     #print(df_all[x].head())
#     df_all[[x]] = ordinal_encoder.fit_transform(df_all[[x]])
# a=df_all[["Education"]]
# y = ordinal_encoder.fit_transform(a)
df_all['fe1']=(df_all['LoanAmount']*1000)/(df_all['Loan_Amount_Term']/12)
df_all['fe2']=((df_all['ApplicantIncome']+df_all['CoapplicantIncome'])*12)*(df_all['Loan_Amount_Term']/12)
df_all['fe3']=(df_all['ApplicantIncome']+df_all['CoapplicantIncome'])
df_all['LoanAmount']=df_all['LoanAmount']*1000
sn.histplot(data=df_all,x=df_all['fe1'],bins=20)
plt.show()
sn.histplot(data=df_all,x=df_all['fe2'],bins=20)
plt.show()
sn.histplot(data=df_all,x=df_all['fe3'],bins=20)
plt.show()
sn.histplot(data=df_all,x=df_all['LoanAmount'],bins=20)
plt.show()
df_all['fe1']=pd.DataFrame(np.log(df_all['fe1']))
df_all['fe2']=pd.DataFrame(np.log(df_all['fe2']))
df_all['fe3']=pd.DataFrame(np.log(df_all['fe3']))
df_all['LoanAmount']=pd.DataFrame(np.log(df_all['LoanAmount']))
sn.histplot(data=df_all,x=df_all['fe1'],bins=20)
plt.show()
sn.histplot(data=df_all,x=df_all['fe2'],bins=20)
plt.show()
sn.histplot(data=df_all,x=df_all['fe3'],bins=20)
plt.show()
sn.histplot(data=df_all,x=df_all['LoanAmount'],bins=20)
plt.show()
print(df_all.info())
print(df1.shape)
df_train=df_all.loc[:613]
df_test=df_all.loc[614:].drop(['Loan_Status'],axis=1)
print(df_train.isnull().sum())
print(df_test.isnull().sum())
print(df_all.corr())
sn.heatmap(df_all.corr())
plt.show()
from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
x=s.fit_transform(df_train.drop(columns=['Loan_Status','Loan_ID','LI','ApplicantIncome','CoapplicantIncome','Gender','Married']))
y=df_train['Loan_Status'].values
X_test=m.fit_transform(df_test.drop(columns=['Loan_ID','LI','ApplicantIncome','CoapplicantIncome','Gender','Married']))
from sklearn.model_selection import train_test_split
X_train, X_test1, y_train, y_test1 = train_test_split(x, y, test_size = 0.25, random_state = 0)
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(criterion = 'gini', random_state = 123,max_depth=7,n_estimators=250)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 200,max_depth=3)
# from sklearn.linear_model import LogisticRegression
# classifier=LogisticRegression(C=1.0, class_weight=None, dual=False,fit_intercept=True,intercept_scaling=1, l1_ratio=None,max_iter=1000, multi_class='ovr', n_jobs=1, penalty='l2',random_state=None,solver='liblinear',tol=0.0001,verbose=0,warm_start=False)
from sklearn.metrics import precision_score, recall_score
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)
print(classifier.score(X_test1,y_test1))
y_pred=classifier.predict(X_test1)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test1, y_pred)
print(cm)
print(precision_score(y_test1, y_pred))
print(recall_score(y_test1, y_pred))
accuracy_score(y_test1, y_pred)
op=pd.DataFrame({'Loan_ID':df_test['Loan_ID'],'Loan_Status':predictions})
op['Loan_Status']=op['Loan_Status'].map(lambda x : 'Y' if(x == 1) else 'N')
op.to_csv('AVHackthon.csv',index=False)










