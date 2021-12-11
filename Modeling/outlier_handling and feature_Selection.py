def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    
plt.figure(figsize=(30,20))
plot(data_train,'Price')

#dealing with outliers
data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])

plt.figure(figsize=(30,20))
plot(data_train,'Price')

#seperating dependent and independent variables

X=data_train.drop('Price',axis=1)
y=data_train['Price']

#feature selection

from sklearn.feature_selection import mutual_info_classif
mutual_info_classif()
mutual_info_classif(X,y)

imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
imp

imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)