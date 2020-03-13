import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import sklearn.tree

# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')

x_train, x_test, y_train, y_test = train_test_split(data[predictors],data['Target'],train_size=0.5)

def fondo(d):
    datos=[]
    for i in range(100):
        n_points = np.shape(y_train)[0]
        indices = np.random.choice(np.arange(n_points), n_points)
        x_new=x_train.iloc[indices,:]
        y_new=y_train.iloc[indices]
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=d)
        clf.fit(x_new,y_new)
        a=[sklearn.metrics.f1_score(y_new,clf.predict(x_new) ),sklearn.metrics.f1_score(y_test, clf.predict(x_test)),clf.feature_importances_]
        datos.append(a)
    return np.array(datos)

dat_fondo=[]
for i in range(10):
    dat_fondo.append(fondo(i+1))

mean=[]
std=[]
for i in dat_fondo:
    mean.append(np.mean(i,axis=0))
    std.append([np.std(i[:,0],axis=0),np.std(i[:,1],axis=0)])

#Gráfica de F1
x=np.arange(1,11)
plt.figure(figsize=(8,8))
plt.errorbar(x,np.array(mean)[:,0],yerr=np.array(std)[:,0],fmt='o',label='Train (50%)')
plt.errorbar(x,np.array(mean)[:,1],yerr=np.array(std)[:,1],fmt='o',label='Test (50%)')
plt.ylabel('Average F1 Score')
plt.xlabel('Max depth')
plt.legend()
loc=0.0
plt.savefig('F1_training_test.png')
plt.close()

#Gráfica de Features
y=np.arange(1,15)
plt.figure(figsize=(8,8))
for i in range(np.shape(np.array(mean)[:,2])[0]):
    plt.plot(y,np.array(mean)[:,2][i],label=str(i+1))
plt.xlabel('Max depth')
plt.ylabel('Average feature importance')
plt.legend()
loc=0.0
plt.savefig('features.png')
plt.close()