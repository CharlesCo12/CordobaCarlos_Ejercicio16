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
a_1=[]
a_2=[]
a_3=[]
a_4=[]
a_5=[]
a_6=[]
a_7=[]
a_8=[]
a_9=[]
a_10=[]
a_11=[]
a_12=[]
a_13=[]
a_14=[]
for i in (np.array(mean)[:,2]):
    a_1.append(i[0])
    a_2.append(i[1])
    a_3.append(i[2])
    a_4.append(i[3])
    a_5.append(i[4])
    a_6.append(i[5])
    a_7.append(i[6])
    a_8.append(i[7])
    a_9.append(i[8])
    a_10.append(i[9])
    a_11.append(i[10])
    a_12.append(i[11])
    a_13.append(i[12])
    a_14.append(i[13])

#Graficar
plt.figure(figsize=(10,10))
plt.plot(x,a_1,label='1')
plt.plot(x,a_2,label='2')
plt.plot(x,a_3,label='3')
plt.plot(x,a_4,label='4')
plt.plot(x,a_5,label='5')
plt.plot(x,a_6,label='6')
plt.plot(x,a_7,label='7')
plt.plot(x,a_8,label='8')
plt.plot(x,a_9,label='9')
plt.plot(x,a_10,label='10')
plt.plot(x,a_11,label='11')
plt.plot(x,a_12,label='12')
plt.plot(x,a_13,label='13')
plt.plot(x,a_14,label='14')
plt.legend()
loc=0.0
plt.xlabel('Max depth')
plt.ylabel('Average feature importance')
plt.savefig('features.png')
plt.close()
