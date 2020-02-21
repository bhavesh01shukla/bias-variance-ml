import numpy as np
import pickle 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn import datasets, linear_model, metrics 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 

def find_bias(y_exp,y_test):  
    y_bias=np.zeros(500)
    
    y_bias=np.subtract(y_test,y_exp)
    y_bias=np.square(y_bias)
    bias2=np.mean(y_bias)
    return bias2

def find_variance(y_exp,store_pred):
    y_var=np.zeros(500)
    for i in range(0,500):         
        for j in range(0,10):     
            y_var[i]+=(store_pred[j][i]-y_exp[i])**2
   
    y_var=np.divide(y_var,10)
    var=np.mean(y_var)
    return var        

# database 
db = {} 
dbfile = open('./Q1_data/data.pkl', 'rb')      
db = pickle.load(dbfile) 


x=[] 
y=[] 
################# loading data into numpy array ##########
for i in db:
    x.append(i[0])
    y.append(i[1])

dbfile.close() 

x = np.array(x)  ##features
y = np.array(y)  ##response


###### splitting data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1) 

x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)



######## dividing training set into 10 subparts ########
ind=0
temp_x=[]
temp_y=[]
for k in range (0,10):
    a1=[]
    a2=[]
    for i in range(ind,ind+450):
        a1.append(x_train[i])
        a2.append(y_train[i])

    temp_x.append(a1)
    temp_y.append(a2)    
    ind+=450

training_x = np.array(temp_x)
training_y = np.array(temp_y) 


for degree in range(1,10):
    poly = PolynomialFeatures(degree)
    plt.scatter(x_test, y_test, color="blue",label="test data")


    y_exp=np.zeros(500)
    store_pred=np.zeros(5000)
    store_pred = np.array_split(store_pred,10) 

    for i in range(0,10):
        x_poly_train = np.array(poly.fit_transform(training_x[i]))
        x_poly_test = np.array(poly.fit_transform(x_test))

        lin = LinearRegression()  
        lin.fit(x_poly_train,training_y[i]) 
        y_predict=lin.predict(x_poly_test)

        for k in range(0,500):
            store_pred[i][k]=y_predict[k]

        y_exp=np.add(y_exp,y_predict)

    y_exp=np.divide(y_exp,10)

    bias2=find_bias(y_exp,y_test)
    var=find_variance(y_exp,store_pred)

    print("degree:",degree,"    bias^2:",bias2,"   var:",var)

    plt.scatter(x_test,y_exp,color="red",label="poly fit")
    plt.legend()
    plt.show()
