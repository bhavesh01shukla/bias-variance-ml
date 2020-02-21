import numpy as np
import pickle 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split  
from sklearn import datasets, linear_model, metrics 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 

def find_bias(y_exp,y_test):  
    y_bias=np.zeros(80)
    
    y_bias=np.subtract(y_test,y_exp)
    y_bias=np.square(y_bias)
    bias2=np.mean(y_bias)
    return bias2

def find_variance(y_exp,store_pred):
    y_var=np.zeros(80)
    for i in range(0,80):         
        for j in range(0,20):     
            y_var[i]+=(store_pred[j][i]-y_exp[i])**2

    y_var=np.divide(y_var,20)    
    var=np.mean(y_var)
    return var        

# database 
db = {} 
##############################################################
dbfile = open('./Q2_data/X_train.pkl', 'rb')      
db = pickle.load(dbfile) 
x_train=[] 
for i in db:
    x_train.append(i)
    # y.append(i[1])

dbfile.close() 
x_train = np.array(x_train)  ## training features

#############################################################
dbfile = open('./Q2_data/Y_train.pkl', 'rb')      
db = pickle.load(dbfile) 
y_train=[] 
for i in db:
    y_train.append(i)

dbfile.close() 
y_train = np.array(y_train)  ## training responses

#############################################################
dbfile = open('./Q2_data/X_test.pkl', 'rb')      
db = pickle.load(dbfile) 
x_test=[] 
for i in db:
    x_test.append(i)

dbfile.close() 
x_test = np.array(x_test)  ## training responses

#############################################################
dbfile = open('./Q2_data/Fx_test.pkl', 'rb')      
db = pickle.load(dbfile) 
y_test=[] 
for i in db:
    y_test.append(i)

dbfile.close() 
y_test = np.array(y_test)  ## training responses
#############################################################


# x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

######## dividing training set into 10 subparts ########


training_x = np.copy(x_train)
training_y = np.copy(y_train) 

bias2_list=[]
var_list=[]
deg_list=[]

for degree in range(1,10):
    poly = PolynomialFeatures(degree)

    y_exp=np.zeros(80)  ### 80 is the test size
    store_pred=np.zeros(1600)  #### 20 training sets
    store_pred = np.array_split(store_pred,20) 

    for i in range(0,20):  ### 20 training sets present
        train_x=training_x[i].reshape(-1,1)
        
        x_poly_train = np.array(poly.fit_transform(train_x))
        x_poly_test = np.array(poly.fit_transform(x_test))

        lin = LinearRegression()  
        lin.fit(x_poly_train,training_y[i]) 
        y_predict=lin.predict(x_poly_test)

        for k in range(0,80):
            store_pred[i][k]=y_predict[k]
        y_exp=np.add(y_exp,y_predict)

 
    y_exp=np.divide(y_exp,20)
    bias2=find_bias(y_exp,y_test)
    var=find_variance(y_exp,store_pred)

    print("degree:",degree,"    bias^2:",bias2,"   var:",var)
    bias2_list.append(bias2)
    var_list.append(var)
    deg_list.append(degree)


plt.plot(deg_list,bias2_list,label="bias^2")
plt.plot(deg_list,var_list,label="variance")
plt.xlabel('Degree of poly')
plt.ylabel('Error')
plt.legend()
plt.show()
