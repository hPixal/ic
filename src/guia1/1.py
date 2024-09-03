import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def train(data,y_d,w, tr):
    #   y_d is y desired
    #   x is input
    #   y is current model outputy_d[i] - y
    #   w is current model weights
    #   tr is training rate
    for i in range(len(data)):
        
        x = [ -1 , data[i][0] , data[i][1] ]
        y_dot = np.dot(w,x)
        y = 1 if y_dot > 0 else -1
        
        for j in range(len(w)):
            w[j] = w[j] + tr/2 * (y_d[i] - y) * x[j]
    return w

def getTestError(data,y_d,w):
    #   y_d is y desired
    #   x is input
    #   y is current model output
    #   w is current model weights
    acum = 0;
    for i in range(len(data)):
        x = [ -1 , data[i][0] , data[i][1] ]

        y = np.dot(w,x)
        y = 1 if y > 0 else -1
        
        if y != y_d[i]:
            acum = acum + 1
    
    return acum * 1/len(data)

# IMPORT CSV
filename = 'XOR_trn.csv'
data = np.genfromtxt(filename, delimiter=',')

# PLOT DATA
plt.figure()
for i in range(len(data)):
    plt.plot(data[i][0],data[i][1],'ro')


# GENERATE PERCEPTRON WEIGHTS
w =  np.random.rand(3)
w_h = [ w ] # w historA
y_d = data[:,2]

# INITIAL SEASON
w = train(data,y_d,w, 0.26)
w_h.append(w)

# TRAINING & REFINE
err = getTestError(data,y_d,w)
print('err: ',err)
period = 0
while err > 0.07:
    w = train(data,y_d,w, 0.02)
    w_h.append(w)
    period = period + 1
    err = getTestError(data,y_d,w)
    print('err: ',err)
    print('period: ',period)

# PLOT LINE
lw = w_h[len(w_h)-1]

dy_dx = -lw[1]/lw[2];
b = lw[0]/lw[2];

x = np.linspace(-2,2);
y = []

for i in range(len(x)):
    y.append(dy_dx*x[i]+b)
plt.plot(x,y)
plt.show()