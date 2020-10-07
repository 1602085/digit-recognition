import numpy as np
import random
import idx2numpy
from sklearn.neural_network import MLPClassifier    

file1 = '/train-images.idx3-ubyte'
input1 = idx2numpy.convert_from_file(file1)

file2 = '/train-labels.idx1-ubyte'
output = idx2numpy.convert_from_file(file2)


w1 = np.random.random(3240).reshape(324,10)
w2 = np.random.random(100).reshape(10,10)
w3 = np.random.random(100).reshape(10,10)
w4 = np.random.random(100).reshape(10,10)


f1 = [[-1/3,0,1/3], [-1/3,0,1/3], [-1/3,0,1/3]]
f2 = [[-1/3,-1/3,-1/3], [0,0,0], [1/3,1/3,1/3]]
final_layer = []


def feature(input1_feature):
    x, y = input1_feature.shape
    final_layer = []
    for p in range(0, y-2):
        for q in range(0, x-2):
            horizontal = (input1_feature[q:q+3, p:p+3]*f1).sum()
            vertical = (input1_feature[q:q+3, p:p+3]*f2).sum()
            mag_gradient = np.sqrt(horizontal**2 + vertical**2)
            mag_gradient = mag_gradient/225
            final_layer.append(mag_gradient)
    final_layer = np.array(final_layer)
    final_layer = final_layer.reshape(26,26)

    final_layer = final_layer[4:22, 4:22]
    feature = final_layer.flatten()
    return feature


def leaky_relu(x):
    if x >= 0:
       y = x
    else:
       y = 0.01*x
    return y


def softmax_layer(a):
    c = []
    d = []
    for i in a:
        c.append(np.exp(i))
    k = np.sum(c)
    for j in c:
        d.append(j/k)
    return d


num_iter = 200
pred_out = []


def Backpropogation(input1, output, w1, w2, w3, w4, mini_batch=0): 
    for k in range(num_iter):  
        for i in range(mini_batch, mini_batch+300):
            inp11 = input1[i]
            inp = feature(inp11)
            out = output[i]
            
            hidden_layer1 = np.dot(inp, w1)
 
            hidden_layer11 = []   
            for j in hidden_layer1:
                hidden_layer11.append(leaky_relu(j))
  
            hidden_layer2 = np.dot(np.array(hidden_layer11), w2)
            hidden_layer22 = []

            for j in hidden_layer2:
                hidden_layer22.append(leaky_relu(j))

            hidden_layer3 = np.dot(np.array(hidden_layer22), w3) 
            hidden_layer33 = []
 
            for j in hidden_layer3:
                hidden_layer33.append(leaky_relu(j))

            output_layer = np.dot(np.array(hidden_layer33).reshape(1,10), w4).astype(int)
            output_layer_mean = output_layer.mean()
            output_layer = output_layer / output_layer_mean #if you don't put this, then softmax_layer give nan values
            
            out_layer = softmax_layer(output_layer)
            print(out_layer)
            output1 = max(out_layer)
            pred_out.append(output1)
  
            l = len(pred_out)

            output_error = (np.array(pred_out[l-1]) - np.array(out))*np.array(hidden_layer33).reshape(10,1)

            w4 = w4 - alpha*output_error
            h_layer3 = (np.array(pred_out[l-1]) - np.array(out))*np.array(hidden_layer33).reshape(10,1)*np.array(hidden_layer22)

            w3 = w3 - alpha*h_layer3 
            h_layer2 = (np.array(pred_out[l-1])-np.array(out))*np.array(hidden_layer33).reshape(10,1)*np.array(hidden_layer22)*np.array(hidden_layer11) 
 
            w2 = w2-alpha*h_layer2
            h_layer1 = np.sum((np.array(pred_out[l-1]) -np.array(out))*np.array(hidden_layer33).reshape(10,1)*np.array(hidden_layer22))*np.array(hidden_layer11).reshape(10,1)*inp

            h_layer1 = h_layer1.transpose()
            w1 = w1-alpha*h_layer1

        mini_batch = mini_batch + 300
    return w4, w3, w2, w1


w4, w3, w2, w1 = Backpropogation(input1, output, w1, w2, w3, w4, mini_batch=0)
print(w4)
print(w3)
print(w2)
print(w1)
   
    
#pred_out = np.array(pred_out, dtype=uint8)

#total=0
#for i in range(len(output)):
#    total += output[i]


#def accuracy(output, pred_out):
#    pred = 0
#    for i in range(len(output)):
#        if output[i] == pred_out[i]:
#           pred += output[i]
#    accur = pred / total
#    return accur

#score = accuracy(output, pred_out)
#print(score)




