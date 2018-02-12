import numpy as np
import matplotlib.pyplot as plt

w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
b = np.random.randn()

w4 = np.random.randn()
w5 = np.random.randn()
w6 = np.random.randn()
b1 = np.random.randn()

w7 = np.random.randn()
w8 = np.random.randn()
w9 = np.random.randn()
b2 = np.random.randn()

w10 = np.random.randn()
w11 = np.random.randn()
w12 = np.random.randn()
b3 = np.random.randn()

#hidden Layer Weigths and bias
w20= np.random.randn()
w21= np.random.randn()
w22= np.random.randn()
w23= np.random.randn()
b24= np.random.randn()


def sigmoid(x):
    return 1/(1+np.exp(-x))


data = [[0,0,1,0],
        [0,1,1,1],
        [1,0,1,1],
        [0,1,0,1],
        [1,0,0,1],
        [1,1,1,0],
        [0,0,0,0]]

learning_rate = 1
costs = []
for i in range(50000):
    ri = np.random.randint(len(data))
    point = data[ri]
    target = point[3]
    
    output1 = (point[0] * w1) + (point[1] * w2) + (point[2]*w3) + b
    pred1 = sigmoid(output1)
    
    output2 = (point[0] * w4) + (point[1] * w5) + (point[2]*w6) + b1
    pred2 = sigmoid(output2)
    
    output3 = (point[0] * w7) + (point[1] * w8) + (point[2]*w9) + b2
    pred3 = sigmoid(output3)
    
    output4 = (point[0] * w10) + (point[1] * w11) + (point[2]*w12) + b3
    pred4 = sigmoid(output4)
    
    #Hidden Layer
    out5 = (pred1 * w20) + (pred2* w21) + (pred3 * w22) + (pred4 * w23) + b24
    pred5 = sigmoid(out5)
    
    cost = (pred5 - target)**2
    
    #derivatives
    dcost_pred = 2 * (pred5 - target)
    dcost_out5 = sigmoid(out5) * (1-sigmoid(out5))
    dout_w20 = pred1
    dout_w21 = pred2
    dout_w22 = pred3
    dout_w23 = pred4
    dout_b24 = 1
    
    dpred1_output1 = sigmoid(output1) * (1-sigmoid(output1))
    dpred1_w1 = point[0]
    dpred1_w2 = point[1]
    dpred1_w3 = point[2]
    dpred1_b = 1
    
    dpred2_output2 = sigmoid(output2) * (1-sigmoid(output2))
    dpred2_w4 = point[0]
    dpred2_w5 = point[1]
    dpred2_w6 = point[2]
    dpred2_b1 = 1
    
    dpred3_output3 = sigmoid(output2) * (1-sigmoid(output3))
    dpred3_w7 = point[0]
    dpred3_w8 = point[1]
    dpred3_w9 = point[2]
    dpred3_b2 = 1
    
    dpred4_output4 = sigmoid(output4) * (1-sigmoid(output4))
    dpred4_w10 = point[0]
    dpred4_w11 = point[1]
    dpred4_w12 = point[2]
    dpred4_b3 = b3
    
    #hidden layer
    dcost_w20 = dcost_pred * dcost_out5 * dout_w20
    dcost_w21 = dcost_pred * dcost_out5 * dout_w21
    dcost_w22 = dcost_pred * dcost_out5 * dout_w22
    dcost_w23 = dcost_pred * dcost_out5 * dout_w23
    dcost_b24 = dcost_pred * dcost_out5 * dout_b24
    
    
    dcost_w1 = dcost_w20 * dpred1_output1 * dpred1_w1
    dcost_w2 = dcost_w20 * dpred1_output1 * dpred1_w2
    dcost_w3 = dcost_w20 * dpred1_output1 * dpred1_w3
    dcost_b =  dcost_w20 * dpred1_output1 * dpred1_b
    
    dcost_w4 = dcost_w21 * dpred2_output2 * dpred2_w4
    dcost_w5 = dcost_w21 * dpred2_output2 * dpred2_w5
    dcost_w6 = dcost_w21 * dpred2_output2 * dpred2_w6
    dcost_b1 = dcost_w21 * dpred2_output2 * dpred2_b1
    
    dcost_w7 = dcost_w22 * dpred3_output3 * dpred3_w7
    dcost_w8 = dcost_w22 * dpred3_output3 * dpred3_w8
    dcost_w9 = dcost_w22 * dpred3_output3 * dpred3_w9
    dcost_b2 = dcost_w22 * dpred3_output3 * dpred3_b2
    
    dcost_w10 = dcost_w23 * dpred3_output3 * dpred4_w10
    dcost_w11 = dcost_w23 * dpred3_output3 * dpred4_w11
    dcost_w12 = dcost_w23 * dpred3_output3 * dpred4_w12
    dcost_b3 =  dcost_w23 * dpred3_output3 * dpred4_b3
    
    #backpropagation
    w20 -= learning_rate * dcost_w20
    w21 -= learning_rate * dcost_w21
    w22 -= learning_rate * dcost_w22
    b24 -= learning_rate * dcost_b24
    
    w1 -= learning_rate * dcost_w1
    w2 -= learning_rate * dcost_w2
    w3 -= learning_rate * dcost_w3
    b -= learning_rate * dcost_b
    
    w4 -= learning_rate * dcost_w4
    w5 -= learning_rate * dcost_w5
    w6 -= learning_rate * dcost_w6
    b1 -= learning_rate * dcost_b1
    
    w7 -= learning_rate * dcost_w7
    w8 -= learning_rate * dcost_w8
    w9 -= learning_rate * dcost_w9
    b2 -= learning_rate * dcost_b2
    
    w10 -= learning_rate * dcost_w10
    w11 -= learning_rate * dcost_w11
    w12 -= learning_rate * dcost_w12
    b3 -= learning_rate * dcost_b3

    if i % 1000 == 0:
        cost_sum = 0
        ri = np.random.randint(len(data))
        point = data[ri]
        target = point[3]
    
        output1 = (point[0] * w1) + (point[1] * w2) + (point[2]*w3) + b
        pred1 = sigmoid(output1)
    
        output2 = (point[0] * w4) + (point[1] * w5) + (point[2]*w6) + b1
        pred2 = sigmoid(output2)
        
        output3 = (point[0] * w7) + (point[1] * w8) + (point[2]*w9) + b2
        pred3 = sigmoid(output3)
    
        output4 = (point[0] * w10) + (point[1] * w11) + (point[2]*w12) + b3
        pred4 = sigmoid(output4)
    
        #Hidden Layer
        out5 = (pred1 * w20) + (pred2* w21) + (pred3 * w22) + (pred4 * w23) + b24
        pred5 = sigmoid(out5)
    
        cost = (pred5 - target)**2
        cost_sum += np.square(pred5 - target)
        costs.append(cost_sum/len(data))
plt.plot([1,1])
plt.grid()
plt.plot(costs)
plt.show()   


def Perceptron(x, y, z):
    output1 = (x * w1) + (y * w2) + (z*w3) + b
    pred1 = sigmoid(output1)
    
    output2 = (x * w4) + (y * w5) + (z*w6) + b1
    pred2 = sigmoid(output2)
    
    output3 = (x * w7) + (y * w8) + (z*w9) + b2
    pred3 = sigmoid(output3)
    
    output4 = (x * w10) + (y * w11) + (z*w12) + b3
    pred4 = sigmoid(output4)
    
    #Hidden Layer
    out5 = (pred1 * w20) + (pred2* w21) + (pred3 * w22) + (pred4 * w23) + b24
    pred5 = sigmoid(out5)
    return pred5
    