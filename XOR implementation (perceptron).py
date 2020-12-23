# XOR Logic Gate implementation (without hidden layer, only 1 perceptron)

import numpy as np

input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
print("input: \n",input_features)

target_output = np.array([0,1,1,0])
target_output = target_output.reshape(4,1)
print("\ntarget: \n",target_output)

weights = np.array([[0.1],[0.2]])
print("\nweights: \n",weights)

bias = 0.3
lr = 0.05

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_der(x):
  return sigmoid(x)*(1-sigmoid(x))

print("\nTraining started...\n")

for epoch in range(300000):
  inputs = input_features
  in_o = np.dot(inputs, weights) + bias
  out_o = sigmoid(in_o)

  error = out_o - target_output
  x = error.sum()

  if epoch % 10000 == 0:
    print(epoch,"th epoch: Error value: ",x)

  derror_douto = error
  douto_dino = sigmoid_der(out_o)

  deriv = derror_douto * douto_dino
  inputs = input_features.T

  deriv_final = np.dot(inputs,deriv)

  weights = weights - lr * deriv_final

  for i in deriv:
    bias = bias - lr * i

print("\nTraining ended...")
print("\noptimal bias: ",bias)
print("optimal weight 1: ",weights[0])
print("optimal weight 2: ",weights[1])

for i in input_features:
  output = i
  result1 = np.dot(output,weights) + bias
  result2 = sigmoid(result1)
  print("--------------------------------------------------------")
  print(output[0]," XOR ",output[1]," = ",result2,"\tTarget Value = ",int((bool(output[0]) ^ bool(output[1]))))
  print("\t\t\t\tERROR : ",int((bool(output[0]) ^ bool(output[1]))) - result2)

