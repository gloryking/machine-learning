#XOR Logic Gate implementation with hidden layer

# Import required libraries
import numpy as np
# define input features :
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
print(input_features)
print(input_features.shape)
# define target output :
target_output = np.array([0,1,1,0])
target_output = target_output.reshape(4,1)
print(target_output)
print(target_output.shape)
# define weights:
# 6 for hidden layer
weight_hidden = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
print(weight_hidden)
# 3 for output layer
weight_output = np.array([[0.7],[0.8],[0.9]])
#learning rate :
lr = 0.05
# sigmoid function
def sigmoid(x):
  return 1/(1+np.exp(-x))
# derivative of sigmoid function
def sigmoid_der(x):
  return sigmoid(x)*(1-sigmoid(x))

print("\nTraining started...\n")

for epoch in range(300000):
  #input for hidden layer:
  input_hidden = np.dot(input_features,weight_hidden)

  #output from hidden layer:
  output_hidden = sigmoid(input_hidden)

  #input for output layer:
  input_op = np.dot(output_hidden, weight_output)

  #output from output layer:
  output_op = sigmoid(input_op)
  #========================== Phase 1 ================================
  error_out = ((1/2) * (np.power((output_op - target_output),2)))
  x = error_out.sum()
  if epoch%10000 == 0:
    print(epoch,"th epoch: Error value: ",x)

  # Derivatives for phase 1:
  derror_douto = output_op - target_output
  douto_dino = sigmoid_der(input_op)
  dino_dwo = output_hidden
  derror_dwo = np.dot(dino_dwo.T,derror_douto * douto_dino)

  # Derivatives for phase 2:
  derror_dino = derror_douto * douto_dino
  dino_douth = weight_output
  derror_douth = np.dot(derror_dino,dino_douth.T)

  douth_dinh = sigmoid_der(input_hidden)
  dinh_dwh = input_features
  derror_dwh = np.dot(dinh_dwh.T,douth_dinh * derror_douth)

  # update weights
  weight_hidden = weight_hidden - lr * derror_dwh
  weight_output = weight_output - lr * derror_dwo

print("\nTraining ended...\n")

# printing optimal weights
print("hidden weights: \n",weight_hidden)
print("output weights: \n",weight_output)

# predictions
for i in input_features:
  output = i
  
  result1 = np.dot(output,weight_hidden)
  result2 = sigmoid(result1)
  result3 = np.dot(result2,weight_output)
  result4 = sigmoid(result3)

  print("--------------------------------------------------------")
  print(output[0]," XOR ",output[1]," = ",result4,"\tTarget Value = ",int((bool(output[0]) ^ bool(output[1]))))
  print("\t\t\t\tERROR : ",int((bool(output[0]) ^ bool(output[1]))) - result4)