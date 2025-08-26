# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model
<img width="935" height="678" alt="481907268-fcae90c4-6a9b-4af0-ba5d-34b8ea2f5249" src="https://github.com/user-attachments/assets/b0b4007e-59e4-4f63-bef1-e990645ef403" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:THRIKESWAR
### Register Number:212222230162
```
import pandas as pd
from os import X_OK

df=pd.read_csv('/content/linear_data_20.csv')

x=df['X'].values
y=df['Y'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=30);

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import torch
import torch.nn as nn
x_train_tensor=torch.tensor(x_train).float()
x_test_tensor=torch.tensor(x_test).float().view(-1,1)
y_train_tensor=torch.tensor(y_train).float()
y_test_tensor=torch.tensor(y_test).float().view(-1,1)

class Neuralnet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(1,8)
    self.fc2=nn.Linear(8,10)
    self.fc3=nn.Linear(10,1)
    self.relu=nn.ReLU()
    self.history={'loss':[]}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x


# Initialize the Model, Loss Function, and Optimizer
ai=Neuralnet()
criterion=nn.MSELoss()
optimizer=torch.optim.RMSprop(ai.parameters(),lr=0.001)


def train_model(ai,x_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(ai(x_train),y_train)
    loss.backward()
    optimizer.step()
    ai.history['loss'].append(loss.item())
    if epoch%200==0:
      print(f'epoch:[{epoch}/{epochs}], loss:{loss.item():.6f}')

train_model(ai,x_train_tensor,y_train_tensor,criterion,optimizer)

with torch.no_grad():
    test_loss = criterion(ai(x_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```
## Dataset Information
<img width="133" height="461" alt="image" src="https://github.com/user-attachments/assets/3111dd48-bcba-4307-aee6-49599f160c0a" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="571" height="455" alt="Untitled" src="https://github.com/user-attachments/assets/17dbf7a0-f6d9-41e2-862e-e4876b5799b8" />


### New Sample Data Prediction
<img width="352" height="52" alt="image" src="https://github.com/user-attachments/assets/75c182dc-cbc1-498b-95cf-2d50c3cc21d6" />


## RESULT
thus,a neural network regression model for the given dataset was developed successfully.
