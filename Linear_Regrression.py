# It's basic linear regression model.
import torch
import torch.nn as nn
import torch.optim as optim

weight, bias = 0.8, 0.2
start, end, step = 0, 1, 0.002
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

train_split= int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

class LinearRegModel(nn.Module):
  def __init__ (self):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(1, requires_grad = True,
                                           dtype = torch.float))
    self.bias = nn.Parameter(torch.randn(1, requires_grad = True,
                                         dtype = torch.float))
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weight * x + self.bias

model_2 = LinearRegModel()
model_2.state_dict()
with torch.inference_mode():
  y_preds1 = model_2(X_test)
y_preds1

loss_fn1 = nn.MSELoss()
optimizer = optim.SGD(params = model_2.parameters(), lr = 0.01)

torch.manual_seed(1000)
epochs1 = 3500
train_loss_values1 = []
test_loss_values1 = []
epochs_count1 = []
for epoch in range(epochs1):
  model_2.train()
  y_pred1 = model_2(X_train)
  loss1 = loss_fn1(y_pred1, y_train)
  optimizer.zero_grad()
  loss1.backward()
  optimizer.step()
  model_2.eval()
  with torch.inference_mode():
    test_preds = model_2(X_test)
    test_loss = loss_fn1(test_preds, y_test)
    if epoch % 10 == 0:
      epochs_count1.append(epoch)
      train_loss_values1.append(loss1.detach().numpy())
      test_loss_values1.append(test_loss.detach().numpy())
      print(f"Epoch: {epoch} | Train Loss: {loss1} | Test Loss: {test_loss}")
