import random
from tensor_engine import Tensor,MLP,SGD

random.seed(42)

# 2. XOR Dataset: 4 samples of 2 inputs each
X = Tensor([[0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]])

# XOR Targets: shape (4, 1)
y = Tensor([[0.0],
            [1.0],
            [1.0],
            [0.0]])


model=MLP(in_features=2, hidden_dims=[16,16], out_features=1)

optimizer=SGD(model.parameters(), lr=0.5)

print("Starting training XOR....")

for epoch in range(501):
    pred=model(X)

    diff=pred-y
    loss=(diff*diff).mean()

    optimizer.zero_grad()

    loss.backwards()

    optimizer.step()
    if epoch % 100 ==0:
        print(f"Epoch {epoch:3d} | Loss: {loss.data[0]:.6f}")
    
print("\nFinal Model Predictions vs Targets:")
final_pred = model(X)
for i in range(4):
    sample_input = X.data[i*2 : i*2 + 2]
    prediction = final_pred.data[i]
    target = y.data[i]
    print(f"  Input: {sample_input} -> Prediction: {prediction:.4f} (Target: {target:.1f})")    
