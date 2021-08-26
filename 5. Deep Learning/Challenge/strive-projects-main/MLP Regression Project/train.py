#Define a train and validation function

import torch

best_val_loss = 100000


with torch .no_grad():
    val_loss = 10
    for i in range(10):
        print(i)

    if val_loss < best_val_loss:
        torch.save(model, "name.pth")
        best_val_loss = val_loss

    