# test_overfit_small.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from emg_model import emg_model
from emg_dataset import MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# HYPERPARAMS
# ------------------------
BATCH_SIZE = 4   # very small for quick overfit
LR = 0.001
EPOCHS = 30     # few epochs enough to overfit
N_CLASSES = 101
MODE = 'finetuneGRU'

# ------------------------
# DATA
# ------------------------
dataset = MyDataset('train', r'D:\Omar\AVE-Speech_treated_small')

# Take only first 8 samples to overfit
subset_indices = list(range(min(8, len(dataset))))
subset = torch.utils.data.Subset(dataset, subset_indices)

loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------
# MODEL
# ------------------------
model = emg_model(MODE, nClasses=N_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------
# TRAIN LOOP
# ------------------------
for epoch in range(EPOCHS):
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        # FIX: remove extra dimension if exists
        if inputs.dim() == 5:
            inputs = inputs.squeeze(1)  # shape: [batch, channels, 36, 36]

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

print("Overfit test finished!")
