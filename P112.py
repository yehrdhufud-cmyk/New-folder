import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("PART 1: DATA PREPARATION & MODEL DEFINITION")
print("=" * 60)


n_samples = 2000

x1 = np.random.uniform(-3, 3, n_samples)
x2 = np.random.uniform(-3, 3, n_samples)
x3 = np.random.uniform(-3, 3, n_samples)
x4 = np.random.uniform(-3, 3, n_samples)


y = 2 * np.sin(x1 * np.pi) + 0.5 * np.cos(x2 * 3) + 0.3 * (x3 ** 3) + 0.1 * (x4 ** 2)
y += np.sin(x1 * x2) * 0.5
y += np.cos(x3 * x4) * 0.3
y += 0.2 * np.abs(x1 + x2 + x3 + x4)


noise = np.random.normal(0, 0.8, n_samples)
y = y + noise

X = np.column_stack([x1, x2, x3, x4])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()


X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.ln1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.shortcut = nn.Linear(4, 64)
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(32, 16)
        self.ln4 = nn.LayerNorm(16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out + identity
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.ln3(out)
        out = F.relu(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        out = self.ln4(out)
        out = F.relu(out)
        out = self.fc5(out)
        return out


def get_optimizer(model, lr=0.1):
    weight_decay_params = []
    no_weight_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'ln' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    
    return optim.AdamW([
        {'params': weight_decay_params, 'weight_decay': 0.01},
        {'params': no_weight_decay_params, 'weight_decay': 0.0}
    ], lr=lr)

criterion = nn.MSELoss()
MAX_GRAD_NORM = 1.0

print(f"Device: {device}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input features: 4")
print(f"Hidden layers: 64 → 64 → 32 → 16 → 1")
print("=" * 60)



# ================== PART 2: REGULAR MODEL TRAINING ==================


print("\n" + "=" * 60)
print("PART 2: TRAINING REGULAR MODEL (WITHOUT SWA)")
print("=" * 60)


model_regular = FixedModel().to(device)
optimizer_regular = get_optimizer(model_regular, lr=0.001)
scheduler_regular = optim.lr_scheduler.ReduceLROnPlateau(optimizer_regular, mode='min', patience=20, factor=0.7)

epochs = 200
patience = 40
best_val_loss = float('inf')
patience_counter = 0

train_losses_regular = []
val_losses_regular = []

for epoch in range(epochs):
    model_regular.train()
    train_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_regular.zero_grad()
        pred = model_regular(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_regular.parameters(), MAX_GRAD_NORM)
        optimizer_regular.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model_regular.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_loss += criterion(model_regular(X_batch), y_batch).item()
    val_loss /= len(test_loader)
    
    scheduler_regular.step(val_loss)
    train_losses_regular.append(train_loss)
    val_losses_regular.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model_regular.state_dict(), "regular_best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("\n" + "=" * 60)
print("PART 2 FINISHED - Regular Model Training Complete")
print("=" * 60)


model_regular.load_state_dict(torch.load("regular_best_model.pth"))
model_regular.eval()

with torch.no_grad():
    y_pred_regular_scaled = model_regular(X_test_t.to(device)).cpu().numpy()
    y_pred_regular = scaler_y.inverse_transform(y_pred_regular_scaled)
    y_true = scaler_y.inverse_transform(y_test_t.cpu().numpy())

mse_regular = mean_squared_error(y_true, y_pred_regular)
rmse_regular = np.sqrt(mse_regular)
r2_regular = r2_score(y_true, y_pred_regular)

print(f"\n📊 REGULAR MODEL RESULTS:")
print(f"   ✅ MSE:  {mse_regular:.4f}")
print(f"   ✅ RMSE: {rmse_regular:.4f}")
print(f"   ✅ R²:   {r2_regular:.4f}")
print("=" * 60)


# ================== PART 3A: SWA MODEL TRAINING ==================

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

print("\n" + "=" * 60)
print("PART 3A: TRAINING SWA MODEL (Stochastic Weight Averaging)")
print("=" * 60)


model_swa = FixedModel().to(device)
swa_model = AveragedModel(model_swa)
optimizer_swa = get_optimizer(model_swa, lr=0.001)

SWA_START_EPOCH = 100  
swa_scheduler = SWALR(optimizer_swa, swa_lr=0.001, anneal_epochs=5, anneal_strategy='cos')
scheduler_cosine = CosineAnnealingLR(optimizer_swa, T_max=SWA_START_EPOCH, eta_min=1e-6)

train_losses_swa = []
val_losses_swa = []
best_val_loss_swa = float('inf')
patience_counter_swa = 0

for epoch in range(epochs):
    model_swa.train()
    train_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_swa.zero_grad()
        pred = model_swa(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_swa.parameters(), MAX_GRAD_NORM)
        optimizer_swa.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    
    if epoch >= SWA_START_EPOCH:
        swa_model.update_parameters(model_swa)  
        swa_scheduler.step()
        optimizer_swa.param_groups[0]['lr'] = swa_scheduler.get_last_lr()[0]
    else:
        scheduler_cosine.step()
    
    
    model_swa.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_loss += criterion(model_swa(X_batch), y_batch).item()
    val_loss /= len(test_loader)
    
    train_losses_swa.append(train_loss)
    val_losses_swa.append(val_loss)
    
    if val_loss < best_val_loss_swa:
        best_val_loss_swa = val_loss
        patience_counter_swa = 0
    else:
        patience_counter_swa += 1
        if patience_counter_swa >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if (epoch + 1) % 30 == 0:
        current_lr = optimizer_swa.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

print("\n" + "=" * 60)
print("PART 3A FINISHED - SWA Model Training Complete")
print("=" * 60)


torch.save(swa_model.state_dict(), "swa_model.pth")



print("\n" + "=" * 60)
print("PART 3B1: EVALUATING SWA MODEL & CALCULATING METRICS")
print("=" * 60)

# Load SWA model if needed
# swa_model.load_state_dict(torch.load("swa_model.pth"))

# Evaluate SWA model on test data
swa_model.eval()
with torch.no_grad():
    y_pred_swa_scaled = swa_model(X_test_t.to(device)).cpu().numpy()
    y_pred_swa = scaler_y.inverse_transform(y_pred_swa_scaled)
    y_true = scaler_y.inverse_transform(y_test_t.cpu().numpy())

# Calculate evaluation metrics for SWA
mse_swa = mean_squared_error(y_true, y_pred_swa)
rmse_swa = np.sqrt(mse_swa)
r2_swa = r2_score(y_true, y_pred_swa)
mae_swa = np.mean(np.abs(y_true - y_pred_swa))

# Check if regular model results exist from PART 2
try:
    mse_regular
    rmse_regular
    r2_regular
    y_pred_regular
    train_losses_regular
    val_losses_regular
except NameError:
    print("\nWARNING: Regular model results not found!")
    print("Please run PART 2 first, then run this section again.")
    mse_regular = None

print(f"\nSWA MODEL METRICS:")
print(f"   MSE:  {mse_swa:.4f}")
print(f"   RMSE: {rmse_swa:.4f}")
print(f"   MAE:  {mae_swa:.4f}")
print(f"   R2:   {r2_swa:.4f}")

# Save results for next part
swa_results = {
    'mse': mse_swa,
    'rmse': rmse_swa,
    'r2': r2_swa,
    'mae': mae_swa,
    'predictions': y_pred_swa,
    'true_values': y_true
}

print("\n" + "=" * 60)
print("PART 3B1 FINISHED - SWA Evaluation Complete")
print("=" * 60)



print("\n" + "=" * 60)
print("PART 3B2: COMPARISON & VISUALIZATION")
print("=" * 60)

if mse_regular is None:
    print("ERROR: Regular model results not found. Please run PART 2 first.")
    exit()

improvement_mse = ((mse_regular - mse_swa) / mse_regular) * 100
improvement_rmse = ((rmse_regular - rmse_swa) / rmse_regular) * 100
improvement_r2 = (r2_swa - r2_regular) * 100

print("\n" + "=" * 60)
print("FINAL COMPARISON: REGULAR vs SWA")
print("=" * 60)
print(f"{'Metric':<12} {'Regular':<15} {'SWA':<15} {'Improvement':<12}")
print("-" * 60)
print(f"{'MSE':<12} {mse_regular:<15.4f} {mse_swa:<15.4f} {improvement_mse:>6.2f}%")
print(f"{'RMSE':<12} {rmse_regular:<15.4f} {rmse_swa:<15.4f} {improvement_rmse:>6.2f}%")
print(f"{'R2':<12} {r2_regular:<15.4f} {r2_swa:<15.4f} {improvement_r2:>6.2f}%")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(train_losses_regular, label='Regular Train', alpha=0.8)
axes[0, 0].plot(val_losses_regular, label='Regular Val', alpha=0.8)
axes[0, 0].plot(train_losses_swa, label='SWA Train', alpha=0.8)
axes[0, 0].plot(val_losses_swa, label='SWA Val', alpha=0.8)
axes[0, 0].axvline(x=SWA_START_EPOCH, color='green', linestyle='--', linewidth=2, label=f'SWA Start (epoch {SWA_START_EPOCH})')
axes[0, 0].set_title('Loss Curves Comparison', fontsize=12)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].legend()
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_true, y_pred_regular, alpha=0.4, s=10, label=f'Regular (R2={r2_regular:.3f})', color='blue')
axes[0, 1].scatter(y_true, y_pred_swa, alpha=0.4, s=10, label=f'SWA (R2={r2_swa:.3f})', color='green', marker='x')
axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predictions')
axes[0, 1].set_title('Predictions: Regular vs SWA')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

errors_regular = y_pred_regular.flatten() - y_true.flatten()
axes[1, 0].hist(errors_regular, bins=30, edgecolor='black', alpha=0.6, label='Regular', color='blue')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].axvline(x=np.mean(errors_regular), color='darkblue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors_regular):.3f}')
axes[1, 0].set_title(f'Error Distribution - Regular Model (std={np.std(errors_regular):.3f})')
axes[1, 0].set_xlabel('Prediction Error')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

errors_swa = y_pred_swa.flatten() - y_true.flatten()
axes[1, 1].hist(errors_swa, bins=30, edgecolor='black', alpha=0.6, label='SWA', color='green')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(x=np.mean(errors_swa), color='darkgreen', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors_swa):.3f}')
axes[1, 1].set_title(f'Error Distribution - SWA Model (std={np.std(errors_swa):.3f})')
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

sample = np.array([[1.5, -2.0, 0.5, 1.0]])
sample_scaled = scaler_X.transform(sample)
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    pred_regular_scaled = model_regular(sample_tensor).cpu().numpy()
    pred_swa_scaled = swa_model(sample_tensor).cpu().numpy()
    pred_regular = scaler_y.inverse_transform(pred_regular_scaled)
    pred_swa = scaler_y.inverse_transform(pred_swa_scaled)

print(f"Input: x1=1.5, x2=-2.0, x3=0.5, x4=1.0")
print(f"Regular Model Prediction: {pred_regular[0][0]:.4f}")
print(f"SWA Model Prediction:     {pred_swa[0][0]:.4f}")
print(f"Difference:               {abs(pred_regular[0][0] - pred_swa[0][0]):.4f}")
print("=" * 60)

print("\n" + "=" * 60)
print("PART 3B2 FINISHED - Comparison Complete")
print("=" * 60)