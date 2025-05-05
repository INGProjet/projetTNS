import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aoa import OriginalAOA
from dataset import CSIDataset
from model import LSTMClassifier
from metrics import get_train_metric
from mealpy import FloatVar
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Device: {}".format(device))

input_dim = 468
output_dim = 7
batch_size = 16

model_path = "./saved_models/best_lstm_model.pth"
best_path = "./saved_models/best_model.pth"
data_path = [
        "./wifi_csi_har_dataset/room_1/1",
        "./wifi_csi_har_dataset/room_1/2",
        "./wifi_csi_har_dataset/room_1/3",
        "./wifi_csi_har_dataset/room_1/4",
        "./wifi_csi_har_dataset/room_2/1",
        "./wifi_csi_har_dataset/room_3/1",
        "./wifi_csi_har_dataset/room_3/2",
        "./wifi_csi_har_dataset/room_3/3",
        "./wifi_csi_har_dataset/room_3/4",
        "./wifi_csi_har_dataset/room_3/5"
    ]

train_dataset = CSIDataset([data_path[0]], window_size=1024, step=8)
val_dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class_weights = torch.Tensor([0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]).double().to(device)
class_weights_inv = 1 / class_weights

# Fonction objectif rapide pour optimiser
def objective_function(solution):
    hidden_dim = int(solution[0])
    num_layers = int(solution[1])
    dropout = float(solution[2])
    learning_rate = float(solution[3])

    # ⚠️ corriger dropout si 1 couche
    if num_layers == 1:
        dropout = 0.0

    model = LSTMClassifier(input_dim, hidden_dim, num_layers, dropout, False, output_dim, batch_size).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(weight=class_weights_inv)

    for epoch in range(2):
        model.train(mode=True)
        for x_batch, y_batch in train_loader:
            if x_batch.size(0) != batch_size:
                continue
            x_batch, y_batch = x_batch.double().to(device), y_batch.long().to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch.long())
            loss.backward()
            optimizer.step()

    _, _, _, val_acc = get_train_metric(model, val_loader, criterion, batch_size)
    _, _, _, train_acc = get_train_metric(model, train_loader, criterion, batch_size)
    tqdm.write(f"[AOA] hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout:.3f}, lr={learning_rate:.5f} → val_acc={val_acc:.4f}")
    return 1 - val_acc

# Définit les bornes des hyperparamètres
problem_dict = {
    "bounds": FloatVar(
        lb=[114, 1, 0.0, 0.0014],
        ub=[568, 3, 0.4, 0.0022],
        name=["hidden_dim", "num_layers", "dropout", "learning_rate"]
    ),
    "minmax": "min",
    "obj_func": objective_function
}

def train(new_model=False):
    print("Starting AOA optimization...")
    optimizer_aoa = OriginalAOA(epoch=30, pop_size=10, verbose=True)
    best_solution = optimizer_aoa.solve(problem_dict)
    best_params = best_solution.solution
    hidden_dim, num_layers, dropout, learning_rate = int(best_params[0]), int(best_params[1]), float(best_params[2]), float(best_params[3])
    dropout = 0.0 if num_layers == 1 else dropout

    print("Found best hyperparameters.")

    epochs_start, epochs_end = 0, 50
    patience, trials, best_acc = 10, 0, 0

    model = LSTMClassifier(input_dim, hidden_dim, num_layers, dropout, False, output_dim, batch_size).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not new_model:
        if os.path.exists(best_path):
            print("Loading saved datas...")
            checkpoint = torch.load(best_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs_start = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"Starting at epoch={epochs_start}, best_acc={best_acc:.4f}")
        else:
            print("No saves found, starting a new training...")
    else:
        print("Starting a new training...")

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_inv)


    print("Starting LSTM training...")
    print(f"Actual parameters: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout:.3f}, learning_rate={learning_rate}")
    for epoch in tqdm(range(epochs_start, epochs_end), desc="training epochs"):
        model.train()
        for x_batch, y_batch in train_loader:
            if x_batch.size(0) != batch_size:
                continue
            x_batch, y_batch = x_batch.double().to(device), y_batch.long().to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch.long())
            loss.backward()
            optimizer.step()

        val_loss, _, _, val_acc = get_train_metric(model, val_loader, criterion, batch_size)
        train_loss, _, _, train_acc = get_train_metric(model, train_loader, criterion, batch_size)
        tqdm.write(f"Epoch {epoch + 1}/{epochs_end} - Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f} - Train Loss: {train_loss:.2f}, Train Acc.: {train_acc:2.2%}")
        if val_acc > best_acc:
            trials = 0
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, best_path)
            tqdm.write(f"Epoch {epoch + 1} - Best model [SAVED] with acc: {best_acc:.4f}")
        else:
            trials += 1
            if trials >= patience:
                tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_loss)



    print("training completed.")
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    train()

