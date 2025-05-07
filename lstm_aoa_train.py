import logging
import sys

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

device = torch.device("cpu")

input_dim = 468
output_dim = 7
batch_size = 16
hidden_dim = 256
num_layers = 2
dropout = 0.3
learning_rate = 0.0022

spinner = ['|', '/', '—', '\\']
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


def objective_function(solution):
    num_layers = int(solution[0])
    dropout = float(solution[1])
    learning_rate = float(solution[2])

    if num_layers == 1:
        dropout = 0.0

    model = LSTMClassifier(input_dim, hidden_dim, num_layers, dropout, False, output_dim, batch_size).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(weight=class_weights_inv)
    j = 0
    for epoch in range(2):
        j+=1
        model.train(mode=True)
        i = 0
        for x_batch, y_batch in train_loader:
            if x_batch.size(0) != batch_size:
                continue
            x_batch, y_batch = x_batch.double().to(device), y_batch.long().to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch.long())
            loss.backward()
            optimizer.step()
            i += 1
            sys.stdout.write('\r' + spinner[i % len(spinner)] + f' Processing[{j}/4]...')
            sys.stdout.flush()

    _, _, _, val_acc = get_train_metric(model, val_loader, criterion, batch_size)
    _, _, _, train_acc = get_train_metric(model, train_loader, criterion, batch_size)
    tqdm.write(f"[AOA] => (num_layers={num_layers}, dropout={dropout:.3f}, lr={learning_rate:.5f}) -> val_acc={val_acc:.4f}")
    return 1 - val_acc

# Définit les bornes des hyperparamètres
problem_dict = {
    "bounds": FloatVar(
        lb=[1, 0.0, 0.0015],
        ub=[3, 0.3, 0.003],
        name=["num_layers", "dropout", "learning_rate"]
    ),
    "minmax": "min",
    "obj_func": objective_function
}

def train(optimize=True, layers=2, dropout_rate=0.0, lr=0.0015, batch=4, cores = 1, aoa_epoch=5, aoa_pop=10, lstm_epoch=30, callback=None):
    num_layers = layers
    dropout = dropout_rate
    learning_rate = lr
    if optimize:
        print("Starting AOA optimization...")
        optimizer_aoa = OriginalAOA(epoch=aoa_epoch, pop_size=aoa_pop, verbose=True)
        best_solution = optimizer_aoa.solve(problem_dict, mode='process', n_workers=cores)
        best_params = best_solution.solution
        layers, dropout_rate, lr = int(best_params[0]), float(best_params[1]), float(best_params[2])
        dropout_rate = 0.0 if layers == 1 else dropout_rate

        print("Found best hyperparameters.")

    epochs_start, epochs_end = 0, lstm_epoch
    patience, trials, best_acc = 10 if lstm_epoch > 15 else lstm_epoch/2 + 1, 0, 0


    model = LSTMClassifier(input_dim, hidden_dim, layers, dropout_rate, False, output_dim, batch).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # if not new_model:
    #     if os.path.exists(best_path):
    #         print("Loading saved datas...")
    #         checkpoint = torch.load(best_path)
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         epochs_start = checkpoint['epoch'] + 1
    #         best_acc = checkpoint['best_acc']
    #         print(f"Starting at epoch={epochs_start}, best_acc={best_acc:.4f}")
    #     else:
    #         print("No saves found, starting a new training...")
    # else:
    #     print("Starting a new training...")

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_inv)


    print("Starting LSTM training...")
    print(f"Actual parameters: num_layers={layers}, dropout={dropout_rate:.3f}, learning_rate={lr:.5f}")
    for epoch in range(epochs_start, epochs_end):
        model.train(mode=True)
        i = 0
        for x_batch, y_batch in train_loader:
            if x_batch.size(0) != batch:
                continue
            x_batch, y_batch = x_batch.double().to(device), y_batch.long().to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch.long())
            loss.backward()
            optimizer.step()
            i += 1
            sys.stdout.write('\r' + spinner[i % len(spinner)] + f' Processing[{i}/{len(train_loader)}]...')
            sys.stdout.flush()

        val_loss, _, _, val_acc = get_train_metric(model, val_loader, criterion, batch)
        train_loss, _, _, train_acc = get_train_metric(model, train_loader, criterion, batch)
        tqdm.write(f"Epoch {epoch + 1}/{epochs_end} - Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f} - Train Acc.: {train_acc:2.2%}, Train Loss: {train_loss:.2f}")
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

    logging.info("Device: {}".format(device))
    train()

