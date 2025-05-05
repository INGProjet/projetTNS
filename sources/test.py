import logging

import torch
from torch.utils.data import DataLoader

from dataset import CSIDataset
from model import LSTMClassifier
from tqdm import tqdm

# LSTM Model parameters
input_dim = 468  # 114 subcarriers * 4 antenna_pairs * 2 (amplitude + phase)
hidden_dim = 256
layer_dim = 2
output_dim = 7
dropout_rate = 0.0
bidirectional = False
SEQ_DIM = 1024
DATA_STEP = 8

BATCH_SIZE = 16
EPOCHS_NUM = 100
LEARNING_RATE = 0.00146

# device = torch.device("cpu")
#
# logging.basicConfig(level=logging.INFO)
# model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim, dropout_rate, bidirectional)
# model.load_state_dict(state_dict=torch.load("./saved_models/to_try.pth"))
# model = model.to(device)
# model = model.double()
#
# model.eval()

# LSTM Model parameters
input_dim = 468  # 114 subcarriers * 4 antenna_pairs * 2 (amplitude + phase)
hidden_dim = 256
layer_dim = 2
output_dim = 7
dropout_rate = 0.0
bidirectional = False
SEQ_DIM = 1024
DATA_STEP = 8

BATCH_SIZE = 16
EPOCHS_NUM = 100
LEARNING_RATE = 0.00146

device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
model = LSTMClassifier(input_dim, hidden_dim, layer_dim, dropout_rate, bidirectional, output_dim, BATCH_SIZE)
model = model.to(device)
model = model.double()

model.eval()

DATA_ARCH = [
             "./wifi_csi_har_dataset/room_1/1", "./wifi_csi_har_dataset/room_1/2", "./wifi_csi_har_dataset/room_1/3",
             "./wifi_csi_har_dataset/room_1/4", "./wifi_csi_har_dataset/room_2/1",
             "./wifi_csi_har_dataset/room_3/1", "./wifi_csi_har_dataset/room_3/2", "./wifi_csi_har_dataset/room_3/3",
             "./wifi_csi_har_dataset/room_3/4", "./wifi_csi_har_dataset/room_3/5"
             ]

val_dataset = CSIDataset(DATA_ARCH, 1024)

val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model.eval()
for i, (x_batch, y_batch) in tqdm(enumerate(val_dl), total=len(val_dl), desc="Testing epoch: "):
    if x_batch.size(0) != BATCH_SIZE:
        continue

    # model.hidden = model.init_hidden(x_batch.size(0))
    x_batch, y_batch = x_batch.double().to(device), y_batch.double().to(device)
    out = model.forward(x_batch)

    print("out: ", out.shape)
    print("out: ", torch.argmax(torch.nn.functional.log_softmax(out, dim=1), dim=1))
    print("y    : ", y_batch)

    preds = torch.nn.functional.log_softmax(out, dim=1).argmax(dim=1)