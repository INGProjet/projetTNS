import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Fonction pour créer des fenêtres
def create_windows(signal, window_size, step_size):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[start:start+window_size])
    return np.array(windows)

# Dossier où tu as extrait le dataset
base_path = r".\wifi_csi_har_dataset"


# Initialisation
all_windows = []
all_labels = []

# Parcours des dossiers Room
for room in ['room_1', 'room_2', 'room_3']:
    room_path = os.path.join(base_path, room)
    for folder in os.listdir(room_path):
        folder_path = os.path.join(room_path, folder)
        if os.path.isdir(folder_path):
            # Charger data.csv
            data_file = os.path.join(folder_path, 'data.csv')
            label_file = os.path.join(folder_path, 'label.csv')
            
            if os.path.exists(data_file) and os.path.exists(label_file):
                data = pd.read_csv(data_file, header=None).values  # (time_steps, subcarriers)
                labels = pd.read_csv(label_file, header=None).values.flatten()  # (time_steps, )

                # Normaliser chaque sous-porteuse individuellement
                scaler = MinMaxScaler()
                data = scaler.fit_transform(data)

                # Fenêtrer les données CSI
                window_size = 200
                step_size = 50
                windows = create_windows(data, window_size, step_size)

                # Fenêtrer les labels (prendre le label majoritaire dans la fenêtre)
                label_windows = []
                for start in range(0, len(labels) - window_size + 1, step_size):
                    window_labels = labels[start:start+window_size]
                    majority_label = pd.Series(window_labels).mode()[0]  # Prendre le label le plus fréquent
                    label_windows.append(majority_label)
            
                # Correction importante : ne prendre que le même nombre de fenêtres et de labels
                min_len = min(len(windows), len(label_windows))
                windows = windows[:min_len]
                label_windows = label_windows[:min_len]
                
                # Stocker
                all_windows.append(windows)
                all_labels.extend(label_windows)

# Mise en forme finale
X = np.vstack(all_windows)  # (nombre_total_fenêtres, window_size, nombre_subcarriers)
y = np.array(all_labels)

# Encodage des labels en entiers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Division en Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Classes: {encoder.classes_}")
