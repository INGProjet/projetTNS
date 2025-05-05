import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from tqdm import tqdm  # Pour les barres de progression

# Paramètres optimisés
WINDOW_SIZE = 100  # Réduit de 200 à 100 (suffisant pour la plupart des activités)
STEP_SIZE = 25     # Chevauchement de 75% conservé
MAX_SUBCARRIERS = 64  # Sélection aléatoire de 64 sous-porteuses sur 1026
TEST_SIZE = 0.2
RANDOM_STATE = 42

def create_windows(signal, window_size, step_size):
    """Version mémoire-optimisée du fenêtrage"""
    n_windows = (len(signal) - window_size) // step_size + 1
    return np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_windows, window_size, signal.shape[1]),
        strides=(signal.strides[0] * step_size, signal.strides[0], signal.strides[1])
    )

# Chargement mémoire-optimisé
def load_and_process(path):
    """Charge les fichiers par chunks et applique le prétraitement"""
    data = pd.read_csv(path, header=None, chunksize=10000)
    return np.concatenate([chunk.values for chunk in data])

base_path = r"../wifi_csi_har_dataset"
X_list, y_list = [], []

for room in tqdm(['room_1', 'room_2', 'room_3'], desc="Rooms"):
    room_path = os.path.join(base_path, room)
    for folder in tqdm(os.listdir(room_path), desc="Sessions", leave=False):
        folder_path = os.path.join(room_path, folder)
        if os.path.isdir(folder_path):
            try:
                # Chargement optimisé
                raw_data = load_and_process(os.path.join(folder_path, 'data.csv'))
                labels = load_and_process(os.path.join(folder_path, 'label.csv')).flatten()
                
                # Sélection aléatoire de sous-porteuses
                if raw_data.shape[1] > MAX_SUBCARRIERS:
                    subcarriers_idx = np.random.choice(raw_data.shape[1], MAX_SUBCARRIERS, replace=False)
                    raw_data = raw_data[:, subcarriers_idx]
                
                # Conversion amplitude + normalisation
                data = np.abs(raw_data) if np.iscomplexobj(raw_data) else raw_data
                data = StandardScaler().fit_transform(data)
                
                # Fenêtrage mémoire-optimisé
                windows = create_windows(data, WINDOW_SIZE, STEP_SIZE)
                label_windows = [pd.Series(labels[i:i+WINDOW_SIZE]).mode()[0] 
                               for i in range(0, len(labels) - WINDOW_SIZE + 1, STEP_SIZE)]
                
                # Stockage partiel
                min_len = min(len(windows), len(label_windows))
                X_list.append(windows[:min_len])
                y_list.extend(label_windows[:min_len])
            except Exception as e:
                print(f"Erreur dans {folder_path}: {str(e)}")
                continue

# Concaténation finale
X = np.concatenate(X_list)
y = np.array(y_list)

# Encodage et split
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)

# Équilibrer les classes (mémoire-optimisé)
unique, counts = np.unique(y_train, return_counts=True)
max_samples = counts.max()
X_resampled, y_resampled = [], []

for class_idx in unique:
    mask = y_train == class_idx
    X_class = X_train[mask]
    repeat_times = max_samples // len(X_class)
    remainder = max_samples % len(X_class)
    
    X_resampled.append(np.concatenate([np.repeat(X_class, repeat_times, axis=0),
                                      X_class[:remainder]]))
    y_resampled.append(np.full(max_samples, class_idx))

X_train = np.concatenate(X_resampled)
y_train = np.concatenate(y_resampled)

# Nettoyage mémoire
del X_list, y_list, X_resampled, y_resampled
import gc; gc.collect()

import pandas as pd

# Convertir les résultats en DataFrame pour une analyse claire
df_train = pd.DataFrame({
    'Classe': encoder.classes_[y_train],
    'Valeur_Encodée': y_train
})
df_test = pd.DataFrame({
    'Classe': encoder.classes_[y_test],
    'Valeur_Encodée': y_test
})

# Afficher les distributions de manière professionnelle
print("=== DISTRIBUTION DES CLASSES ===")
print("\n**Train set** (après oversampling):")
train_dist = df_train['Classe'].value_counts().sort_index()
print(train_dist)

print("\n**Test set** (distribution originale):")
test_dist = df_test['Classe'].value_counts().sort_index()
print(test_dist)

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=train_dist.index, y=train_dist.values, palette="viridis")
plt.title("Distribution des classes (Train)\nAprès équilibrage")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x=test_dist.index, y=test_dist.values, palette="magma")
plt.title("Distribution des classes (Test)\nDistribution originale")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()