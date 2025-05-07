# interface.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import io
import threading
from dataset import CSIDataset, DATA_SUBROOMS, read_all_data_from_files
from lstm_aoa_train import train
from model.LSTMClassifier import LSTMClassifier
from metrics import get_train_metric
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuration de l'interface
st.set_page_config(
    page_title="CSI Training Interface",
    layout="wide",
    page_icon="📶"
)

# Style personnalisé
st.markdown("""
<style>
    .stSlider [data-baseweb="slider"] { margin: 15px 0; }
    .stProgress > div > div > div { background-color: #4a90e2; }
    .metric-box { padding: 20px; border-radius: 10px; background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

def get_valid_sessions(room):
    return list(map(int, DATA_SUBROOMS[room-1]))

def get_dataset_paths(base_path, room, session):
    valid_sessions = get_valid_sessions(room)
    return [os.path.join(base_path, f"room_{room}", str(session))]

def initialize_datasets(data_paths, batch_size):
    train_dataset = CSIDataset(data_paths, window_size=1024, step=8)
    val_dataset = train_dataset  # Using same dataset for validation in this implementation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

# Function to capture the output
def capture_output(func, *args, **kwargs):
    output_capture = io.StringIO()
    sys.stdout = output_capture
    sys.stderr = output_capture
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    return output_capture.getvalue()

# Function to simulate the training process
def simulate_training():
    return train(new_model=True, cpu=num_cores)

# Gestion de l'état de session
if 'model' not in st.session_state:
    st.session_state.model = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Interface utilisateur
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuration")

    num_cores = st.slider("Nombre de CPU cores", 1, 8, 4, 1,
                         help="Nombre de coeurs processeur à utiliser pour l'entraînement")
    use_aoa = st.checkbox("Optimisation AOA", value=True)
    batch_size = st.slider("Batch size", 2, 64, 16, 2)

    if not use_aoa:
        st.subheader("Paramètres LSTM")
        hidden_dim = st.slider("Hidden dim", 114, 568, 256, 10)
        num_layers = st.slider("Number of layers", 1, 3, 2)
        dropout = st.slider("Dropout", 0.0, 0.4, 0.3, 0.05)
        learning_rate = st.slider("Learning rate", 0.001, 0.005, 0.0022, 0.0001)

    dataset_option = st.radio("Source des données :",
                            ["Dossier local", "Upload manuel"])

    if dataset_option == "Dossier local":
        base_path = st.text_input("Chemin du dataset :", "./wifi_csi_har_dataset")
        room = st.slider("Numéro de la salle", 1, len(DATA_SUBROOMS), 1)
        session = st.selectbox("Numéro de la session", get_valid_sessions(room))
        data_paths = get_dataset_paths(base_path, room, session)
    else:
        uploaded_files = st.file_uploader("Choisir les fichiers CSI", accept_multiple_files=True)
        data_paths = [file.name for file in uploaded_files] if uploaded_files else []

    def callback(text, current, total, val_acc=None, val_loss=None):
        progress = int((current / total) * 100)
        st.session_state.progress = progress
        if val_acc is not None and val_loss is not None:
            st.write(f"{text} - Accuracy: {val_acc:.2f} - Loss: {val_loss:.4f}")
        else:
            st.write(text)
        st.progress(progress)

    if st.button("Démarrer l'entraînement"):
        if not data_paths:
            st.warning("Veuillez fournir des données pour l'entraînement.")
        else:
            try:
                st.write("Training started")

                # Capture training output
                output_area = st.empty()
                def run_training():
                    output_capture = io.StringIO()
                    sys.stdout = output_capture
                    sys.stderr = output_capture
                    try:
                        returned_model = simulate_training()
                        st.session_state.model = returned_model
                        st.success("Entraînement terminé avec succès!")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                    finally:
                        sys.stdout = sys.__stdout__
                        sys.stderr = sys.__stderr__
                        st.session_state.captured_output = output_capture.getvalue()
                        output_area.code(st.session_state.captured_output, language="text")


                # Run the training in a separate thread to avoid blocking the UI
                thread = threading.Thread(target=run_training)
                thread.start()

                if use_aoa:
                    # Training with AOA optimization
                    returned_model = train(new_model=True, cpu=num_cores)
                else:
                    # Training with manual parameters
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model = LSTMClassifier(
                        input_dim=468,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=False,
                        output_dim=7,
                        batch_size=batch_size
                    ).double().to(device)

                    # Custom training loop for manual parameters
                    # (You would need to implement this part)
                    pass

                if returned_model:
                    st.session_state.model = returned_model
                    st.success("Entraînement terminé avec succès!")

                    # Validation
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    train_loader, val_loader = initialize_datasets(data_paths, batch_size)
                    all_preds = []
                    all_labels = []
                    st.session_state.model.eval()
                    with torch.no_grad():
                        for x_batch, y_batch in val_loader:
                            x_batch = x_batch.double().to(device)
                            preds = torch.argmax(st.session_state.model(x_batch), dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(y_batch.numpy())

                    fig, ax = plt.subplots()
                    confusion_matrix = np.zeros((7, 7))
                    for true, pred in zip(all_labels, all_preds):
                        confusion_matrix[true][pred] += 1
                    ax.matshow(confusion_matrix, cmap='Blues')
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Erreur lors de l'entraînement : {str(e)}")
                st.error(f"Traceback: {sys.exc_info()}")

# Console output section
with col2:
    st.header("Console Output")
    console_output = st.empty()

    # Display captured output in the console section
    if 'captured_output' in st.session_state:
        console_output.code(st.session_state.captured_output, language="text")

# Section visualisation
st.divider()
with st.expander("Analyse des données brutes"):
    if data_paths:
        try:
            amps, phases, labels = read_all_data_from_files(data_paths)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Amplitudes")
                st.line_chart(amps[0][:100])
            with col2:
                st.subheader("Phases")
                st.line_chart(phases[0][:100])

            st.write(f"Nombre total d'échantillons : {len(amps)}")
            st.write("Répartition des classes :")
            classes, counts = np.unique(labels, return_counts=True)
            st.bar_chart(dict(zip(classes, counts)))

        except Exception as e:
            st.warning(f"Visualisation impossible : {str(e)}")

# Section téléchargement
st.divider()
if st.session_state.model:
    st.subheader("Export du modèle")
    model_path = "model.pth"
    torch.save(st.session_state.model.state_dict(), model_path)
    st.download_button(
        label="Télécharger le modèle entraîné",
        data=open(model_path, "rb").read(),
        file_name="csi_model.pth",
        mime="application/octet-stream"
    )