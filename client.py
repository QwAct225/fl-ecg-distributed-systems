# client.py
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
from scipy.signal import find_peaks
import pickle
import Crypto
from Crypto.Cipher import AES
import os
import wfdb

# Konfigurasi Klien
HOST = '127.0.0.1' # Loopback address (harus sama dengan server)
PORT = 65432 # Port server (harus sama dengan server)
KEY = b'Sixteen byte key' # Kunci Enkripsi (harus sama dengan server!)

# Definisi Model (Harus sama dengan server!)
# Di global_server.py dan client.py
class EcgResNet34(nn.Module):
    def __init__(self, input_channels=1):
        super(EcgResNet34, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Blok Residual
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 5)  # 5 kelas

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

# Fungsi untuk memuat dan memproses data MIT-BIH
def load_mitbih_data(client_id, resample_size=128):
    record = wfdb.rdrecord('mit-bih/100', channels=[0])
    signal = record.p_signal[:, 0]
    annotations = wfdb.rdann('mit-bih/100', 'atr')

    # Ekstrak label (5 kelas: N, L, R, V, A)
    label_map = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}
    labels = np.zeros(len(signal), dtype=int)
    for ann_sample, ann_symbol in zip(annotations.sample, annotations.symbol):
        if ann_symbol in label_map and ann_sample < len(labels):
            labels[ann_sample] = label_map[ann_symbol]

    # Pembagian data untuk klien
    if client_id == 1:
        signal = signal[:len(signal)//2]
        labels = labels[:len(labels)//2]
    else:
        signal = signal[len(signal)//2:]
        labels = labels[len(signal)//2:]

    # Normalisasi
    signal = (signal - np.mean(signal)) / np.std(signal)

    # Deteksi Puncak R dan Ekstraksi Window Sinyal Mentah
    peaks, _ = find_peaks(signal, distance=50)
    features = []
    feature_labels = []
    for peak_index in peaks:
        start = max(0, peak_index - resample_size//2)
        end = min(len(signal), peak_index + resample_size//2)
        window = signal[start:end]

        # Padding atau Truncate ke 128 titik
        if len(window) < resample_size:
            window = np.pad(window, (0, resample_size - len(window)))
        elif len(window) > resample_size:
            window = window[:resample_size]
        features.append(window)
        feature_labels.append(labels[peak_index])

    # Reshape ke [samples, 1, 128] untuk Conv1d
    features = np.array(features).reshape(-1, 1, resample_size)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(feature_labels)

# Fungsi untuk Mengenkripsi Data
def encrypt(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce + ciphertext + tag

# Fungsi untuk Mendekripsi Data
def decrypt(encrypted_data, key):
    nonce = encrypted_data[:16]
    tag = encrypted_data[-16:]
    ciphertext = encrypted_data[16:-16]
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

# Fungsi untuk melatih model lokal
def train_local_model(model, train_loader, epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float()  # Hapus unsqueeze(0)
            labels = torch.as_tensor(labels).long()  # Ganti torch.tensor dengan torch.as_tensor

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

    return model

# Fungsi untuk menambahkan noise (Differential Privacy sederhana)
def add_noise(model, sensitivity, epsilon):
    """Menambahkan noise ke parameter model untuk differential privacy."""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * (sensitivity / epsilon)
            param.add_(noise)
    return model

if __name__ == '__main__':
    client_id = 1 # Ganti dengan ID klien yang sesuai (1 atau 2)
    print(f"Klien {client_id} dimulai...")

    # Muat Data Lokal
    features, labels = load_mitbih_data(client_id)
    
    # Ubah data menjadi format yang sesuai untuk PyTorch
    X_train = features.clone().detach()  # Gunakan clone().detach()
    y_train = labels.clone().detach()

    # Buat DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Inisialisasi Model Lokal
    local_model = EcgResNet34()

    # Koneksi ke Server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            print(f"Terhubung ke server di {HOST}:{PORT}")

            # Latih Model Lokal
            trained_model = train_local_model(local_model, train_loader)

            # Tambahkan Noise (Differential Privacy - Sederhana!)
            sensitivity = 0.1 # Sesuaikan
            epsilon = 1.0 # Sesuaikan
            trained_model = add_noise(trained_model, sensitivity, epsilon)

            # Kirim Model ke Server
            model_data = torch.save(trained_model.state_dict(), 'temp_model.pt')
            with open('temp_model.pt', 'rb') as f:
                model_bytes = f.read()
            encrypted_data = encrypt(model_bytes, KEY)

            # Kirim data terenkripsi ke server
            s.sendall(encrypted_data)

            encrypted_response = b''
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                encrypted_response += chunk
                
            # Deskripsi Respon
            decrypted_data = decrypt(encrypted_response, KEY)
            with open('global_model.pt', 'wb') as f:
                f.write(decrypted_data)
            updated_global_model = EcgResNet34()
            updated_global_model.load_state_dict(torch.load('global_model.pt'))

        except Exception as e:
            print(f"Error: {e}")
        finally:
            s.close()
    print(f"Klien {client_id} selesai.")