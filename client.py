# client.py
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.io import loadmat, savemat
import numpy as np
from scipy.signal import find_peaks
import pickle
import Crypto
from Crypto.Cipher import AES
import os
import wfdb
import sys

HOST = '10.34.100.116'
PORT = 65432
KEY = b'Sixteen byte key' 

class EcgResNet34(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 5)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        layers += [ResidualBlock(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
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
        x = self.dropout(x)
        return self.fc(x)


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
        
def convert_all_wfdb_to_matlab():
    records = range(100, 106)
    for record_id in records:
        mat_file = f"mit-bih/{record_id}.mat"
        if os.path.exists(mat_file): continue
        
        try:
            record = wfdb.rdrecord(f'mit-bih/{record_id}', channels=[0])
            signal = record.p_signal[:, 0]
            ann = wfdb.rdann(f'mit-bih/{record_id}', 'atr')
            
            label_map = {'N':0, 'L':1, 'R':2, 'V':3, 'A':4}
            labels = np.zeros(len(signal), dtype=int)
            for s, sym in zip(ann.sample, ann.symbol):
                if sym in label_map and s < len(labels):
                    labels[s] = label_map[sym]
            
            assert len(np.unique(labels)) > 1, f"Record {record_id} hanya memiliki 1 kelas!"
            assert not np.all(signal == 0), f"Signal {record_id} korup!"
            
            savemat(mat_file, {'val': signal, 'label': labels})
            print(f"Converted {record_id}.mat")
            
        except Exception as e:
            print(f"Error converting {record_id}: {str(e)}")
            sys.exit(1)

def load_mitbih_data(client_id, resample_size=128):
    if client_id == 1:
        records = [100, 101, 102]
    else:
        records = [103, 104, 105]
    
    
    all_features = []
    all_labels = []
    
    for record_id in records:
        data = loadmat(f"mit-bih/{record_id}.mat")
        signal = data['val'][0]
        labels = data['label'][0].astype(int)
        
        signal = (signal - np.mean(signal)) / np.std(signal)
        peaks, _ = find_peaks(signal, distance=50, prominence=0.5)
        
        for i in range(1, len(peaks)):
            scale_factor = np.random.uniform(0.8, 1.2)
            scaled_signal = signal * scale_factor
            
            start = max(0, peaks[i] - resample_size//2 + np.random.randint(-5,5))
            end = start + resample_size
            window = scaled_signal[start:end]
            
            if len(window) < resample_size:
                window = np.pad(window, (0, resample_size - len(window)))
            else:
                window = window[:resample_size]
            
            rri = (peaks[i] - peaks[i-1]) * (1000/360) + np.random.normal(0,10)
            rri_channel = np.full(resample_size, rri/1000)
            
            all_features.append(np.stack([window, rri_channel]))
            all_labels.append(labels[peaks[i]])
    
    indices = np.random.permutation(len(all_features))
    return torch.tensor(np.array(all_features)[indices], dtype=torch.float32), \
           torch.tensor(np.array(all_labels)[indices], dtype=torch.long)

def encrypt(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce + ciphertext + tag

def decrypt(encrypted_data, key):
    nonce = encrypted_data[:16]
    tag = encrypted_data[-16:]
    ciphertext = encrypted_data[16:-16]
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

def decrypt_file(input_file, output_file, key):
    """Mendekripsi file yang telah dienkripsi menggunakan AES."""
    try:
        with open(input_file, 'rb') as f:
            encrypted_data = f.read()
        
        if len(encrypted_data) < 32:
            raise ValueError("File terlalu kecil untuk menjadi file terenkripsi yang valid")
            
        nonce = encrypted_data[:16]
        tag = encrypted_data[-16:]
        ciphertext = encrypted_data[16:-16]
        
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
        print(f"File berhasil didekripsi: {output_file}")
        return True
    except Exception as e:
        print(f"Error dekripsi file: {e}")
        return False

def encrypt_file(input_file, output_file, key):
    """Mengenkripsi file menggunakan AES."""
    try:
        with open(input_file, 'rb') as f:
            data = f.read()
        
        cipher = AES.new(key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(data)
        
        with open(output_file, 'wb') as f:
            f.write(nonce + ciphertext + tag)
        print(f"File berhasil dienkripsi: {output_file}")
        return True
    except Exception as e:
        print(f"Error enkripsi file: {e}")
        return False

def train_local_model(model, train_loader, epochs=10, lr=0.001):
    model = model.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 
                                         base_lr=1e-4,
                                         max_lr=1e-3,
                                         step_size_up=500,
                                         cycle_momentum=False)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            lam = np.random.beta(0.2, 0.2) 
            index = torch.randperm(inputs.size(0))
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            outputs = model(mixed_inputs)
            
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (lam * (predicted == labels).float() + 
                      (1 - lam) * (predicted == labels[index]).float()).sum().item()
            
            if (i+1) % 50 == 0:
                acc = 100 * correct / total
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch [{epoch+1}/{epochs}] Batch [{i+1}] '
                      f'Loss: {total_loss/(i+1):.4f} '
                      f'Acc: {acc:.2f}% '
                      f'LR: {current_lr:.2e}')
        
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1} Summary: '
              f'Loss: {total_loss/len(train_loader):.4f} '
              f'Acc: {epoch_acc:.2f}%')
    
    return model

def add_noise(model, sensitivity, epsilon):
    """Menambahkan noise ke parameter model untuk differential privacy."""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * (sensitivity / epsilon)
            param.add_(noise)
    return model


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error: Gunakan perintah -> python client.py <client_id>")
        sys.exit(1)

    convert_all_wfdb_to_matlab() 

    client_id = int(sys.argv[1])
    print(f"Klien {client_id} dimulai...")

    features, labels = load_mitbih_data(client_id)

    X_train = features.clone().detach()
    y_train = labels.clone().detach()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    local_model = EcgResNet34()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(120)
            s.connect((HOST, PORT))
            print(f"Terhubung ke server di {HOST}:{PORT}")

            trained_model = train_local_model(local_model, train_loader)

            sensitivity = 0.1
            epsilon = 1.0
            trained_model = add_noise(trained_model, sensitivity, epsilon)

            local_model_path = f'local_model_client{client_id}.pt'
            torch.save(trained_model.state_dict(), local_model_path)
            encrypt_file(local_model_path, f'{local_model_path}.enc', KEY)

            with open(local_model_path, 'rb') as f:
                model_bytes = f.read()
            encrypted_data = encrypt(model_bytes, KEY)

            s.sendall(encrypted_data)

            encrypted_response = b''
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                encrypted_response += chunk

            global_model_path = f'global_model.pt'
            decrypted_data = decrypt(encrypted_response, KEY)
            with open(global_model_path, 'wb') as f:
                f.write(decrypted_data)
                
            encrypt_file(global_model_path, f'{global_model_path}.enc', KEY)

            updated_global_model = EcgResNet34()
            updated_global_model.load_state_dict(torch.load(global_model_path))

        except Exception as e:
            print(f"Error: {e}")
        finally:
            s.close()
    print(f"Klien {client_id} selesai.")