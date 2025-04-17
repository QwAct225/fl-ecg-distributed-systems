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
import json
import time
import random

HOST = '127.0.0.1'  # Changed to match server
PORT = 65432
KEY = b'Sixteen byte key'
MAX_CYCLES = 3  # Maximum number of federated learning cycles


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
    records = range(100, 109)
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

            rri = (peaks[i] - peaks[i - 1]) * (1000 / 360) + np.random.normal(0, 10)
            rri_channel = np.full(resample_size, rri / 1000)

            all_features.append(np.stack([window, rri_channel]))
            all_labels.append(labels[peaks[i]])
    
    indices = np.random.permutation(len(all_features))
    features = np.array(all_features)[indices]
    labels = np.array(all_labels)[indices]

    # Split into train, validation, test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))


def encrypt(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    encrypted_data = nonce + tag + ciphertext
    return encrypted_data


def decrypt(encrypted_data, key):
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
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


def train_local_model(model, train_loader, epochs=5, lr=0.001):
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

            if (i + 1) % 50 == 0:
                acc = 100 * correct / total
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch [{epoch + 1}/{epochs}] Batch [{i + 1}] '
                      f'Loss: {total_loss / (i + 1):.4f} '
                      f'Acc: {acc:.2f}% '
                      f'LR: {current_lr:.2e}')

        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1} Summary: '
              f'Loss: {total_loss / len(train_loader):.4f} '
              f'Acc: {epoch_acc:.2f}%')
    
    return model


def evaluate_model(model, loader):
    """Evaluasi model pada data validation atau test."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)

    class_accuracy = {}
    for label in range(5):
        if class_total[label] > 0:
            class_accuracy[str(label)] = 100 * class_correct[label] / class_total[label]
        else:
            class_accuracy[str(label)] = 0

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'class_accuracy': class_accuracy
    }


def add_noise(model, sensitivity, epsilon):
    """Menambahkan noise ke parameter model untuk differential privacy."""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * (sensitivity / epsilon)
            param.add_(noise)
    return model


def run_federated_client(client_id, cycle):
    """Jalankan proses federated learning untuk satu klien di satu siklus."""
    print(f"\n{'=' * 50}")
    print(f"🔄 SIKLUS FEDERATED LEARNING {cycle + 1}/{MAX_CYCLES}")
    print(f"{'=' * 50}")
    print(f"🖥️ Klien {client_id} memulai siklus {cycle + 1}")

    # Tentukan apakah harus menggunakan model global dari siklus sebelumnya
    use_previous_model = cycle > 0

    # Model lokal
    local_model = EcgResNet34(input_channels=2)

    # Jika bukan siklus pertama, muat model global dari siklus sebelumnya
    if use_previous_model:
        try:
            global_model_path = f'global_model_cycle_{cycle}.pt'
            local_model.load_state_dict(torch.load(global_model_path))
            print(f"✅ Model global dari siklus sebelumnya dimuat")
        except Exception as e:
            print(f"⚠️ Tidak dapat memuat model global sebelumnya: {e}")

    # Load data
    print(f"📊 Memuat dan mempersiapkan data untuk Klien {client_id}...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_mitbih_data(client_id)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training
    print(f"🏋️ Melatih model lokal untuk Klien {client_id}...")
    trained_model = train_local_model(local_model, train_loader, epochs=5)

    # Evaluasi model pada data validasi lokal
    print(f"📏 Mengevaluasi model pada data validasi lokal...")
    validation_metrics = evaluate_model(trained_model, valid_loader)
    print(f"📊 Validasi - Akurasi: {validation_metrics['accuracy']:.2f}%, Loss: {validation_metrics['loss']:.4f}")
    print(
        f"   Akurasi per kelas: {', '.join([f'Kelas {k}: {v:.2f}%' for k, v in validation_metrics['class_accuracy'].items()])}")

    # Tambahkan noise untuk differential privacy
    sensitivity = 0.1
    epsilon = 1.0
    print(f"🔒 Menerapkan differential privacy (ε={epsilon})...")
    trained_model = add_noise(trained_model, sensitivity, epsilon)

    # Simpan model
    local_model_path = f'local_model_client{client_id}_cycle{cycle + 1}.pt'
    torch.save(trained_model.state_dict(), local_model_path)
    print(f"💾 Model lokal disimpan sebagai {local_model_path}")

    # Enkripsi model untuk komunikasi dengan server
    encrypt_file(local_model_path, f'{local_model_path}.enc', KEY)

    # Komunikasi dengan server
    print(f"🔌 Menghubungi server untuk kirim model dan terima model global...")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(180)  # Extended timeout

            # Connect to server with retry logic
            max_retries = 5
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    s.connect((HOST, PORT))
                    print(f"✅ Terhubung ke server di {HOST}:{PORT}")
                    break
                except (socket.error, socket.timeout) as e:
                    if attempt < max_retries - 1:
                        print(
                            f"⚠️ Gagal terhubung, mencoba lagi dalam {retry_delay} detik... ({attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                    else:
                        raise ConnectionError(f"Tidak dapat terhubung ke server setelah {max_retries} percobaan")

            # Kirim model terlatih
            with open(local_model_path, 'rb') as f:
                model_bytes = f.read()
            encrypted_data = encrypt(model_bytes, KEY)

            # Send data length first
            s.sendall(len(encrypted_data).to_bytes(4, byteorder='big'))

            # Then send actual data
            s.sendall(encrypted_data)
            print(f"📤 Model lokal berhasil dikirim ke server")

            # Kirim metrik validasi
            encrypted_metrics = encrypt(json.dumps(validation_metrics).encode('utf-8'), KEY)

            # Send metrics length first
            s.sendall(len(encrypted_metrics).to_bytes(4, byteorder='big'))

            # Then send metrics data
            s.sendall(encrypted_metrics)
            print(f"📤 Metrik validasi berhasil dikirim ke server")

            # Terima model global
            print(f"⏳ Menunggu model global dari server...")

            # First receive data length
            length_data = s.recv(4)
            if not length_data:
                raise ConnectionError("Koneksi ditutup oleh server sebelum menerima panjang data")

            data_length = int.from_bytes(length_data, byteorder='big')
            print(f"Akan menerima {data_length / 1048576:.2f} MB dari server")

            # Then receive actual data with progress bar
            encrypted_response = b''
            received_length = 0
            bar_length = 50
            last_percent = -1

            print("Downloading global model: ", end='', flush=True)

            while received_length < data_length:
                chunk = s.recv(min(65536, data_length - received_length))
                if not chunk:
                    break
                encrypted_response += chunk
                received_length += len(chunk)

                # Update progress bar
                percent = int((received_length / data_length) * 100)
                if percent > last_percent:
                    bars = '=' * int((percent / 100) * bar_length)
                    spaces = ' ' * (bar_length - len(bars))
                    sys.stdout.write(
                        f"\rDownloading global model: [{bars}{spaces}] {percent}% ({received_length / 1048576:.2f}/{data_length / 1048576:.2f} MB)")
                    sys.stdout.flush()
                    last_percent = percent

            print()  # New line after progress bar

            if received_length < data_length:
                print(
                    f"⚠️ Penerimaan data tidak lengkap: {received_length / 1048576:.2f}/{data_length / 1048576:.2f} MB")

            if encrypted_response:
                global_model_path = f'global_model_cycle_{cycle + 1}.pt'
                decrypted_data = decrypt(encrypted_response, KEY)
                with open(global_model_path, 'wb') as f:
                    f.write(decrypted_data)
                print(f"✅ Model global diterima dan disimpan sebagai {global_model_path}")

                # Enkripsi model global untuk keamanan penyimpanan
                encrypt_file(global_model_path, f'{global_model_path}.enc', KEY)

                # Evaluasi model global pada data test lokal
                global_model = EcgResNet34()
                global_model.load_state_dict(torch.load(global_model_path))

                test_metrics = evaluate_model(global_model, test_loader)
                print(f"\n📊 Evaluasi model global pada data test lokal:")
                print(f"   Akurasi: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}")
                print(
                    f"   Akurasi per kelas: {', '.join([f'Kelas {k}: {v:.2f}%' for k, v in test_metrics['class_accuracy'].items()])}")

                # Simpan metrik
                metrics_file = f'client{client_id}_metrics_cycle{cycle + 1}.json'
                with open(metrics_file, 'w') as f:
                    json.dump({
                        'validation': validation_metrics,
                        'test': test_metrics
                    }, f, indent=4)

                print(f"💾 Metrik evaluasi disimpan sebagai {metrics_file}")

                return True
            else:
                print("❌ Tidak menerima data model global dari server")
                return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error: Gunakan perintah -> python client.py <client_id>")
        sys.exit(1)

    # Konversi file WFDB ke MATLAB
    print("🔄 Memeriksa dan mengkonversi file WFDB ke format MATLAB...")
    convert_all_wfdb_to_matlab()

    client_id = int(sys.argv[1])
    print(f"🖥️ Klien {client_id} dimulai...")

    # Jalankan federated learning untuk beberapa siklus
    for cycle in range(MAX_CYCLES):
        success = run_federated_client(client_id, cycle)

        if not success:
            print(f"⚠️ Siklus {cycle + 1} tidak berhasil. Mencoba lagi...")
            time.sleep(random.uniform(1, 5))  # Random delay before retry
            success = run_federated_client(client_id, cycle)

            if not success:
                print(f"❌ Gagal menyelesaikan siklus {cycle + 1} setelah percobaan ulang.")
                break

        if cycle < MAX_CYCLES - 1:
            delay = random.uniform(3, 8)  # Random delay between cycles
            print(f"\n⏳ Menunggu {delay:.1f} detik sebelum memulai siklus berikutnya...")
            time.sleep(delay)

    print(f"\n🏁 Klien {client_id} telah menyelesaikan {MAX_CYCLES} siklus federated learning.")