# global_server.py
import socket
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import pickle
import Crypto
from Crypto.Cipher import AES
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
import json

HOST = '127.0.0.1'
PORT = 65432
KEY = b'Sixteen byte key'
MAX_CYCLES = 3  # Maximum number of federated learning cycles


class EcgResNet34(nn.Module):
    def __init__(self, input_channels=2):
        super(EcgResNet34, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 5)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
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


def federated_averaging(global_model, client_models, client_weights):
    with torch.no_grad():
        global_params = OrderedDict(global_model.named_parameters())
        for name, param in global_params.items():
            param.data.zero_()

        for client_model, weight in zip(client_models, client_weights):
            client_params = OrderedDict(client_model.named_parameters())
            for name, param in global_params.items():
                param.data += client_params[name].data * weight

    return global_model


def load_server_test_data(resample_size=128):
    """Load central test set for server evaluation."""
    test_records = [106, 107, 108]  # Server test records

    all_features = []
    all_labels = []

    for record_id in test_records:
        try:
            data = loadmat(f"mit-bih/{record_id}.mat")
            signal = data['val'][0]
            labels = data['label'][0].astype(int)

            signal = (signal - np.mean(signal)) / np.std(signal)
            peaks, _ = find_peaks(signal, distance=50, prominence=0.5)

            for i in range(1, len(peaks)):
                start = max(0, peaks[i] - resample_size // 2)
                end = start + resample_size
                window = signal[start:end]

                if len(window) < resample_size:
                    window = np.pad(window, (0, resample_size - len(window)))
                else:
                    window = window[:resample_size]

                rri = (peaks[i] - peaks[i - 1]) * (1000 / 360)
                rri_channel = np.full(resample_size, rri / 1000)

                all_features.append(np.stack([window, rri_channel]))
                all_labels.append(labels[peaks[i]])
        except Exception as e:
            print(f"Error loading test record {record_id}: {e}")

    indices = np.random.permutation(len(all_features))
    return torch.tensor(np.array(all_features)[indices], dtype=torch.float32), \
        torch.tensor(np.array(all_labels)[indices], dtype=torch.long)


def evaluate_model(model, test_loader):
    """Evaluate model on given test data."""
    model.eval()
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
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
    class_accuracy = {label: 100 * class_correct[label] / max(1, class_total[label])
                      for label in class_correct}

    return accuracy, class_accuracy


def handle_client(conn, addr, global_model, client_models, client_data, client_index, cycle, metrics):
    print(f"📡 Koneksi dari {addr} (Klien {client_index + 1}, Siklus {cycle + 1})")

    try:
        # Receive client model
        encrypted_data = b''
        data_length = None

        # First receive the data length (4 bytes containing an integer)
        length_data = conn.recv(4)
        if length_data:
            data_length = int.from_bytes(length_data, byteorder='big')
            print(f"Expected data length: {data_length} bytes")

        # Now receive the actual data
        received_length = 0
        while received_length < data_length:
            chunk = conn.recv(min(65536, data_length - received_length))
            if not chunk:
                break
            encrypted_data += chunk
            received_length += len(chunk)
            print(f"Received {received_length}/{data_length} bytes")

        if not encrypted_data or received_length < data_length:
            raise ValueError(f"Incomplete data received: {received_length}/{data_length} bytes")

        # Decrypt client model
        decrypted_data = decrypt(encrypted_data, KEY)
        client_model_path = f'client_model_{client_index + 1}_cycle_{cycle + 1}.pt'
        with open(client_model_path, 'wb') as f:
            f.write(decrypted_data)

        # Load client model
        client_model = EcgResNet34(input_channels=2)
        client_model.load_state_dict(torch.load(client_model_path, map_location='cpu'))
        client_models[client_index] = client_model

        # Receive client validation metrics
        length_data = conn.recv(4)
        if length_data:
            metrics_length = int.from_bytes(length_data, byteorder='big')

            metrics_data = b''
            received_length = 0
            while received_length < metrics_length:
                chunk = conn.recv(min(4096, metrics_length - received_length))
                if not chunk:
                    break
                metrics_data += chunk
                received_length += len(chunk)

            if metrics_data:
                client_metrics = json.loads(decrypt(metrics_data, KEY).decode('utf-8'))
                metrics['client_validation'][client_index] = client_metrics
                print(f"📊 Metrik validasi dari Klien {client_index + 1}: Akurasi = {client_metrics['accuracy']:.2f}%")

        # Wait until all client models are received
        while None in client_models:
            time.sleep(0.5)

        # The last client to connect performs federated averaging
        if client_index == len(client_models) - 1:
            print("\n🔀 Memulai Federated Averaging")
            start_agg = time.time()

            federated_averaging(global_model, client_models, client_weights)

            global_model_path = f'global_model_cycle_{cycle + 1}.pt'
            torch.save(global_model.state_dict(), global_model_path)

            print(f"⏱ Waktu Agregasi: {time.time() - start_agg:.2f} detik")

            # Load test data for global evaluation
            test_features, test_labels = load_server_test_data()
            test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Evaluate global model on server test set
            server_accuracy, server_class_acc = evaluate_model(global_model, test_loader)
            print(f"🌐 Evaluasi Server: Akurasi Global = {server_accuracy:.2f}%")
            print(f"   Per-kelas: {', '.join([f'Kelas {k}: {v:.2f}%' for k, v in server_class_acc.items()])}")

            metrics['server_evaluation'][cycle] = {
                'accuracy': server_accuracy,
                'class_accuracy': server_class_acc
            }

            # Cross-silo validation (evaluate model on each client's data)
            if cycle % 1 == 0:  # Evaluate every cycle
                print("\n🔄 Memulai Cross-Silo Validation")
                for i, (client_valid_features, client_valid_labels) in enumerate(client_data):
                    valid_dataset = torch.utils.data.TensorDataset(client_valid_features, client_valid_labels)
                    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

                    cross_accuracy, cross_class_acc = evaluate_model(global_model, valid_loader)
                    print(f"   📊 Cross-Val pada data Klien {i + 1}: Akurasi = {cross_accuracy:.2f}%")

                    metrics['cross_silo'][cycle][i] = {
                        'accuracy': cross_accuracy,
                        'class_accuracy': cross_class_acc
                    }

            # Save metrics to file
            with open(f'federated_metrics_cycle_{cycle + 1}.json', 'w') as f:
                json.dump(metrics, f, indent=4)

            print(f"📤 Mengirim model global ke semua klien")

        # Send global model back to client
        with open(global_model_path, 'rb') as f:
            global_bytes = f.read()

        encrypted_response = encrypt(global_bytes, KEY)

        # Send data length first
        conn.sendall(len(encrypted_response).to_bytes(4, byteorder='big'))

        # Then send actual data
        conn.sendall(encrypted_response)

        print(f"✅ Model global berhasil dikirim ke Klien {client_index + 1}")

    except Exception as e:
        print(f"Error handling client {client_index + 1}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


def load_client_validation_data():
    """Create validation datasets for cross-silo validation."""
    client_datasets = []

    # Client 1 validation data
    client1_records = [100, 101, 102]
    client1_features, client1_labels = prepare_validation_data(client1_records)
    client_datasets.append((client1_features, client1_labels))

    # Client 2 validation data
    client2_records = [103, 104, 105]
    client2_features, client2_labels = prepare_validation_data(client2_records)
    client_datasets.append((client2_features, client2_labels))

    return client_datasets


def prepare_validation_data(records, resample_size=128):
    """Prepare validation data from records."""
    all_features = []
    all_labels = []

    for record_id in records:
        try:
            data = loadmat(f"mit-bih/{record_id}.mat")
            signal = data['val'][0]
            labels = data['label'][0].astype(int)

            signal = (signal - np.mean(signal)) / np.std(signal)
            peaks, _ = find_peaks(signal, distance=50, prominence=0.5)

            # Use last 20% of peaks for validation
            valid_peaks = peaks[int(len(peaks) * 0.8):]

            for i in range(1, len(valid_peaks)):
                start = max(0, valid_peaks[i] - resample_size // 2)
                end = start + resample_size
                window = signal[start:end]

                if len(window) < resample_size:
                    window = np.pad(window, (0, resample_size - len(window)))
                else:
                    window = window[:resample_size]

                rri = (valid_peaks[i] - valid_peaks[i - 1]) * (1000 / 360)
                rri_channel = np.full(resample_size, rri / 1000)

                all_features.append(np.stack([window, rri_channel]))
                all_labels.append(labels[valid_peaks[i]])
        except Exception as e:
            print(f"Error loading validation data for record {record_id}: {e}")

    if len(all_features) == 0:
        return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.long)

    indices = np.random.permutation(len(all_features))
    return torch.tensor(np.array(all_features)[indices], dtype=torch.float32), \
        torch.tensor(np.array(all_labels)[indices], dtype=torch.long)


if __name__ == '__main__':
    # Initialize global model
    global_model = EcgResNet34(input_channels=2)
    torch.save(global_model.state_dict(), 'global_model_initial.pt')

    num_clients = 2
    client_weights = [0.5, 0.5]  # Equal weights

    # Load validation data for cross-silo evaluation
    client_validation_data = load_client_validation_data()

    # Initialize metrics tracking
    metrics = {
        'client_validation': [{} for _ in range(num_clients)],
        'server_evaluation': {},
        'cross_silo': [{} for _ in range(MAX_CYCLES)]
    }

    # Run for multiple cycles
    for cycle in range(MAX_CYCLES):
        print(f"\n{'=' * 50}")
        print(f"🔄 FEDERATED LEARNING CYCLE {cycle + 1}/{MAX_CYCLES}")
        print(f"{'=' * 50}")

        client_models = [None] * num_clients

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(180)  # Extended timeout
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reuse of address
            s.bind((HOST, PORT))
            s.listen()
            print(f"🔌 Server mendengarkan di {HOST}:{PORT}")

            client_count = 0
            while client_count < num_clients:
                try:
                    conn, addr = s.accept()
                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(conn, addr, global_model, client_models,
                              client_validation_data, client_count, cycle, metrics)
                    )
                    client_thread.start()
                    client_count += 1
                except socket.timeout:
                    print("Timeout waiting for clients, continuing...")
                    break

            print("Semua klien terhubung, menunggu pembaruan model...")

            # Wait for all threads to complete
            main_thread = threading.current_thread()
            for thread in threading.enumerate():
                if thread is not main_thread:
                    thread.join(timeout=180)

        if cycle < MAX_CYCLES - 1:
            print(f"\n🔄 Menunggu siklus berikutnya ({cycle + 2}/{MAX_CYCLES})...")
            time.sleep(5)  # Wait before starting next cycle

    print("\n🏁 FEDERATED LEARNING COMPLETED 🏁")
    print(f"Model terakhir disimpan sebagai 'global_model_cycle_{MAX_CYCLES}.pt'")

    # Save final metrics
    with open('federated_metrics_final.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("📊 Metrik evaluasi disimpan dalam 'federated_metrics_final.json'")