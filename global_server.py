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

HOST = '127.0.0.1'
PORT = 65432
KEY = b'Sixteen byte key'

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

def handle_client(conn, addr, global_model, client_models, client_weights, client_index):
    print(f"📡 Koneksi dari {addr}")

    try:
        encrypted_data = b''
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            encrypted_data += chunk

        if not encrypted_data:
            raise ValueError("Data kosong")

        decrypted_data = decrypt(encrypted_data, KEY)
        with open('temp_model.pt', 'wb') as f:
            f.write(decrypted_data)

        client_model = EcgResNet34(input_channels=2)
        client_model.load_state_dict(torch.load('temp_model.pt', map_location='cpu'))

        client_models[client_index] = client_model

        if all(model is not None for model in client_models):
            print("\n🔀 Memulai Federated Averaging")
            start_agg = time.time()
            
            federated_averaging(global_model, client_models, client_weights)
            
            global_model_path = 'global_model.pt'
            torch.save(global_model.state_dict(), global_model_path)
            
            encrypted_model_path = 'global_model.pt.enc'
            encrypt_file(global_model_path, encrypted_model_path, KEY)
            
            with open(global_model_path, 'rb') as f:
                global_bytes = f.read()
            
            print(f"⏱ Waktu Agregasi: {time.time()-start_agg:.2f} detik")
            print(f"📤 Mengirim model global ke {addr}")
            
            encrypted_response = encrypt(global_bytes, KEY)
            conn.sendall(encrypted_response)

    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        conn.close()


if __name__ == '__main__':
    global_model = EcgResNet34(input_channels=2)

    num_clients = 2
    client_models = [None] * num_clients
    client_weights = [0.5, 0.5]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(120)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server mendengarkan di {HOST}:{PORT}")

        client_count = 0
        while client_count < num_clients:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(
            conn, addr, global_model, client_models, client_weights, client_count))
            client_thread.start()
            client_count += 1
        print("Semua klien terhubung, menunggu pembaruan model...")