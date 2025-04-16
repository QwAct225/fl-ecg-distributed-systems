# FL-ECG Distributed Systems

A federated learning implementation for ECG classification using MIT-BIH dataset with distributed systems architecture.

## Overview

This project implements a federated learning system for ECG signal classification. The system consists of:

1. A central server that orchestrates the federated learning process
2. Multiple client nodes that train local models on their own ECG data
3. An aggregation mechanism that combines client models into a global model

The system is built to classify 5 types of ECG signals (Normal, Left Bundle Branch Block, Right Bundle Branch Block, Ventricular Premature Complex, and Atrial Premature Complex) using a ResNet34-based architecture.

## Features

- Distributed training across multiple clients
- Privacy-preserving with differential privacy noise addition
- Secure communication using AES encryption
- Data augmentation techniques for ECG signals
- MIT-BIH Arrhythmia Database integration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/QwAct225/fl-ecg-distributed-systems.git
   cd fl-ecg-distributed-systems
   ```

2. **Initialize & Run Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```
   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```bash
   python server.py
   ```

2. Start client 1 in a new terminal:
   ```bash
   python client.py 1
   ```

3. Start client 2 in another new terminal:
   ```bash
   python client.py 2
   ```

The system will automatically:
1. Convert WFDB records to MATLAB format
2. Load and preprocess ECG data for each client
3. Train local models on each client
4. Add differential privacy noise
5. Send encrypted models to the server
6. Perform federated averaging at the server
7. Return the updated global model to each client

## Security Features

- All model transfers are encrypted using AES in EAX mode
- Local differential privacy is applied before model sharing
- Models are saved in both encrypted and decrypted formats for analysis

## Project Structure

```
.
├── client.py              # Client implementation
├── server.py              # Server implementation
├── mit-bih/               # Directory for dataset files
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── *.pt, *.pt.enc         # Generated model files (after running)
```
