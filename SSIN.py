import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

# GPU am HPC suchen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"GPU gefunden.")
else:
    print("GPU NICHT gefunden.")


#MD5 Funktion
#Rotationsfunktion
def left_rotate(x, amount):
    left_part = x << amount
    right_part = x >> (32 - amount)
    rotiert = left_part | right_part
    return rotiert & 0xFFFFFFFF

# Register Update aus Skript
def md5_step(A, B, C, D, M, s, K):
    F = (B & C) | (~B & D)
    sum = (A + F + M + K) & 0xFFFFFFFF
    new_B = (B + left_rotate(sum, s)) & 0xFFFFFFFF
    return D, new_B, B, C


def to_bits(value):
    bit_liste = []
    # MD5 Register sind 32 Bit lang
    for i in range(32):
        shifted = value >> i
        einzelnes_bit = shifted & 1
        bit_liste.append(einzelnes_bit)
    return bit_liste


def state_to_bits(A, B, C, D):
    bits_A = to_bits(A)
    bits_B = to_bits(B)
    bits_C = to_bits(C)
    bits_D = to_bits(D)
    alle_bits = bits_A + bits_B + bits_C + bits_D
    return torch.tensor(alle_bits, dtype=torch.float32) # wie bei np.array für das Netzwerk


# Daten generieren für 8 Schritte
def generate_data(num_samples):
    #Initial Konstanten
    K = [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
         0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501]
    S = [7, 12, 17, 22, 7, 12, 17, 22]

    S0_list, S1_list, S2_list, S3_list, S4_list, S5_list, S6_list, S7_list, S8_list = [], [], [], [], [], [], [], [], []

    for _ in range(num_samples):
        # Zufälliger Startzustand (S0)
        A0 = random.randint(0, 0xFFFFFFFF)
        B0 = random.randint(0, 0xFFFFFFFF)
        C0 = random.randint(0, 0xFFFFFFFF)
        D0 = random.randint(0, 0xFFFFFFFF)

        # 8 zufällige Klartext-Blöcke
        M0 = random.randint(0, 0xFFFFFFFF)
        M1 = random.randint(0, 0xFFFFFFFF)
        M2 = random.randint(0, 0xFFFFFFFF)
        M3 = random.randint(0, 0xFFFFFFFF)
        M4 = random.randint(0, 0xFFFFFFFF)
        M5 = random.randint(0, 0xFFFFFFFF)
        M6 = random.randint(0, 0xFFFFFFFF)
        M7 = random.randint(0, 0xFFFFFFFF)

        # 8 MD5 Schritte
        A1, B1, C1, D1 = md5_step(A0, B0, C0, D0, M0, S[0], K[0])
        A2, B2, C2, D2 = md5_step(A1, B1, C1, D1, M1, S[1], K[1])
        A3, B3, C3, D3 = md5_step(A2, B2, C2, D2, M2, S[2], K[2])
        A4, B4, C4, D4 = md5_step(A3, B3, C3, D3, M3, S[3], K[3])
        A5, B5, C5, D5 = md5_step(A4, B4, C4, D4, M4, S[4], K[4])
        A6, B6, C6, D6 = md5_step(A5, B5, C5, D5, M5, S[5], K[5])
        A7, B7, C7, D7 = md5_step(A6, B6, C6, D6, M6, S[6], K[6])
        A8, B8, C8, D8 = md5_step(A7, B7, C7, D7, M7, S[7], K[7])

        S0_list.append(state_to_bits(A0, B0, C0, D0))
        S1_list.append(state_to_bits(A1, B1, C1, D1))
        S2_list.append(state_to_bits(A2, B2, C2, D2))
        S3_list.append(state_to_bits(A3, B3, C3, D3))
        S4_list.append(state_to_bits(A4, B4, C4, D4))
        S5_list.append(state_to_bits(A5, B5, C5, D5))
        S6_list.append(state_to_bits(A6, B6, C6, D6))
        S7_list.append(state_to_bits(A7, B7, C7, D7))
        S8_list.append(state_to_bits(A8, B8, C8, D8))

    return (torch.stack(S0_list), torch.stack(S1_list), torch.stack(S2_list),
            torch.stack(S3_list), torch.stack(S4_list), torch.stack(S5_list),
            torch.stack(S6_list), torch.stack(S7_list), torch.stack(S8_list))


#Sequential State Inversion Network
class SSIN(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            # Input Layer
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # erstes Hidden Layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # zweites Hidden Layer
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            # Output Layer
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


#Netzwerk trainieren
def train_single_step(model, X_train, Y_train, X_val, Y_val, step_name, epochs=1000, patience=25):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    batch_size = 256
    dataset_size = X_train.shape[0]
    val_dataset_size = X_val.shape[0]

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0

        # Trainingsschleife
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = X_train[indices].to(device)
            batch_Y = Y_train[indices].to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            hard_preds = (preds > 0.5).float()
            epoch_acc += (hard_preds == batch_Y).float().mean().item()
            batches += 1

        train_loss = epoch_loss / batches
        train_acc = epoch_acc / batches

        # Validation schleife
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, val_dataset_size, batch_size):
                batch_X_val = X_val[i:i + batch_size].to(device)
                batch_Y_val = Y_val[i:i + batch_size].to(device)

                val_preds = model(batch_X_val)
                v_loss = criterion(val_preds, batch_Y_val)
                val_loss += v_loss.item()

                hard_val_preds = (val_preds > 0.5).float()
                val_acc += (hard_val_preds == batch_Y_val).float().mean().item()
                val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[{step_name}] Epoch {epoch + 1:03d}/{epochs} | Loss: {train_loss:.4f} - Acc: {train_acc * 100:.2f}% | Val-Loss: {val_loss:.4f} - Val-Acc: {val_acc * 100:.2f}%")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly Stopping Triggered after Epoch {epoch + 1}")
                print(f"No improvement in Validation Loss for {patience} epochs.")
                break
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    return model, history


# plots für visualisierung
def plot_8_step_history(h1, h2, h3, h4, h5, h6, h7, h8):
    epochs_1 = range(len(h1['loss']))
    epochs_2 = range(len(h2['loss']))
    epochs_3 = range(len(h3['loss']))
    epochs_4 = range(len(h4['loss']))
    epochs_5 = range(len(h5['loss']))
    epochs_6 = range(len(h6['loss']))
    epochs_7 = range(len(h7['loss']))
    epochs_8 = range(len(h8['loss']))

    fig, axs = plt.subplots(8, 2, figsize=(15, 32))

    histories = [
        (h1, epochs_1, "Network 1 (S1 to S0)"),
        (h2, epochs_2, "Network 2 (S2 to S1)"),
        (h3, epochs_3, "Network 3 (S3 to S2)"),
        (h4, epochs_4, "Network 4 (S4 to S3)"),
        (h5, epochs_5, "Network 5 (S5 to S4)"),
        (h6, epochs_6, "Network 6 (S6 to S5)"),
        (h7, epochs_7, "Network 7 (S7 to S6)"),
        (h8, epochs_8, "Network 8 (S8 to S7)")
    ]

    for i, (hist, epochs, title) in enumerate(histories):
        # Loss
        axs[i, 0].plot(epochs, hist['loss'], label='Train Loss', color='blue')
        axs[i, 0].plot(epochs, hist['val_loss'], label='Val Loss', color='orange', linestyle='--')
        axs[i, 0].set_title(f'Convergence of BCE Loss ({title})')
        axs[i, 0].set_ylabel('BCE Loss')
        axs[i, 0].grid(True, alpha=0.3)
        axs[i, 0].legend()

        # Accuracy
        axs[i, 1].plot(epochs, [a * 100 for a in hist['acc']], label='Train Acc', color='green')
        axs[i, 1].plot(epochs, [a * 100 for a in hist['val_acc']], label='Val Acc', color='lightgreen', linestyle='--')
        axs[i, 1].set_title(f'Validation Accuracy ({title})')
        axs[i, 1].set_ylabel('Accuracy (%)')
        axs[i, 1].grid(True, alpha=0.3)
        axs[i, 1].legend()

    plt.tight_layout()
    plt.savefig('ssin_training_history.png')


def plot_wunsch_array(soft_S7, true_S7, soft_S6, true_S6, soft_S5, true_S5, soft_S4, true_S4, soft_S3, true_S3, soft_S2,
                      true_S2, soft_S1, true_S1, soft_S0, true_S0):
    plt.figure(figsize=(16, 36))

    plots_data = [
        (soft_S7, true_S7, "Inference Network 8: Predicted Probabilities for State S7"),
        (soft_S6, true_S6, "Cascade Inference (Net8 -> Net7): Predicted Probabilities for State S6"),
        (soft_S5, true_S5, "Cascade Inference (Net8 -> ... -> Net6): Predicted Probabilities for State S5"),
        (soft_S4, true_S4, "Cascade Inference (Net8 -> ... -> Net5): Predicted Probabilities for State S4"),
        (soft_S3, true_S3, "Cascade Inference (Net8 -> ... -> Net4): Predicted Probabilities for State S3"),
        (soft_S2, true_S2, "Cascade Inference (Net8 -> ... -> Net3): Predicted Probabilities for State S2"),
        (soft_S1, true_S1, "Cascade Inference (Net8 -> ... -> Net2): Predicted Probabilities for State S1"),
        (soft_S0, true_S0, "Full Cascade Inference (Net8 -> ... -> Net1): Predicted Probabilities for State S0")
    ]

    for i, (soft, true, title) in enumerate(plots_data):
        hard = (soft > 0.5).astype(int)

        plt.subplot(8, 1, i + 1)
        plt.bar(range(128), soft, color='blue', alpha=0.5, label='Model Prediction (Soft Bits)')
        plt.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
        plt.stem(range(128), true, linefmt='g-', markerfmt='go', basefmt=' ', label='Ground Truth')
        plt.stem(range(128), hard, linefmt='r--', markerfmt='rx', basefmt=' ', label='Model Decision (Hard Bits)')

        plt.title(title)
        plt.ylabel('Probability / Bit Value')
        plt.xlim(-1, 128)
        if i == 7: plt.xlabel('Bit Position (0-127)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('ssin_single_inferenz_test.png')


#Ausführung im Main skript
if __name__ == "__main__":
    SAMPLES = 500000
    VAL_SAMPLES = 50000

    #generiere trainingsdaten
    S0_tr, S1_tr, S2_tr, S3_tr, S4_tr, S5_tr, S6_tr, S7_tr, S8_tr = generate_data(SAMPLES)

    #generiere validierungsdaten
    S0_val, S1_val, S2_val, S3_val, S4_val, S5_val, S6_val, S7_val, S8_val = generate_data(VAL_SAMPLES)

    #training
    net_step1 = SSIN()
    net_step1, h1 = train_single_step(
        net_step1, S1_tr, S0_tr, S1_val, S0_val, "Network 1 (S1 to S0)", epochs=1000, patience=25
    )

    net_step2 = SSIN()
    net_step2, h2 = train_single_step(
        net_step2, S2_tr, S1_tr, S2_val, S1_val, "Network 2 (S2 to S1)", epochs=1000, patience=25
    )

    net_step3 = SSIN()
    net_step3, h3 = train_single_step(
        net_step3, S3_tr, S2_tr, S3_val, S2_val, "Network 3 (S3 to S2)", epochs=1000, patience=25
    )

    net_step4 = SSIN()
    net_step4, h4 = train_single_step(
        net_step4, S4_tr, S3_tr, S4_val, S3_val, "Network 4 (S4 to S3)", epochs=1000, patience=25
    )

    net_step5 = SSIN()
    net_step5, h5 = train_single_step(
        net_step5, S5_tr, S4_tr, S5_val, S4_val, "Network 5 (S5 to S4)", epochs=1000, patience=25
    )

    net_step6 = SSIN()
    net_step6, h6 = train_single_step(
        net_step6, S6_tr, S5_tr, S6_val, S5_val, "Network 6 (S6 to S5)", epochs=1000, patience=25
    )

    net_step7 = SSIN()
    net_step7, h7 = train_single_step(
        net_step7, S7_tr, S6_tr, S7_val, S6_val, "Network 7 (S7 to S6)", epochs=1000, patience=25
    )

    net_step8 = SSIN()
    net_step8, h8 = train_single_step(
        net_step8, S8_tr, S7_tr, S8_val, S7_val, "Network 8 (S8 to S7)", epochs=1000, patience=25
    )

    # History Plotten
    plot_8_step_history(h1, h2, h3, h4, h5, h6, h7, h8)

    #inferenztest: single inferenztest
    S0_test, S1_test, S2_test, S3_test, S4_test, S5_test, S6_test, S7_test, S8_test = generate_data(1) #Nur einen Wert generieren

    net_step1.eval()
    net_step2.eval()
    net_step3.eval()
    net_step4.eval()
    net_step5.eval()
    net_step6.eval()
    net_step7.eval()
    net_step8.eval()

    with torch.no_grad():
        # 8er Kaskade durchlaufen
        pred_S7_soft = net_step8(S8_test.to(device))
        pred_S6_soft = net_step7(pred_S7_soft)
        pred_S5_soft = net_step6(pred_S6_soft)
        pred_S4_soft = net_step5(pred_S5_soft)
        pred_S3_soft = net_step4(pred_S4_soft)
        pred_S2_soft = net_step3(pred_S3_soft)
        pred_S1_soft = net_step2(pred_S2_soft)
        pred_S0_soft = net_step1(pred_S1_soft)

        soft7 = pred_S7_soft[0].cpu().numpy()
        soft6 = pred_S6_soft[0].cpu().numpy()
        soft5 = pred_S5_soft[0].cpu().numpy()
        soft4 = pred_S4_soft[0].cpu().numpy()
        soft3 = pred_S3_soft[0].cpu().numpy()
        soft2 = pred_S2_soft[0].cpu().numpy()
        soft1 = pred_S1_soft[0].cpu().numpy()
        soft0 = pred_S0_soft[0].cpu().numpy()

    plot_wunsch_array(
        soft7, S7_test[0].numpy(),
        soft6, S6_test[0].numpy(),
        soft5, S5_test[0].numpy(),
        soft4, S4_test[0].numpy(),
        soft3, S3_test[0].numpy(),
        soft2, S2_test[0].numpy(),
        soft1, S1_test[0].numpy(),
        soft0, S0_test[0].numpy()
    )

    #trainingsdatensatz inferenz
    S0_tb, S1_tb, S2_tb, S3_tb, S4_tb, S5_tb, S6_tb, S7_tb, S8_tb = generate_data(10000)

    with torch.no_grad():
        out_S7 = net_step8(S8_tb.to(device))
        out_S6 = net_step7(out_S7)
        out_S5 = net_step6(out_S6)
        out_S4 = net_step5(out_S5)
        out_S3 = net_step4(out_S4)
        out_S2 = net_step3(out_S3)
        out_S1 = net_step2(out_S2)
        out_S0 = net_step1(out_S1)

        final_S0_hard = (out_S0 > 0.5).float().cpu()

    bit_accuracy = (final_S0_hard == S0_tb).float().mean().item() * 100
    perfect_matches = torch.all(final_S0_hard == S0_tb, dim=1).sum().item()

    print(f"Kaskaden Bit-Genauigkeit (S0): {bit_accuracy:.2f} %")
    print(f"Perfekt rekonstruierte S0: {perfect_matches} von 10000")