import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import string

# GPU des HPCs suchen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"GPU gefunden")
else:
    print("GPU NICHT gefunden")

# MD5 Konstanten
K_MD5 = [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
         0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
         0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
         0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821]
S_MD5 = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22]
IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)  # Initialwerte der Register

# MD5 Funktion:
# MD5 Padding
def padding(message):
    orig_len_in_bits = len(message) * 8
    padded = bytearray(message)
    padded.append(0x80) #erst 1 anhängen
    while len(padded) % 64 != 56: #solange 0 anhängen bis länge ein vielfaches von 64-8 ist
        padded.append(0x00)

    # Länge anhängen
    padded += orig_len_in_bits.to_bytes(8, byteorder='little')

    # Input in 16 32 Bit Listen schreiben
    words = []
    for i in range(0, 64, 4):
        four_bytes = padded[i: i + 4]
        word = int.from_bytes(four_bytes, byteorder='little')
        words.append(word)
    return words

# Rotationsfunktion
def left_rotate(x, amount):
    left_part = x << amount
    rigth_part = x >> (32 - amount)
    rotiert = left_part | rigth_part
    return rotiert & 0xFFFFFFFF

# Register update funktion aus Skript
def md5_step(A, B, C, D, M, s, K):
    F = (B & C) | (~B & D)
    sum = (A + F + M + K) & 0xFFFFFFFF
    new_B = (B + left_rotate(sum, s)) & 0xFFFFFFFF
    return D, new_B, B, C #rotiert zurückgeben

# Wort aus registern extrahieren
def extract_M_hard(A_prev, B_prev, C_prev, D_prev, B_curr, s, K):
    F = (B_prev & C_prev) | (~B_prev & D_prev)
    diff = (B_curr - B_prev) & 0xFFFFFFFF
    rotated = ((diff >> s) | (diff << (32 - s))) & 0xFFFFFFFF
    M = (rotated - A_prev - F - K) & 0xFFFFFFFF
    return M

#in bit Tensor umwandeln
def to_bits(val):
    bit_liste = []
    for i in range(32):
        shifted = val >> i
        einzelnes_bit = shifted & 1
        bit_liste.append(einzelnes_bit)
    return bit_liste


def state_to_bits(A, B, C, D):
    # Aus jedem Register eine 32-Bit-Liste machen
    bits_A = to_bits(A)
    bits_B = to_bits(B)
    bits_C = to_bits(C)
    bits_D = to_bits(D)
    #Zu langn Liste zusammenfügen
    alle_bits = bits_A + bits_B + bits_C + bits_D
    return torch.tensor(alle_bits, dtype=torch.float32) #In Tensor umwandeln

# Bits in Zahl umwandeln für Trainingsprozss
def bits_to_int32(bit_array):
    integer = 0
    for i in range(32):
        if bit_array[i] > 0.5:
            integer |= (1 << i)
    return integer


#Daten generieren: 16 schritte md5 ausführen, alle zustände und wörter speichern
def generate_bvp_data(num_samples):
    S_lists = [[] for _ in range(17)]  # S0 bis S16
    M_lists = [[] for _ in range(16)]  # M0 bis M15

    for _ in range(num_samples):
        #zufällige Texte generieren
        laenge = random.randint(12, 48)
        zufalls_buchstaben = random.choices(string.ascii_lowercase, k=laenge)
        rand_str = ''.join(zufalls_buchstaben)

        #in 16 words aufteilen mit richtgem padding
        words = padding(rand_str.encode('utf-8'))

        #aktueller Wortabschnitt
        M = [words[i] for i in range(16)]

        #MD5 Schritte
        A, B, C, D = IV
        S_lists[0].append(state_to_bits(A, B, C, D))

        for i in range(16):
            A, B, C, D = md5_step(A, B, C, D, M[i], S_MD5[i], K_MD5[i])
            S_lists[i + 1].append(state_to_bits(A, B, C, D))
            M_lists[i].append(torch.tensor(to_bits(M[i]), dtype=torch.float32))

    return tuple(torch.stack(lst) for lst in S_lists + M_lists)


#BVP Network erstellen
class BVPNetwork(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            #Input Layer
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            #erstes Hidden Layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            #zweites Hidden Layer
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            #Output Layer
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

    # Zustand mit Initialzustand verknüpfen
    def forward(self, current_state, target_state):
        x = torch.cat((current_state, target_state), dim=1)
        return self.net(x)


# Netzwerk trainieren
# Mischt die Daten in jeder Epoche per Zufallspermutation,
# teilt sie in Mini-Batches auf, berechnet die Vorhersage, den Loss,
# macht den Backpropagation-Schritt (loss.backward()) und aktualisiert die Gewichte (optimizer.step()).
def train_bvp_step(model, X_curr, X_target, Y_prev, X_val_curr, X_val_target, Y_val_prev, step_name, epochs=1000,
                   patience=25):
    print(f"Training {step_name}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    batch_size = 256
    dataset_size = X_curr.shape[0]

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(dataset_size)
        epoch_loss, epoch_acc = 0.0, 0.0
        batches = 0

        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_curr = X_curr[indices].to(device)
            batch_target = X_target[indices].to(device)
            batch_prev = Y_prev[indices].to(device)

            optimizer.zero_grad()
            preds = model(batch_curr, batch_target)
            loss = criterion(preds, batch_prev)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            hard_preds = (preds > 0.5).float()
            epoch_acc += (hard_preds == batch_prev).float().mean().item()
            batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_curr.to(device), X_val_target.to(device))
            val_loss = criterion(val_preds, Y_val_prev.to(device)).item()
            hard_val = (val_preds > 0.5).float()
            val_acc = (hard_val == Y_val_prev.to(device)).float().mean().item()

        # History speichern
        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches

        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[{step_name}] Epoch {epoch + 1:04d} | Train Acc: {avg_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | Val Loss: {val_loss:.4f} | Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} for {step_name}.")
            break

    print(f"Loading best model weights (Val Loss: {best_val_loss:.4f})\n")
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, history


#Plots zur visualisierung im Latex stil
def plot_bvp_history(histories):

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    })

    #in 4 gruppen aufteilen, damit es besser auf eine Seite passt
    group_size = 4
    for group_idx in range(4):
        fig, axs = plt.subplots(group_size, 2, figsize=(5.8, 6.0), sharex=True)

        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_histories = histories[start_idx:end_idx]

        for i, (hist, title) in enumerate(group_histories):
            eps = range(len(hist['loss']))

            #linke spalte: Loss
            axs[i, 0].plot(eps, hist['loss'], label='Train', color='royalblue', lw=1.2)
            axs[i, 0].plot(eps, hist['val_loss'], label='Val', color='darkorange', linestyle='--', lw=1.2)
            axs[i, 0].set_title(f'Loss: {title}', pad=3, fontsize=9)
            axs[i, 0].set_ylabel('BCE')
            axs[i, 0].grid(True, linestyle=':', alpha=0.6)

            if i == 0:
                axs[i, 0].legend(loc='upper right', framealpha=0.9, ncol=2)

            if i == group_size - 1:
                axs[i, 0].set_xlabel('Epochs')

            #rechte spalte: Accurarcy
            axs[i, 1].plot(eps, hist['acc'], label='Train', color='forestgreen', lw=1.2)
            axs[i, 1].plot(eps, hist['val_acc'], label='Val', color='crimson', linestyle='--', lw=1.2)
            axs[i, 1].set_title(f'Accuracy: {title}', pad=3, fontsize=9)
            axs[i, 1].set_ylabel('Acc')
            axs[i, 1].grid(True, linestyle=':', alpha=0.6)

            if i == 0:
                axs[i, 1].legend(loc='lower right', framealpha=0.9, ncol=2)

            if i == group_size - 1:
                axs[i, 1].set_xlabel('Epochs')

        plt.tight_layout()
        filename = f'bvp_history_part_{group_idx + 1}.pdf'
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"-> Trainingsverlauf unter '{filename}' gespeichert.")


def plot_wunsch_array_compact(soft_preds, true_vals, target_states):

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8
    })

    group_size = 4
    for group_idx in range(4):
        fig, axs = plt.subplots(group_size, 1, figsize=(5.8, 5.5), sharex=True)

        start_idx = group_idx * group_size
        end_idx = start_idx + group_size

        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            soft = soft_preds[global_i]
            true = true_vals[global_i]
            hard = (soft > 0.5).astype(int)
            state_label = target_states[global_i]

            ax = axs[local_i]

            # Soft-Predictions (blaue Balken)
            ax.bar(range(128), soft, color='royalblue', alpha=0.6, width=1.0, label='Soft Label')

            # Rote Linie für die 0.5 Entscheidungsgrenze
            ax.axhline(y=0.5, color='crimson', linestyle='--', lw=0.8, alpha=0.8)

            # Ground Truth (Grüne Punkte)
            ax.scatter(range(128), true, color='forestgreen', s=4, zorder=3, label='Ground Truth')

            # Fehler markieren (Kreuz)
            errors = np.where(hard != true)[0]
            if len(errors) > 0:
                ax.scatter(errors, hard[errors], color='red', marker='x', s=10, zorder=4, label='Error')

            ax.set_title(f'State $S_{{{state_label}}}$', pad=3, fontsize=9)
            ax.set_xlim(-2, 129)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.2, linestyle=':')
            ax.set_ylabel('$P(Bit=1)$')

            if local_i == group_size - 1:
                ax.set_xlabel('Bit Index (0-127)')

        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper center',
                   bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.35)

        filename = f'bvp_wunsch_array_part_{group_idx + 1}.pdf'
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"-> Inferenz-Grafik unter '{filename}' gespeichert.")

#Main Skript zur Ausführung
if __name__ == "__main__":
    SAMPLES = 500000
    VAL_SAMPLES = 50000

    data_tr = generate_bvp_data(SAMPLES)
    S_tr = data_tr[:17]
    M_tr = data_tr[17:]

    data_val = generate_bvp_data(VAL_SAMPLES)
    S_val = data_val[:17]

    networks = []
    histories_plot = []

    # Rückwärts durch die Schritte (Netz 16 bis Netz 1)
    for step in range(16, 0, -1):
        net = BVPNetwork()
        step_name = f"Netzwerk {step}"
        title_notation = f"$S_{{{step}}} \\rightarrow S_{{{step - 1}}}$"

        net, h = train_bvp_step(net, S_tr[step], S_tr[0], S_tr[step - 1],
                                S_val[step], S_val[0], S_val[step - 1],
                                step_name, epochs=1000, patience=25)

        #Model Gewichte speichern
        model_filename = f"bvp_model_step_{step}.pth"
        torch.save(net.state_dict(), model_filename)

        networks.append(net)
        histories_plot.append((h, title_notation))

    plot_bvp_history(histories_plot)

    #Inferenz-Test: single inferenztst

    data_test = generate_bvp_data(1)
    S_t = data_test[:17]

    for net in networks:
        net.eval()

    with torch.no_grad():
        target_beacon = S_t[0].to(device)
        current_input = S_t[16].to(device)  # Start bei S16

        soft_preds = []

        # Kaskade durchlaufen
        for step_idx, net in enumerate(networks):
            pred_soft = net(current_input, target_beacon)
            soft_preds.append(pred_soft[0].cpu().numpy())
            current_input = pred_soft  #Input für nächstes Netz

    target_states = [15 - i for i in range(16)]
    true_vals = [S_t[i][0].numpy() for i in range(15, -1, -1)]

    plot_wunsch_array_compact(soft_preds, true_vals, target_states)

    #inferenztest 2: trainingsdatensatz
    print("\nTestdatensatz Inferenz wird gestartet")

    data_tb = generate_bvp_data(10000)
    S0_tb, S16_tb = data_tb[0], data_tb[16]
    M_tb_list = data_tb[17:]

    with torch.no_grad():
        target_tb = S0_tb.to(device)
        current_input_tb = S16_tb.to(device)

        pred_states = [current_input_tb]
        for net in networks:
            current_input_tb = net(current_input_tb, target_tb)
            pred_states.append(current_input_tb)

        pred_states = pred_states[::-1]
        pred_states[0] = S0_tb.to(device)

        hard_states = [(s > 0.5).int().cpu().numpy() for s in pred_states]

    #Wortextraktion
    total_bits_per_word = 10000 * 32
    correct_bits_per_word = [0] * 16
    perfect_words_per_word = [0] * 16

    for idx in range(10000):
        for w in range(16):
            state_prev = hard_states[w][idx]
            A_prev = bits_to_int32(state_prev[0:32])
            B_prev = bits_to_int32(state_prev[32:64])
            C_prev = bits_to_int32(state_prev[64:96])
            D_prev = bits_to_int32(state_prev[96:128])

            state_curr = hard_states[w + 1][idx]
            B_curr = bits_to_int32(state_curr[32:64])

            extracted_M = extract_M_hard(A_prev, B_prev, C_prev, D_prev, B_curr, S_MD5[w], K_MD5[w])
            true_M = bits_to_int32(M_tb_list[w][idx].numpy())

            if extracted_M == true_M:
                perfect_words_per_word[w] += 1

            mismatches = bin(extracted_M ^ true_M).count('1')
            correct_bits_per_word[w] += (32 - mismatches)

    print(f"Evaluierte Test-Szenarien: 10000")
    print("\nErgebnisse:")

    total_correct_bits = sum(correct_bits_per_word)
    total_all_bits = total_bits_per_word * 16

    for w in range(16):
        acc = (correct_bits_per_word[w] / total_bits_per_word) * 100
        print(f"Wort M{w}: Bit-Genauigkeit = {acc:.2f} % | Perfekt rekonstruiert = {perfect_words_per_word[w]}/10000")

    overall_acc = (total_correct_bits / total_all_bits) * 100
    print(f"gesamtnachricht (M0-M15):")
    print(f"Gesamte Bit-Genauigkeit    : {overall_acc:.2f} %")