import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import string

# GPU suchen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendung von {device}")


# MD5 Konstanten für 16 Schritte
K_MD5 = [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8,
         0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821]
S_MD5 = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22]
# Initial Values S0
IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)


# KAN implementierung
class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        #Schrittweite
        h = (grid_range[1] - grid_range[0]) / grid_size
        #Gitter von kontrollpunkten
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                 -1).contiguous())
        self.register_buffer("grid", grid) #buffer=kein trainierbarer parameter

        #basisgewicht (wie bei MLP mit activierungsfunktion)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        #koefizienten für b spline funktionen
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        #mit skalierungsfaktor
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    #gewichtsinitialisierung mit rauschen
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features,
                                 self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    #bsplines basisfunktion für eingabe werte rekursiv berechnen
    def b_splines(self, x: torch.Tensor):
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + (
                        (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    # Spline-Koeffizienten an vorgegebene Funktionswerte anpassen
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    #summe aus SiLu mit bspline
    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                 self.scaled_spline_weight.view(self.out_features, -1))
        return (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)

#Netzwerk aus mehreren Schichten bauen
class KAN(torch.nn.Module):
    def __init__(self, layers_hidden, grid_size=10, spline_order=3):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


# MD5 Funktion:
#MD5 Padding
def padding(message: bytes):
    orig_len_in_bits = len(message) * 8
    padded = bytearray(message)
    padded.append(0x80) #erst 1
    while len(padded) % 64 != 56: #dann 0
        padded.append(0x00)

    #Länge anhängen
    padded += orig_len_in_bits.to_bytes(8, byteorder='little')

    #gepaddete Message in 16 32 bit words zerlegen
    words = []
    for i in range(0, 64, 4):
        four_bytes = padded[i: i + 4]
        word = int.from_bytes(four_bytes, byteorder='little')
        words.append(word)
    return words

#Rotationsfunktion
def left_rotate(x, amount):
    left_part = x << amount
    rigth_part = x >> (32 - amount)
    rotiert = left_part | rigth_part
    return rotiert & 0xFFFFFFFF

#Registerupdate
def md5_step(A, B, C, D, M, s, K):
    F_val = (B & C) | (~B & D)
    sum = (A + F_val + M + K) & 0xFFFFFFFF
    new_B = (B + left_rotate(sum, s)) & 0xFFFFFFFF
    return D, new_B, B, C

#Plaintext extrahieren
def extract_M_hard(A_prev, B_prev, C_prev, D_prev, B_curr, s, K):
    F_val = (B_prev & C_prev) | (~B_prev & D_prev)
    diff = (B_curr - B_prev) & 0xFFFFFFFF
    rotated = ((diff >> s) | (diff << (32 - s))) & 0xFFFFFFFF
    return (rotated - A_prev - F_val - K) & 0xFFFFFFFF


# Bitliste erstellen
def to_bits(val):
    bit_liste = []
    for i in range(32):
        shifted = val >> i
        einzelnes_bit = shifted & 1
        bit_liste.append(einzelnes_bit)
    return bit_liste

#Register in Vektor aus 1.0 und 0.0 schreiben
def state_to_bits(A, B, C, D):
    #Register einzeln in Bitliste schrieben
    bits_A = to_bits(A)
    bits_B = to_bits(B)
    bits_C = to_bits(C)
    bits_D = to_bits(D)
    #zu einer Liste zusammenfügen
    alle_bits = bits_A + bits_B + bits_C + bits_D
    #in einen Tensor umwandeln
    return torch.tensor(alle_bits, dtype=torch.float32)

#Bits in int32 Zahl umwandeln
def bits_to_int32(bit_array):
    number = 0
    for i in range(32):
        if bit_array[i] > 0.5: #wenn netz eine 1 vorhersagt
            number |= (1 << i) #1 an diese stelle schreiben und dazu addieren
    return number

#datengenerieren
def generate_backward_data(num_samples, steps=16):
    S_lists = [[] for _ in range(steps + 1)]
    M_lists = [[] for _ in range(steps)]

    for _ in range(num_samples):
        #zufällige Texte generieren
        laenge = random.randint(12, 48)
        zufalls_buchstaben = random.choices(string.ascii_lowercase, k=laenge)
        rand_str = ''.join(zufalls_buchstaben)

        #md5 berechnen
        words = padding(rand_str.encode('utf-8'))

        A, B, C, D = IV
        S_lists[0].append(state_to_bits(A, B, C, D))

        for i in range(steps):
            A, B, C, D = md5_step(A, B, C, D, words[i], S_MD5[i], K_MD5[i])
            S_lists[i + 1].append(state_to_bits(A, B, C, D))
            M_lists[i].append(torch.tensor([(words[i] >> j) & 1 for j in range(32)], dtype=torch.float32))

    return tuple(torch.stack(lst) for lst in S_lists + M_lists)


# KAN Netzwerk generieren
class PIKAN(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.kan_net = KAN(layers_hidden=[256, hidden_dim, 128], grid_size=10, spline_order=3)
        self.diffusion_net = nn.Sequential(
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    #laplace operator
    def compute_laplacian(self, u):
        u_left = torch.roll(u, shifts=1, dims=1)
        u_right = torch.roll(u, shifts=-1, dims=1)
        return u_left - 2.0 * u + u_right

    #vorhersage berechnen und dabei loss berechnen
    def forward(self, current_state, target_state):
        x = torch.cat((current_state, target_state), dim=1)
        pred = torch.sigmoid(self.kan_net(x))

        pred_bipolar = pred * 2.0 - 1.0
        curr_bipolar = current_state * 2.0 - 1.0
        d_u = self.diffusion_net(pred)
        laplacian = self.compute_laplacian(pred_bipolar)
        pde_residual = (pred_bipolar - curr_bipolar) - (d_u * laplacian)

        return pred, pde_residual


# Netzwerk trainieren
def train_hybrid_step(model, X_curr, X_target, Y_prev, X_val_curr, X_val_target, Y_val_prev, step_name, epochs=1000,
                      patience=25, beta=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    batch_size = 256
    #gesamtzahl an samples
    dataset_size = X_curr.shape[0]

    history = {
        'loss': [], 'val_loss': [],
        'data_loss': [], 'val_data_loss': [],
        'physics_loss': [], 'val_physics_loss': [],
        'acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        #daten durchmischen um netzwerk nicht immer die gleiche reihenfolge zu geben
        permutation = torch.randperm(dataset_size)
        epoch_loss, epoch_data, epoch_phys, epoch_acc = 0.0, 0.0, 0.0, 0.0
        batches = 0

        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_curr, batch_target, batch_prev = X_curr[indices].to(device), X_target[indices].to(device), Y_prev[
                indices].to(device)

            optimizer.zero_grad()
            preds, pde_residual = model(batch_curr, batch_target)

            data_loss = criterion(preds, batch_prev)
            physics_loss = torch.mean(pde_residual ** 2)

            loss = data_loss + beta * physics_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_data += data_loss.item()
            epoch_phys += physics_loss.item()
            epoch_acc += ((preds > 0.5).float() == batch_prev).float().mean().item()
            batches += 1

        model.eval()
        with torch.no_grad():
            v_curr, v_tgt, v_prev = X_val_curr.to(device), X_val_target.to(device), Y_val_prev.to(device)
            v_preds, v_pde = model(v_curr, v_tgt)

            val_data_loss = criterion(v_preds, v_prev).item()
            val_phys_loss = torch.mean(v_pde ** 2).item()
            val_loss = val_data_loss
            val_acc = ((v_preds > 0.5).float() == v_prev).float().mean().item()

        history['loss'].append(epoch_loss / batches)
        history['data_loss'].append(epoch_data / batches)
        history['physics_loss'].append(epoch_phys / batches)
        history['acc'].append(epoch_acc / batches)

        history['val_loss'].append(val_loss)
        history['val_data_loss'].append(val_data_loss)
        history['val_physics_loss'].append(val_phys_loss)
        history['val_acc'].append(val_acc)

        #early stopping überprüfen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[{step_name}] Epoche {epoch + 1:04d} | Total: {epoch_loss / batches:.4f} (Data: {epoch_data / batches:.4f}, Phys: {epoch_phys / batches:.4f}) | Val Data Loss: {val_loss:.6f} | Val Acc: {val_acc * 100:.2f}%")

        if patience_counter >= patience:
            print(f"Early stopping ausgelöst bei Epoche {epoch + 1}.")
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    return model, history


# Plotten im Latex stil
def plot_hybrid_history_latex(histories):
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

    group_size = 4
    for group_idx in range(4):
        fig, axs = plt.subplots(group_size, 2, figsize=(5.8, 6.0), sharex=True)

        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_histories = histories[start_idx:end_idx]

        for i, (hist, title) in enumerate(group_histories):
            eps = range(len(hist['loss']))

            # Linke Spalte: Loss
            axs[i, 0].plot(eps, hist['data_loss'], label='Train BCE', color='royalblue', lw=1.2)
            axs[i, 0].plot(eps, hist['physics_loss'], label='Train PDE', color='purple', lw=1.2)
            axs[i, 0].plot(eps, hist['val_data_loss'], label='Val BCE', color='darkorange', linestyle='--', lw=1.2)

            axs[i, 0].set_title(f'Loss: {title}', pad=3, fontsize=9)
            axs[i, 0].set_ylabel('Loss (log)')
            axs[i, 0].set_yscale('log')
            axs[i, 0].grid(True, linestyle=':', alpha=0.6)

            if i == 0:
                axs[i, 0].legend(loc='upper right', framealpha=0.9, ncol=1)
            if i == group_size - 1:
                axs[i, 0].set_xlabel('Epochs')

            # Rechte Spalte: Accuracy
            axs[i, 1].plot(eps, hist['acc'], label='Train Acc', color='forestgreen', lw=1.2)
            axs[i, 1].plot(eps, hist['val_acc'], label='Val Acc', color='crimson', linestyle='--', lw=1.2)

            axs[i, 1].set_title(f'Accuracy: {title}', pad=3, fontsize=9)
            axs[i, 1].set_ylabel('Acc')
            axs[i, 1].grid(True, linestyle=':', alpha=0.6)

            if i == 0:
                axs[i, 1].legend(loc='lower right', framealpha=0.9, ncol=2)
            if i == group_size - 1:
                axs[i, 1].set_xlabel('Epochs')

        plt.tight_layout()
        filename = f'pikan_history_part_{group_idx + 1}.pdf'
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"-> Trainingsverlauf unter '{filename}' gespeichert.")


def plot_wunsch_array_latex(soft_preds, true_vals, target_states):
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

            ax.bar(range(128), soft, color='royalblue', alpha=0.6, width=1.0, label='Soft Label')
            ax.axhline(y=0.5, color='crimson', linestyle='--', lw=0.8, alpha=0.8)
            ax.scatter(range(128), true, color='forestgreen', s=4, zorder=3, label='Ground Truth')

            errors = np.where(hard != true)[0]
            if len(errors) > 0:
                ax.scatter(errors, hard[errors], color='red', marker='x', s=10, zorder=4, label='Error')

            ax.set_title(f'State $S_{{{state_label}}}$ (PIKAN)', pad=3, fontsize=9)
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

        filename = f'pikan_single_inferenz_test_part_{group_idx + 1}.pdf'
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"-> Inferenz-Grafik unter '{filename}' gespeichert.")


# Main alles starten
if __name__ == "__main__":
    SAMPLES = 500000
    VAL_SAMPLES = 50000
    STEPS = 16

    print(f"Generiere Daten für {STEPS} Schritte...")
    data_tr = generate_backward_data(SAMPLES, steps=STEPS)
    S_tr = data_tr[:STEPS + 1]

    data_val = generate_backward_data(VAL_SAMPLES, steps=STEPS)
    S_val = data_val[:STEPS + 1]

    # erstes training: Kaskaden training

    networks = []
    histories_plot = []

    for step in range(STEPS, 0, -1):
        net = PIKAN()
        step_name = f"KAN-Netzwerk {step}"
        title_notation = f"$S_{{{step}}} \\rightarrow S_{{{step - 1}}}$"

        net, h = train_hybrid_step(net, S_tr[step], S_tr[0], S_tr[step - 1],
                                   S_val[step], S_val[0], S_val[step - 1],
                                   step_name, epochs=1000, patience=25, beta=0.001)

        #speichern der gewichte
        model_filename = f"pikan_model_step_{step}.pth"
        torch.save(net.state_dict(), model_filename)
        print(f"-> Modell {step_name} sicher unter '{model_filename}' gespeichert.\n")

        networks.append(net)
        histories_plot.append((h, title_notation))

    plot_hybrid_history_latex(histories_plot)

    #inferenz test: Single inferenz test
    data_test = generate_backward_data(1, steps=STEPS)
    S_t = data_test[:STEPS + 1]

    for net in networks:
        net.eval()

    with torch.no_grad():
        target_beacon = S_t[0].to(device)
        current_input = S_t[STEPS].to(device)
        soft_preds = []

        for net in networks:
            pred_soft, _ = net(current_input, target_beacon)
            soft_preds.append(pred_soft[0].cpu().numpy())
            current_input = pred_soft

    target_states = [STEPS - 1 - i for i in range(STEPS)]
    true_vals = [S_t[i][0].numpy() for i in range(STEPS - 1, -1, -1)]

    plot_wunsch_array_latex(soft_preds, true_vals, target_states)

    #inferenz test: testdatensatz
    test_samples = 10000
    data_tb = generate_backward_data(test_samples, steps=STEPS)
    S_tb = data_tb[:STEPS + 1]
    M_tb_list = data_tb[STEPS + 1:]

    with torch.no_grad():
        target_tb = S_tb[0].to(device)
        current_input_tb = S_tb[STEPS].to(device)

        pred_states = [current_input_tb]
        for net in networks:
            current_input_tb, _ = net(current_input_tb, target_tb)
            pred_states.append(current_input_tb)

        pred_states = pred_states[::-1]
        pred_states[0] = S_tb[0].to(device)
        hard_states = [(s > 0.5).int().cpu().numpy() for s in pred_states]

    total_bits_per_word = test_samples * 32
    correct_bits_per_word = [0] * STEPS
    perfect_words = [0] * STEPS

    for idx in range(test_samples):
        for w in range(STEPS):
            state_prev = hard_states[w][idx]
            A_prev, B_prev, C_prev, D_prev = bits_to_int32(state_prev[0:32]), bits_to_int32(
                state_prev[32:64]), bits_to_int32(state_prev[64:96]), bits_to_int32(state_prev[96:128])
            B_curr = bits_to_int32(hard_states[w + 1][idx][32:64])

            extracted_M = extract_M_hard(A_prev, B_prev, C_prev, D_prev, B_curr, S_MD5[w], K_MD5[w])
            true_M = bits_to_int32(M_tb_list[w][idx].numpy())

            if extracted_M == true_M: perfect_words[w] += 1
            correct_bits_per_word[w] += (32 - bin(extracted_M ^ true_M).count('1'))

    print("\nErgebnisse:")
    for w in range(STEPS):
        acc = (correct_bits_per_word[w] / total_bits_per_word) * 100
        print(f"Wort M{w}: Bit-Genauigkeit = {acc:.2f} % | Perfekt rekonstruiert = {perfect_words[w]}/{test_samples}")

    # Inferenz password
    print("BEISPIEL-INFERENZ FÜR: 'password'")

    beispiel_wort = "password"
    # Padding für das password
    words_beispiel = padding(beispiel_wort.encode('utf-8'))

    #MD5 Zustände berechnen
    A, B, C, D = IV
    states_forward = [state_to_bits(A, B, C, D)]

    for i in range(STEPS):
        A, B, C, D = md5_step(A, B, C, D, words_beispiel[i], S_MD5[i], K_MD5[i])
        states_forward.append(state_to_bits(A, B, C, D))

    current_input_ex = states_forward[STEPS].unsqueeze(0).to(device)  # S_16
    target_beacon_ex = states_forward[0].unsqueeze(0).to(device)  # S_0 (IV)

    # Rückwärts-Inferenz durch alle KAN-Netzwerke schicken
    pred_states_ex = [current_input_ex[0].cpu()]

    with torch.no_grad():
        for net in networks:
            net.eval()
            current_input_ex, _ = net(current_input_ex, target_beacon_ex)
            pred_states_ex.append(current_input_ex[0].cpu())

    # Die Liste umdrehen, damit sie chronologisch von S_15 bis S_0 sortiert ist
    pred_states_ex = pred_states_ex[::-1]

    for w in range(STEPS):
        # Vorhergesagten Zustand in Bit-Array konvertieren
        state_prev_soft = pred_states_ex[w]
        state_prev_hard = (state_prev_soft > 0.5).int().numpy()

        # Aus den Bits die 32-bit Integer Register rekonstruieren
        A_prev = bits_to_int32(state_prev_hard[0:32])
        B_prev = bits_to_int32(state_prev_hard[32:64])
        C_prev = bits_to_int32(state_prev_hard[64:96])
        D_prev = bits_to_int32(state_prev_hard[96:128])

        # Den darauffolgenden Zustand (B-Register) holen
        next_state_hard = (pred_states_ex[w + 1].numpy() > 0.5).astype(int)
        B_curr = bits_to_int32(next_state_hard[32:64])

        # Wort M_w über die mathematische Umkehrfunktion berechnen
        extracted_M = extract_M_hard(A_prev, B_prev, C_prev, D_prev, B_curr, S_MD5[w], K_MD5[w])
        true_M = words_beispiel[w]

        # Versuchen, das Integer-Wort wieder in lesbare ASCII-Zeichen zu wandeln
        try:
            ext_bytes = extracted_M.to_bytes(4, byteorder='little').decode('utf-8', errors='ignore').strip('\x00')
            true_bytes = true_M.to_bytes(4, byteorder='little').decode('utf-8', errors='ignore').strip('\x00')
        except:
            ext_bytes = "?"
            true_bytes = "?"

        status = "Korrekt" if extracted_M == true_M else "Fehler"
        print(
            f"Runde {w + 1:2d}  | 0x{extracted_M:08x} ('{ext_bytes}')     | 0x{true_M:08x} ('{true_bytes}')     | {status}")