import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import pickle

np.random.seed(42)
tf.random.set_seed(42)

# für die Grafiken im Latex stil
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


# MD5 Funktion:
# Rotationsfunktion
def left_rotate(x, amount):
    part_1 = x << amount  # Bits von x nach links schieben
    part_2 = x >> (32 - amount)  # Die vordersten Bits nach rechts schieben
    rotate = part_1 | part_2  # Beides zusammenkleben
    return rotate & 0xFFFFFFFF  # Nach 32 Bit abschneiden

# Update des Registers (Formel aus Skript)
def md5_step(A, B, C, D, M, s, K):
    F = (B & C) | (~B & D) # Function F
    sum = (A + F + M + K) & 0xFFFFFFFF
    new_B = (B + left_rotate(sum, s)) & 0xFFFFFFFF # Register Updaten
    return D, new_B, B, C # Register in neuer Reihenfolge zurückgeben

# Gesamte MD5 Schritte für bestimmte Schrittanzahl
def complete_md5_steps(message, num_steps=16):

    # MD5 Padding
    padded = bytearray(message)
    padded.append(0x80) #zuerst eine 1 anhängen
    while len(padded) % 64 != 56: #so viele nullen anhängen bis länge ein vielfaches von 64-8 ist (in Byte)
        padded.append(0x00)

    # Länge anhängen
    orig_len_in_bits = len(message) * 8 # Anzahl in Bits
    padded += orig_len_in_bits.to_bytes(8, byteorder='little') # aus Länge 8 byte machen und in little Endian Reihenfolge anhängen
    words = [] # Liste für Wörter
    for i in range(0, 64, 4): # 64 Byte in 4er Schritten durchgehen, für 16 32 bit words
        four_bytes = padded[i: i + 4]
        word = int.from_bytes(four_bytes, byteorder='little') # einzelnes word M[k]
        words.append(word) # Liste aller 16 words

    # initiale Konstanten
    A, B, C, D = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
    K = [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
         0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
         0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
         0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821]
    S = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22]

    # Schritte ausführen
    for i in range(num_steps):
        A, B, C, D = md5_step(A, B, C, D, words[i], S[i], K[i])
    return (A << 96) | (B << 64) | (C << 32) | D # Register zusammenführen

# Liste aus einzelnen Bits als Float zahlen
def get_bit_vector(val, length=128):
    bit_liste = []
    for i in range(length):
        temp_position = val >> i # Position des jeweiligem Bits
        temp_bit = temp_position & 1 # Rest abschneiden, damit nur Bit bleibt
        bit_liste.append(temp_bit)
    return np.array(bit_liste, dtype=np.float32)


# daten generieren
def generate_mlp_dataset(n_samples=100000, n_bits=64, hash_bits=128):
    print(f"generate {n_samples} samples")
    X, Y = [], [] #Input, Outputliste
    seen_samples = set() # schnelleres []

    # Label generieren
    while len(seen_samples) < n_samples: # so viele genrerieren bis n_samples erreicht ist
        n_ones = np.random.randint(0, n_bits + 1) # zufällige Anzahl an 1 auswählen
        ones_position = np.random.choice(n_bits, n_ones, replace=False)

        #Label Liste mit gewünschten positionen der 1 erstellen
        label = np.zeros(n_bits, dtype=np.float32)
        label[ones_position] = 1.0

        # Für den MD5 Algorithmus in Bytes umwandeln
        temp_number = 0
        for bit_pos in ones_position:
            shifted = 1 << bit_pos
            temp_number = temp_number | shifted

        input_bytes = temp_number.to_bytes(8, byteorder='little')

        # doppelte Werte überspringen
        if input_bytes in seen_samples:
            continue

        seen_samples.add(input_bytes)

        # Input generieren
        hash_value = complete_md5_steps(input_bytes, num_steps=16)
        hash_vector = get_bit_vector(hash_value, hash_bits)

        X.append(hash_vector)
        Y.append(label)
    return np.array(X), np.array(Y), seen_samples # von Liste in Array umwandeln für Netzwerk


# MLP Architektur
def mlp_architecture(input_dim):
    model = models.Sequential([
        #Input Layer
        layers.Input(shape=(input_dim,)),
        # erstes Hidden Layer
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        #zweites Hidden Layer
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        #drittes Hidden Layer
        layers.Dense(128, activation='relu'),
        #Output Layer
        layers.Dense(64, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model


def plot_training_history(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8))

    # loss plot
    axes[0].plot(epochs_range, loss, label='Train Loss', color='royalblue', lw=1.2)
    axes[0].plot(epochs_range, val_loss, label='Val Loss', color='darkorange', linestyle='--', lw=1.2)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', framealpha=0.9)

    # accuracy plot
    axes[1].plot(epochs_range, acc, label='Train Acc', color='forestgreen', lw=1.2)
    axes[1].plot(epochs_range, val_acc, label='Val Acc', color='crimson', linestyle='--', lw=1.2)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower right', framealpha=0.9)

    fig.tight_layout()
    plt.savefig('training_history_mlp_16_schritte.pdf', format='pdf', bbox_inches='tight')
    plt.close()


# mainskript
if __name__ == "__main__":

    # Schritt 1: Netzwerk trainieren
    N_BITS = 64
    HASH_BITS = 128
    MODEL_FILE = 'mlp_16_schritte.keras'
    SEEN_FILE = 'seen_samples.pkl'

    seen_train_samples = set()

    # um doppeltes training zu vermeiden gewichte suchen
    if os.path.exists(MODEL_FILE) and os.path.exists(SEEN_FILE):
        print(f"\nModell '{MODEL_FILE}' vorhanden, Training wird übersprungen")
        model = models.load_model(MODEL_FILE) # Model lden
        model.summary() # Konsolenausgabe überdas Netzwerk

        with open(SEEN_FILE, 'rb') as f:
            seen_train_samples = pickle.load(f) # generierte samples laden
    else:
        print(f"\nKein Modell vorhanden, Training startet")
        X_train, Y_train, seen_train_samples = generate_mlp_dataset(n_samples=1000000, n_bits=N_BITS,
                                                                    hash_bits=HASH_BITS)

        model = mlp_architecture(input_dim=HASH_BITS)
        model.summary() # Konsolenausgabe über Netzwerk

        #Early Stopping hinzufügen
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        # Model Trainieren
        history = model.fit(
            X_train, Y_train,
            epochs=1000,
            batch_size=128,
            validation_split=0.2,
            callbacks=[early_stop]
        )

        plot_training_history(history) #in Konsole ausgeben
        model.save(MODEL_FILE) #Model speichern

        # Generierte Samples abspeichern
        with open(SEEN_FILE, 'wb') as f:
            pickle.dump(seen_train_samples, f)

        print(f"Modell unter '{MODEL_FILE}' gespeichert.\n")

    # Schritt 2: Model testen
    # DATABASE Inferenz Test:
    print("\nStarte DATABASE-Inferenz-Test")

    true_active_bits = [0, 1, 5, 7, 12, 15, 23, 25, 27, 30, 37, 40, 44, 52, 55, 59, 61, 63] #DATABASE positionen
    print(f"Tatsächlich gesetzte Bits (Wahrheit): {true_active_bits}")

    #Label erstellen
    true_label = np.zeros(N_BITS)
    true_label[true_active_bits] = 1

    # Input erstellen
    # In Bytes umwandeln für MD5 funktion
    temp_number = 0
    for bit_pos in true_active_bits:
        temp_number |= (1 << bit_pos)
    test_input = temp_number.to_bytes(8, byteorder='little')

    # md5 berechnen
    test_hash = complete_md5_steps(test_input, num_steps=16)
    #in bits umwandeln
    hash_vector = get_bit_vector(test_hash, HASH_BITS)

    # Dimension anpassen für das Netzwerk
    mlp_input_batch = np.expand_dims(hash_vector, axis=0)

    # Vorhersagen ausgeben lassen
    prediction = model.predict(mlp_input_batch, verbose=0)[0]
    # threshold von 0.5
    predicted_active_bits = np.where(prediction > 0.5)[0]
    print(f"Vom MLP erkannte Bits: {list(predicted_active_bits)}")

    # Inferenztest plotten:
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 4.5))
    # Plot 1: Soft Predictions
    axes[0].bar(range(N_BITS), prediction, color='royalblue', alpha=0.7, width=0.8, label='Soft Label')
    axes[0].axhline(y=0.5, color='crimson', linestyle='--', lw=1, label='Threshold (0.5)')
    axes[0].set_ylabel('Probability')
    axes[0].set_xlim(-1, N_BITS)
    axes[0].grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', framealpha=0.9, ncol=2)
    # Plot 2: Hard Predictions vs Ground Truth
    markerline_g, stemlines_g, _ = axes[1].stem(range(N_BITS), true_label, linefmt='forestgreen', markerfmt='go',
                                                basefmt=' ', label='Ground Truth')
    plt.setp(markerline_g, markersize=3)
    plt.setp(stemlines_g, linewidth=0.8)

    if predicted_active_bits.size > 0:
        markerline_r, stemlines_r, _ = axes[1].stem(predicted_active_bits, np.ones_like(predicted_active_bits),
                                                    linefmt='crimson', markerfmt='rx', basefmt=' ',
                                                    label='Model Decision')
        plt.setp(markerline_r, markersize=4)
        plt.setp(stemlines_r, linewidth=0.8, linestyle='--')
    else:
        axes[1].plot([-1], [0], 'rx', label='Model Decision (None Detected)')

    axes[1].set_ylabel('Bit Value')
    axes[1].set_xlabel('Bit Position (0-63)')
    axes[1].set_xlim(-1, N_BITS)
    axes[1].set_ylim(-0.1, 1.4)
    axes[1].set_xticks(range(0, N_BITS, 4))
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right', framealpha=0.9, ncol=2)

    fig.tight_layout()
    plt.savefig('single_inferenz_mlp_16_schritte.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    if set(true_active_bits) == set(predicted_active_bits):
        print("Erfolg: Das MLP hat alle Bits korrekt invertiert!")
    else:
        print("Fehler: Die Vorhersage stimmt nicht mit der Wahrheit überein.")



    # Schritt 3: Inferenz auf Testdatensatz
    print("\nStarte Inferenztest über 10000 zufällige Hashes")

    N_TEST_RUNS = 10000
    total_correct_bits = 0
    perfect_matches = 0

    for run in range(N_TEST_RUNS):
        # Generiere ungesehene Daten für den Test
        while True:
            # Zufällige Labels generiern
            n_ones = np.random.randint(0, N_BITS + 1)
            ones_position = np.random.choice(range(N_BITS), n_ones, replace=False)

            test_true_vector = np.zeros(N_BITS, dtype=np.float32)
            test_true_vector[ones_position] = 1.0

            # generiere Input
            # in byte umwandeln für MD5
            temp_number = 0
            for b in ones_position:
                temp_number |= (1 << b)

            input_bytes = temp_number.to_bytes(8, byteorder='little')
            if input_bytes not in seen_train_samples:
                seen_train_samples.add(input_bytes)
                break

        # MD5 berechnn
        loop_hashvalue = complete_md5_steps(input_bytes, num_steps=16)

        loop_hash_vector = get_bit_vector(loop_hashvalue, HASH_BITS)

        # Batch-Dimension für das MLP hinzufügen
        loop_mlp_in = np.expand_dims(loop_hash_vector, axis=0)
        p1 = model.predict(loop_mlp_in, verbose=0)[0]

        final_binary_pred = (p1 > 0.5).astype(int)

        # zählen wie viele bits korrekt sind
        correct_in_this_run = np.sum(test_true_vector == final_binary_pred)
        total_correct_bits += correct_in_this_run

        # zählen wie viele klartexte richtig sind
        if correct_in_this_run == N_BITS:
            perfect_matches += 1

    total_evaluated_bits = N_TEST_RUNS * N_BITS
    average_bit_accuracy = (total_correct_bits / total_evaluated_bits) * 100
    perfect_match_percentage = (perfect_matches / N_TEST_RUNS) * 100

    print("\nAuswertung: 16 Schritte MLP")
    print(f"Evaluierte Hashes: {N_TEST_RUNS}")
    print(f"Evaluierte Einzelbits insgesamt: {total_evaluated_bits}")
    print(f"Korrekt vorhergesagte Bits: {total_correct_bits}")
    print(f"Durchschnittliche Bit-Genauigkeit: {average_bit_accuracy:.4f} %")
    print(
        f"Perfekt geknackte Hashes (100% korr.): {perfect_matches} von {N_TEST_RUNS} ({perfect_match_percentage:.2f} %)")