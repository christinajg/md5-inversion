import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.fft import fft
import matplotlib.pyplot as plt
import os
import pickle

np.random.seed(42)
tf.random.set_seed(42)

# visualisierungen im Latex Stil
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
    part_left = x << amount
    part_right = x >> (32 - amount)
    rotiert = part_left | part_right
    return rotiert & 0xFFFFFFFF


# Update Formel aus Skript
def md5_step(A, B, C, D, M, s, K):
    F = (B & C) | (~B & D)
    temp = (A + F + M + K) & 0xFFFFFFFF
    new_B = (B + left_rotate(temp, s)) & 0xFFFFFFFF
    return D, new_B, B, C

# Komplette MD5 Funktion
def complete_md5_steps(message, num_steps=16):
    # Padding
    orig_len_in_bits = len(message) * 8
    padded = bytearray(message)
    padded.append(0x80)
    while len(padded) % 64 != 56:
        padded.append(0x00)
    # Länge anhängen
    padded += orig_len_in_bits.to_bytes(8, byteorder='little')

    # in 16 words zerlegen
    words = []
    for i in range(0, 64, 4):
        four_bytes = padded[i: i + 4]
        word = int.from_bytes(four_bytes, byteorder='little')
        words.append(word)

    # Initiale Konstanten
    A, B, C, D = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476

    K = [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
         0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
         0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
         0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821]

    S = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22]

    # MD5 Ausführung
    for i in range(num_steps):
        A, B, C, D = md5_step(A, B, C, D, words[i], S[i], K[i])

    return (A << 96) | (B << 64) | (C << 32) | D


# Bit Liste erstellen
def get_bit_vector(val, length=128):
    bit_liste = []
    for i in range(length):
        shifted = val >> i
        einzelnes_bit = shifted & 1
        bit_liste.append(einzelnes_bit)
    return np.array(bit_liste, dtype=np.float32)


# Gaußsignal berechnen
def to_continuous_gauss(bit_vector, x_axis, sigma=0.7):
    signal = np.zeros_like(x_axis)
    for i, bit_active in enumerate(bit_vector):
        if bit_active > 0:
            signal += bit_active * np.exp(-0.5 * ((x_axis - i) / sigma) ** 2)
    return signal

# Fourier Spektrum berechnen
def prepare_fft_input(histogram_vector, x_axis_local, sigma=0.7):
    gauss_signal = to_continuous_gauss(histogram_vector, x_axis_local, sigma)
    f_transform = fft(gauss_signal)
    return np.stack([np.real(f_transform), np.imag(f_transform)], axis=-1)


# Samples generieren
def generate_hash_dataset(n_samples=100000, n_bits=64, resolution=10):
    print(f"Generiere {n_samples} Trainingssamples")
    HASH_BITS = 128
    x_axis = np.linspace(0, HASH_BITS - 1, HASH_BITS * resolution)
    ref_input = (0).to_bytes(8, byteorder='little')  # 64 Nullen
    ref_hash = complete_md5_steps(ref_input, num_steps=16)

    X, Y = [], []
    seen_train_samples = set()

    while len(seen_train_samples) < n_samples:
        n_flips = np.random.randint(1, 64)
        ones_position = np.random.choice(n_bits, n_flips, replace=False)

        label = np.zeros(n_bits)
        label[ones_position] = 1

        #Bytes für MD5
        temp_number = 0
        for bit_pos in ones_position:
            temp_number |= (1 << bit_pos)
        byte_input = temp_number.to_bytes(8, byteorder='little')

        # Dopplte samples überspringen
        if byte_input in seen_train_samples:
            continue
        seen_train_samples.add(byte_input)

        # Input generieren
        hash_value = complete_md5_steps(byte_input, num_steps=16)
        diff_vector = get_bit_vector(ref_hash ^ hash_value, HASH_BITS)

        cnn_input = prepare_fft_input(diff_vector, x_axis)

        X.append(cnn_input)
        Y.append(label)

    return np.array(X), np.array(Y), seen_train_samples


# CNN Architektur
def cnn_architecture(input_shape):
    model = models.Sequential([
        #Input Layer
        layers.Input(shape=input_shape),
        #erstes Hidden Layer
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        #zweites Hidden Layer
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        #Flattening Layer
        layers.Flatten(),
        #MLP Layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        #Output Layer
        layers.Dense(64, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model

# Loss und Accuracy Plotten
def plot_training_history(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8))

    # Loss Plot
    axes[0].plot(epochs_range, loss, label='Train Loss', color='royalblue', lw=1.2)
    axes[0].plot(epochs_range, val_loss, label='Val Loss', color='darkorange', linestyle='--', lw=1.2)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', framealpha=0.9)

    # Accuracy Plot
    axes[1].plot(epochs_range, acc, label='Train Acc', color='forestgreen', lw=1.2)
    axes[1].plot(epochs_range, val_acc, label='Val Acc', color='crimson', linestyle='--', lw=1.2)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower right', framealpha=0.9)

    fig.tight_layout()
    plt.savefig('training_history_cnn_16schritte.pdf', format='pdf', bbox_inches='tight')
    plt.close()


# Main
if __name__ == "__main__":

    #Schritt 1: Netzwerk trainiren
    N_BITS = 64
    RES = 10
    HASH_BITS = 128
    MODEL_FILE = 'cnn_model_16_schritte.keras'
    SEEN_FILE = 'cnn_seen_samples.pkl'
    x_axis = np.linspace(0, HASH_BITS - 1, HASH_BITS * RES)

    ref_input = (0).to_bytes(8, byteorder='little')
    ref_hash = complete_md5_steps(ref_input, num_steps=16)

    seen_train_samples = set()

    # Model suchen um doppeltes training zu vermeiden
    if os.path.exists(MODEL_FILE) and os.path.exists(SEEN_FILE):
        print(f"\nModell '{MODEL_FILE}' gefunden")
        model = models.load_model(MODEL_FILE)
        model.summary()

        #generierte Daten laden
        with open(SEEN_FILE, 'rb') as f:
            seen_train_samples = pickle.load(f)
    else:
        print(f"\nKein Modell gefunden")
        # Netzwerk trainieren
        X_train, Y_train, seen_train_samples = generate_hash_dataset(n_samples=200000, n_bits=N_BITS, resolution=RES)

        model = cnn_architecture(X_train.shape[1:])
        model.summary() # Konsolenausgabe

        #Early stopping um overfitting zu vermeiden
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, Y_train,
            epochs=1000,
            batch_size=128,
            validation_split=0.2,
            callbacks=[early_stop]
        )

        plot_training_history(history)
        model.save(MODEL_FILE)

        # Generierte Trainings-Samples abspeichern
        with open(SEEN_FILE, 'wb') as f:
            pickle.dump(seen_train_samples, f)



    # Schritt 2: DATABASE-Inferenz-Test (Einzeltest)
    print("\nStarte DATABASE-Inferenz-Test")

    true_flipped_bits = [0, 1, 5, 7, 12, 15, 23, 25, 27, 30, 42, 48, 53, 59, 61, 62] #DATABASE
    print(f"Tatsächlich geflippte Bits: {true_flipped_bits}")

    true_label = np.zeros(N_BITS)
    true_label[true_flipped_bits] = 1

    #Byte für MD5
    temp_number = 0
    for bit_pos in true_flipped_bits:
        temp_number |= (1 << bit_pos)
    test_input = temp_number.to_bytes(8, byteorder='little')

    # generiere Input
    test_hash = complete_md5_steps(test_input, num_steps=16)
    diff_vector = get_bit_vector(ref_hash ^ test_hash, HASH_BITS)
    cnn_input = prepare_fft_input(diff_vector, x_axis)
    # Dimension für Netzwerk anpassen
    cnn_input_batch = np.expand_dims(cnn_input, axis=0)

    # Vorhersage ausgeben lassen
    prediction = model.predict(cnn_input_batch, verbose=0)[0]
    predicted_flipped_bits = np.where(prediction > 0.5)[0]
    print(f"Vorhergesagte Bits: {list(predicted_flipped_bits)}")

    # Inferenz plotten lassen
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 4.5))

    axes[0].bar(range(N_BITS), prediction, color='royalblue', alpha=0.7, width=0.8, label='Soft Label')
    axes[0].axhline(y=0.5, color='crimson', linestyle='--', lw=1, label='Threshold (0.5)')
    axes[0].set_ylabel('Probability')
    axes[0].set_xlim(-1, N_BITS)
    axes[0].grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', framealpha=0.9, ncol=2)

    markerline_g, stemlines_g, _ = axes[1].stem(range(N_BITS), true_label, linefmt='forestgreen', markerfmt='go',
                                                basefmt=' ', label='Ground Truth')
    plt.setp(markerline_g, markersize=3)
    plt.setp(stemlines_g, linewidth=0.8)

    if predicted_flipped_bits.size > 0:
        markerline_r, stemlines_r, _ = axes[1].stem(predicted_flipped_bits, np.ones_like(predicted_flipped_bits),
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
    plt.savefig('single_inference_16steps_eng.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    if set(true_flipped_bits) == set(predicted_flipped_bits):
        print("Erfolg: Alle Bits korrekt vorhergesagt")
    else:
        print("Fehler: Die Vorhersage stimmt nicht.")



    # Schritt 3: Testdatensatz Inferenztest
    print("\nStarte Inferenztest über 10000 garantierte ungesehene Test-Hashes")

    N_TEST_RUNS = 10000
    X_test_batch = []
    Y_test_batch = []
    seen_test_samples = set()

    for run in range(N_TEST_RUNS):
        while True:
            #zufällige Labels generieren
            n_flips_test = np.random.randint(1, N_BITS)
            ones_position = np.random.choice(N_BITS, n_flips_test, replace=False)

            #byte für MD5
            temp_number = 0
            for b in ones_position:
                temp_number |= (1 << b)

            loop_input = temp_number.to_bytes(8, byteorder='little')

            # schon gesehene samples überspringen
            if loop_input not in seen_train_samples and loop_input not in seen_test_samples:
                seen_test_samples.add(loop_input)
                break

        test_true_vector = np.zeros(N_BITS)
        test_true_vector[ones_position] = 1

        # Input generiere
        loop_hashvalue = complete_md5_steps(loop_input, num_steps=16)
        loop_diff_vector = get_bit_vector(ref_hash ^ loop_hashvalue, HASH_BITS)
        loop_cnn_in = prepare_fft_input(loop_diff_vector, x_axis)

        # batches erstellen
        X_test_batch.append(loop_cnn_in)
        Y_test_batch.append(test_true_vector)

    X_test_array = np.array(X_test_batch)
    Y_test_array = np.array(Y_test_batch)

    #Vorhersagen ausgeben lassen
    predictions = model.predict(X_test_array, verbose=0)
    final_binary_preds = (predictions > 0.5).astype(int)

    correct_bits_per_run = np.sum(Y_test_array == final_binary_preds, axis=1)

    total_correct_bits = np.sum(correct_bits_per_run)
    perfect_matches = np.sum(correct_bits_per_run == N_BITS)

    total_evaluated_bits = N_TEST_RUNS * N_BITS
    average_bit_accuracy = (total_correct_bits / total_evaluated_bits) * 100
    perfect_match_percentage = (perfect_matches / N_TEST_RUNS) * 100

    print("\nAuswertung: 16 Schritte CNN")
    print(f"Evaluierte Hashes: {N_TEST_RUNS}")
    print(f"Evaluierte Einzelbits insgesamt: {total_evaluated_bits}")
    print(f"Korrekt vorhergesagte Bits: {total_correct_bits}")
    print(f"Durchschnittliche Bit-Genauigkeit: {average_bit_accuracy:.4f} %")
    print(f"Perfekt geknackte Hashes: {perfect_matches} von {N_TEST_RUNS} ({perfect_match_percentage:.2f} %)")