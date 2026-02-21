# voice-recognition-cnn-rnn-nlp
 Voice Recognition System using CNN-RNN and NLP in Python
"""
Voice Recognition System - Main Entry Point
Run: python main.py
"""

import os
import warnings
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, LSTM, Dense,
    Dropout, BatchNormalization, Bidirectional, Reshape
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("=" * 60)
print("  VOICE RECOGNITION SYSTEM - CNN-RNN + NLP")
print("=" * 60)
print(f"  TensorFlow: {tf.__version__}")
print(f"  GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 60)

# ============================================================
# STEP 1: GENERATE SYNTHETIC AUDIO DATASET
# ============================================================
def generate_voice_audio(command, duration=1.0, sr=16000, num_samples=80):
    """Generate synthetic voice-like audio for each command."""
    patterns = {
        'yes':   {'formants': [270, 530, 2500], 'pitch': 150, 'mod': 5,  'env': 'rise_fall'},
        'no':    {'formants': [400, 800, 2300], 'pitch': 120, 'mod': 3,  'env': 'falling'},
        'up':    {'formants': [300, 900, 2200], 'pitch': 180, 'mod': 7,  'env': 'rising'},
        'down':  {'formants': [250, 600, 2600], 'pitch': 100, 'mod': 4,  'env': 'falling'},
        'left':  {'formants': [350, 700, 2400], 'pitch': 160, 'mod': 6,  'env': 'pulse'},
        'right': {'formants': [320, 850, 2100], 'pitch': 140, 'mod': 8,  'env': 'double_pulse'},
        'stop':  {'formants': [280, 750, 2700], 'pitch': 110, 'mod': 2,  'env': 'sharp'},
        'go':    {'formants': [500, 1000, 2000], 'pitch': 200, 'mod': 10, 'env': 'smooth'},
        'hello': {'formants': [330, 660, 2300], 'pitch': 170, 'mod': 5,  'env': 'multi'},
        'help':  {'formants': [290, 580, 2500], 'pitch': 190, 'mod': 9,  'env': 'sharp_rise'},
    }
    p = patterns[command]
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = []
    for i in range(num_samples):
        pitch = p['pitch'] * (1 + np.random.uniform(-0.15, 0.15))
        glottal = np.zeros_like(t)
        for h in range(1, 8):
            amp = 1.0 / (h ** 1.2)
            glottal += amp * np.sin(2 * np.pi * pitch * h * t + np.random.uniform(0, 2 * np.pi))
        signal = np.zeros_like(t)
        for j, formant in enumerate(p['formants']):
            f_var = formant * (1 + np.random.uniform(-0.08, 0.08))
            bw = 60 + j * 20
            resonance = np.sin(2 * np.pi * f_var * t) * np.exp(-np.pi * bw * np.abs(t - 0.5))
            signal += resonance / (j + 1)
        signal = signal * 0.4 + glottal * 0.6
        mod_f = p['mod'] * (1 + np.random.uniform(-0.2, 0.2))
        signal *= (1 + 0.4 * np.sin(2 * np.pi * mod_f * t))
        env = np.ones_like(t)
        if p['env'] == 'rise_fall':
            env = np.sin(np.pi * t / duration) ** 0.7
        elif p['env'] == 'falling':
            env = np.exp(-2.5 * t / duration)
        elif p['env'] == 'rising':
            env = 1 - np.exp(-3.0 * t / duration)
        elif p['env'] == 'pulse':
            env = np.exp(-((t - 0.4 * duration) ** 2) / (0.04 * duration ** 2))
        elif p['env'] == 'double_pulse':
            env = (np.exp(-((t - 0.3 * duration) ** 2) / (0.02 * duration ** 2)) +
                   np.exp(-((t - 0.7 * duration) ** 2) / (0.02 * duration ** 2)))
        elif p['env'] == 'sharp':
            env = np.exp(-5 * t / duration) * np.abs(np.sin(6 * np.pi * t / duration))
        elif p['env'] == 'smooth':
            env = np.sin(np.pi * t / duration)
        elif p['env'] == 'multi':
            env = np.sin(np.pi * t / duration) * (1 + 0.3 * np.sin(4 * np.pi * t / duration))
        elif p['env'] == 'sharp_rise':
            env = (1 - np.exp(-8 * t / duration)) * np.exp(-1.5 * t / duration)
        signal *= env
        signal += np.random.uniform(0.01, 0.05) * np.random.randn(len(t))
        shift = int(np.random.uniform(-0.1, 0.1) * len(t))
        signal = np.roll(signal, shift)
        signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.85
        samples.append(signal.astype(np.float32))
    return samples


COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go', 'hello', 'help']
SR = 16000
DURATION = 1.0
SAMPLES_PER_CLASS = 80
BASE_DIR = 'voice_dataset'
os.makedirs(BASE_DIR, exist_ok=True)

all_files = []
all_labels = []

print("\n🎵 STEP 1: Generating Synthetic Voice Dataset...")
print("-" * 50)
for cmd in COMMANDS:
    cmd_dir = os.path.join(BASE_DIR, cmd)
    os.makedirs(cmd_dir, exist_ok=True)
    samples = generate_voice_audio(cmd, DURATION, SR, SAMPLES_PER_CLASS)
    for idx, sample in enumerate(samples):
        fpath = os.path.join(cmd_dir, f'{cmd}_{idx:03d}.wav')
        sf.write(fpath, sample, SR)
        all_files.append(fpath)
        all_labels.append(cmd)
    print(f"  ✅ '{cmd:>5s}' → {SAMPLES_PER_CLASS} files")
print(f"📁 Total: {len(all_files)} files\n")


# ============================================================
# STEP 2: EXTRACT MFCC FEATURES
# ============================================================
N_MFCC = 40
MAX_LEN = 32

def extract_features(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN):
    """Extract MFCC + Delta features from audio file."""
    try:
        y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
        mfcc_delta = librosa.feature.delta(mfcc)
        features = np.concatenate([mfcc, mfcc_delta], axis=0)
        if features.shape[1] < max_len:
            features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
        else:
            features = features[:, :max_len]
        return features
    except Exception as e:
        print(f"Error: {e}")
        return None


print("🔧 STEP 2: Extracting MFCC features...")
features_list = []
labels_list = []
for i, (fpath, label) in enumerate(zip(all_files, all_labels)):
    feat = extract_features(fpath)
    if feat is not None:
        features_list.append(feat)
        labels_list.append(label)
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(all_files)}")

X = np.array(features_list)
y_labels_arr = np.array(labels_list)
print(f"✅ Features shape: {X.shape}\n")


# ============================================================
# STEP 3: PREPARE DATA
# ============================================================
print("📊 STEP 3: Preparing data...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels_arr)
y_onehot = to_categorical(y_encoded)
num_classes = len(label_encoder.classes_)

X_mean = np.mean(X, axis=(0, 2), keepdims=True)
X_std = np.std(X, axis=(0, 2), keepdims=True) + 1e-8
X_normalized = (X - X_mean) / X_std
X_cnn = X_normalized[..., np.newaxis]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_cnn, y_onehot, test_size=0.15, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.176, random_state=42
)
print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}\n")


# ============================================================
# STEP 4: BUILD CNN-RNN MODEL
# ============================================================
print("🏗️  STEP 4: Building CNN-RNN Model...")

def build_cnn_rnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    cnn_shape = x.shape
    x = Reshape((cnn_shape[2], cnn_shape[1] * cnn_shape[3]))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

model = build_cnn_rnn_model(X_train.shape[1:], num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================
print("\n🚀 STEP 5: Training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_voice_model.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
]

history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=60, batch_size=32, callbacks=callbacks, verbose=1
)
print(f"✅ Best val accuracy: {max(history.history['val_accuracy']):.4f}\n")


# ============================================================
# STEP 6: EVALUATE
# ============================================================
print("📊 STEP 6: Evaluating...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.1f}%\n")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("💾 Saved: confusion_matrix.png")


# ============================================================
# STEP 7: NLP POST-PROCESSING
# ============================================================
print("\n🧠 STEP 7: NLP Post-Processing...")

class NLPPostProcessor:
    COMMAND_ACTIONS = {
        'yes': {'action': 'CONFIRM', 'category': 'response', 'description': 'User confirmed'},
        'no': {'action': 'DENY', 'category': 'response', 'description': 'User denied'},
        'up': {'action': 'MOVE_UP', 'category': 'direction', 'description': 'Move upward'},
        'down': {'action': 'MOVE_DOWN', 'category': 'direction', 'description': 'Move downward'},
        'left': {'action': 'MOVE_LEFT', 'category': 'direction', 'description': 'Move left'},
        'right': {'action': 'MOVE_RIGHT', 'category': 'direction', 'description': 'Move right'},
        'stop': {'action': 'HALT', 'category': 'control', 'description': 'Stop action'},
        'go': {'action': 'START', 'category': 'control', 'description': 'Start/proceed'},
        'hello': {'action': 'GREET', 'category': 'social', 'description': 'Greeting detected'},
        'help': {'action': 'HELP', 'category': 'emergency', 'description': 'Needs assistance'},
    }
    TEMPLATES = {
        'response': "📝 '{cmd}' — {desc}",
        'direction': "🧭 '{cmd}' — {desc}",
        'control': "⚙️ '{cmd}' — {desc}",
        'social': "👋 '{cmd}' — {desc}",
        'emergency': "🚨 '{cmd}' — {desc}",
    }

    def __init__(self, le, threshold=0.4):
        self.le = le
        self.threshold = threshold
        self.history = []

    def process(self, probs):
        top_idx = np.argsort(probs)[::-1][:3]
        top = [{'command': self.le.classes_[i], 'confidence': float(probs[i])} for i in top_idx]
        primary = top[0]
        cmd = primary['command']
        info = self.COMMAND_ACTIONS.get(cmd, {})
        category = info.get('category', 'unknown')
        tmpl = self.TEMPLATES.get(category, "'{cmd}' — {desc}")
        result = {
            'status': 'ACCEPTED' if primary['confidence'] >= self.threshold else 'LOW_CONFIDENCE',
            'top_predictions': top,
            'action': info.get('action', 'UNKNOWN'),
            'nl_response': tmpl.format(cmd=cmd, desc=info.get('description', '')),
            'tokens': word_tokenize(info.get('description', cmd))
        }
        self.history.append({'command': cmd, 'confidence': primary['confidence']})
        return result

    def summary(self):
        if not self.history:
            return "No commands."
        cmds = [h['command'] for h in self.history]
        freq = Counter(cmds).most_common(3)
        avg = np.mean([h['confidence'] for h in self.history])
        return f"Total: {len(self.history)} | Avg conf: {avg:.1%} | Top: {freq}"


nlp = NLPPostProcessor(label_encoder)

# Test predictions with NLP
print("\n🎯 Testing Pipeline with NLP:")
print("=" * 60)
test_idx = np.random.choice(len(all_files), 10, replace=False)
correct = 0
for idx in test_idx:
    fpath = all_files[idx]
    true_label = all_labels[idx]
    feat = extract_features(fpath)
    if feat is None:
        continue
    feat_norm = (feat - X_mean.squeeze(axis=(0, 2))[:, np.newaxis]) / X_std.squeeze(axis=(0, 2))[:, np.newaxis]
    probs = model.predict(feat_norm[np.newaxis, ..., np.newaxis], verbose=0)[0]
    result = nlp.process(probs)
    pred = result['top_predictions'][0]['command']
    is_correct = pred == true_label
    correct += is_correct
    icon = "✅" if is_correct else "❌"
    print(f"  {icon} True: {true_label:>5s} | Pred: {pred:>5s} | {result['nl_response']}")
    print(f"     Tokens: {result['tokens']} | Action: {result['action']}")

print(f"\n📊 Quick test: {correct}/{len(test_idx)} correct")
print(f"📊 {nlp.summary()}")


# ============================================================
# STEP 8: SAVE TRAINING PLOTS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], 'b-', lw=2, label='Train')
axes[0].plot(history.history['val_accuracy'], 'r-', lw=2, label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'], 'b-', lw=2, label='Train')
axes[1].plot(history.history['val_loss'], 'r-', lw=2, label='Val')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("💾 Saved: training_history.png")

print("\n" + "=" * 60)
print("  🎉 COMPLETE PIPELINE FINISHED!")
print("=" * 60)
