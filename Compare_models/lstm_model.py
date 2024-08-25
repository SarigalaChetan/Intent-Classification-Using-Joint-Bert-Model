import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from urllib.request import urlretrieve


SNIPS_DATA_BASE_URL = (
    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
    "master/data/snips/"
)
for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)


lines_train = Path('train').read_text('utf-8').strip().splitlines()

def parse_line(line):
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(':', 1)[0] for item in items]
    word_labels = [item.rsplit(':', 1)[1] for item in items]
    return {
        'intent_label': intent_label,
        'words': " ".join(words),
        'words_label': " ".join(word_labels),
        'length': len(words)
    }

parsed = [parse_line(line) for line in lines_train]
df_train = pd.DataFrame([p for p in parsed if p is not None])


intent_map = {intent: idx for idx, intent in enumerate(df_train['intent_label'].unique())}
intent_names = list(intent_map.keys())


max_words = 5000
max_sequence_length = max(df_train['length'])

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['words'])


intent_train = np.array([intent_map[label] for label in df_train['intent_label']])
sequences_train = tokenizer.texts_to_sequences(df_train['words'])
padded_train = pad_sequences(sequences_train, maxlen=max_sequence_length, padding='post', truncating='post')

intent_slot_lstm_model = Sequential()
intent_slot_lstm_model.add(Embedding(input_dim=max_words, output_dim=50, input_length=max_sequence_length))
intent_slot_lstm_model.add(LSTM(100, return_sequences=True))
intent_slot_lstm_model.add(LSTM(100))
intent_slot_lstm_model.add(Dropout(0.5))
intent_slot_lstm_model.add(Dense(len(intent_map), activation='softmax'))

intent_slot_lstm_model.compile(optimizer=Adam(learning_rate=1e-3),
                               loss=SparseCategoricalCrossentropy(from_logits=False),
                               metrics=[SparseCategoricalAccuracy('accuracy')])

history_intent_slot_lstm = intent_slot_lstm_model.fit(padded_train, intent_train,
                                                      epochs=5, batch_size=32)


# Input text
input_text = "Add RRR song to the plalist "


input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post', truncating='post')


predicted_probabilities = intent_slot_lstm_model.predict(padded_input)
predicted_intent_index = np.argmax(predicted_probabilities)
predicted_intent = intent_names[predicted_intent_index]

print("Predicted Intent:", predicted_intent)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


lines_test = Path('test').read_text('utf-8').strip().splitlines()

parsed_test = [parse_line(line) for line in lines_test]
df_test = pd.DataFrame([p for p in parsed_test if p is not None])


intent_test = np.array([intent_map[label] for label in df_test['intent_label']])
sequences_test = tokenizer.texts_to_sequences(df_test['words'])
padded_test = pad_sequences(sequences_test, maxlen=max_sequence_length, padding='post', truncating='post')


intent_predictions = intent_slot_lstm_model.predict(padded_test)
intent_predictions_classes = np.argmax(intent_predictions, axis=1)

# Convert intent labels to one-hot encoding for ROC AUC
intent_test_one_hot = label_binarize(intent_test, classes=list(range(len(intent_map))))

# Evaluation metrics
accuracy = accuracy_score(intent_test, intent_predictions_classes)
precision = precision_score(intent_test, intent_predictions_classes, average='weighted')
recall = recall_score(intent_test, intent_predictions_classes, average='weighted')
f1 = f1_score(intent_test, intent_predictions_classes, average='weighted')
roc_auc = roc_auc_score(intent_test_one_hot, intent_predictions, average='weighted', multi_class='ovr')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Area under ROC Curve: {roc_auc:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(8, 8))

for i in range(len(intent_map)):
    fpr, tpr, _ = roc_curve(intent_test_one_hot[:, i], intent_predictions[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve for each class')
ax.legend(loc="lower right")

plt.show()
