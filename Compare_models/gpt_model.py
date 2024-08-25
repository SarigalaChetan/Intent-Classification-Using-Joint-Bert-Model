# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from transformers import GPT2Tokenizer, TFGPT2Model
from urllib.request import urlretrieve

# Downloading the data
SNIPS_DATA_BASE_URL = (
    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
    "master/data/snips/"
)
for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)

# Parsing the data
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

# Encoding using GPT-2 tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence, max_length=max_length)
        token_ids[i, :] = encoded + [0] * (max_length - len(encoded))
    attention_masks = (token_ids != 0).astype(np.int32)

    return {'input_ids': token_ids, 'attention_masks': attention_masks}

encoded_train = encode_dataset(tokenizer, df_train['words'], 45)

# Define intent_map after loading GPT-2 tokenizer
intent_names = df_train['intent_label'].unique()
intent_map = dict((label, idx) for idx, label in enumerate(intent_names))

# Model definition
class IntentClassificationModel(tf.keras.Model):
    def __init__(self, intent_num_labels=None, model_name="gpt2", dropout_prob=0.1):
        super().__init__(name="intent_classification_model")
        self.gpt_model = TFGPT2Model.from_pretrained(model_name)
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
        self.intent_classifier = tf.keras.layers.Dense(intent_num_labels)

    def call(self, inputs, training=False):
        sequence_output = self.gpt_model(inputs['input_ids']).last_hidden_state
        pooled_output = sequence_output[:, -1, :]
        pooled_output = self.dropout(pooled_output, training=training)
        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits

# Create an instance of the model
intent_model = IntentClassificationModel(intent_num_labels=len(intent_names))

# Compile the model
intent_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy('accuracy')]
)

# Training the model
history = intent_model.fit(
    {'input_ids': encoded_train['input_ids'], 'attention_masks': encoded_train['attention_masks']},
    df_train['intent_label'].map(intent_map).values,
    epochs=2,
    batch_size=32
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Get model predictions on the test set
test_predictions = intent_model.predict(
    {'input_ids': encoded_test['input_ids'], 'attention_masks': encoded_test['attention_masks']}
)

# Convert logits to predicted labels
predicted_labels = np.argmax(test_predictions, axis=1)

# True labels
true_labels = df_test['intent_label'].map(intent_map).values

# Calculate accuracy
overall_accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f'Overall Accuracy: {overall_accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get model predicted probabilities on the test set
test_probabilities = intent_model.predict(
    {'input_ids': encoded_test['input_ids'], 'attention_masks': encoded_test['attention_masks']}
)

# Extract the true labels
true_labels = pd.get_dummies(df_test['intent_label']).values

# Compute ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(intent_names)):
    fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], test_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), test_probabilities.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curve
plt.figure(figsize=(10, 8))

# Plot each class's ROC curve
for i in range(len(intent_names)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for {intent_names[i]}')

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})', linestyle='--', color='black')

# Plot random guess line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

# Customize the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

first_sentence = df_train.iloc[0]['words']
print(first_sentence)

tokenizer.tokenize(first_sentence)

