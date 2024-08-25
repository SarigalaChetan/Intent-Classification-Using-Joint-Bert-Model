pip install transformers

# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from transformers import BertTokenizer, TFBertModel
from urllib.request import urlretrieve

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

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
print(f'First line of training set: {lines_train[0]}.')

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

parse_line(lines_train[0])

print(Path('vocab.intent').read_text('utf-8'))

print(Path('vocab.slot').read_text('utf-8'))

parsed = [parse_line(line) for line in lines_train]
df_train = pd.DataFrame([p for p in parsed if p is not None])

df_train.head(5)

df_train.intent_label.value_counts()

lines_validation = Path('valid').read_text('utf-8').strip().splitlines()
lines_test = Path('test').read_text('utf-8').strip().splitlines()

df_validation = pd.DataFrame([parse_line(line) for line in lines_validation])
df_test = pd.DataFrame([parse_line(line) for line in lines_test])

model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

first_sentence = df_train.iloc[0]['words']
print(first_sentence)

tokenizer.tokenize(first_sentence)

tokenizer.encode(first_sentence)

tokenizer.decode(tokenizer.encode(first_sentence))

train_sequence_lengths = [len(tokenizer.encode(text))
                          for text in df_train['words']]
plt.hist(train_sequence_lengths, bins=30)
plt.title(f'Max sequence length: {max(train_sequence_lengths)}')
plt.xlabel('Length')
plt.ylabel('Count')
plt.show()

print(f'Vocabulary size: {tokenizer.vocab_size} words.')
bert_vocab_items = list(tokenizer.vocab.items())

def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length),
                         dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    
    return {'input_ids': token_ids, 'attention_masks': attention_masks}

encoded_train = encode_dataset(tokenizer, df_train['words'], 45)
encoded_validation = encode_dataset(tokenizer, df_validation['words'], 45)
encoded_test = encode_dataset(tokenizer, df_test['words'], 45)

encoded_train['input_ids']
encoded_train['attention_masks']

intent_names = Path('vocab.intent').read_text('utf-8').split()
intent_map = dict((label, idx) for idx, label in enumerate(intent_names))

intent_map

intent_train = df_train['intent_label'].map(intent_map).values
intent_validation = df_validation['intent_label'].map(intent_map).values
intent_test = df_test['intent_label'].map(intent_map).values

base_bert_model = TFBertModel.from_pretrained('bert-base-cased')
base_bert_model.summary()

outputs = base_bert_model(input_ids=encoded_validation['input_ids'], attention_mask=encoded_validation['attention_masks'])
print(f'Shape of the first output of the BERT model: {outputs[0].shape}.')

print(f'Shape of the second output of the BERT model: {outputs[1].shape}.')
# Define IntentClassification model
class IntentClassificationModel(tf.keras.Model):
    def __init__(self, intent_num_labels=None,
                 model_name='bert-base-cased',
                 dropout_prob=0.1):
        super().__init__(name='joint_intent_slot')
      
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        
    
        self.intent_classifier = Dense(intent_num_labels)
        
    def call(self, inputs, **kwargs):
        
        sequence_output, pooled_output = self.bert(inputs['input_ids'], attention_mask=inputs['attention_masks']).values()
        
        
        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits

intent_model = IntentClassificationModel(intent_num_labels=len(intent_map))


intent_model.compile(optimizer=Adam(learning_rate=3e-5, epsilon=1e-08),
                     loss=SparseCategoricalCrossentropy(from_logits=True),
                     metrics=[SparseCategoricalAccuracy('accuracy')])


history = intent_model.fit(
    {'input_ids': encoded_train['input_ids'], 'attention_masks': encoded_train['attention_masks']},
    intent_train,
    epochs=2,
    batch_size=32,
    validation_data=(
        {'input_ids': encoded_validation['input_ids'], 'attention_masks': encoded_validation['attention_masks']},
        intent_validation
    )
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate on the validation set
predictions = np.argmax(intent_model.predict(encoded_validation), axis=1)

# Convert one-hot encoded labels back to original labels
original_labels_validation = df_validation['intent_label'].map(intent_map).values

# Calculate metrics
accuracy = accuracy_score(original_labels_validation, predictions)
precision = precision_score(original_labels_validation, predictions, average='weighted')
recall = recall_score(original_labels_validation, predictions, average='weighted')
f1 = f1_score(original_labels_validation, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Convert labels to one-hot encoded format
one_hot_labels_validation = label_binarize(original_labels_validation, classes=np.unique(original_labels_validation))

# Get predicted probabilities for each class
probabilities = intent_model.predict(encoded_validation)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(intent_map)):
    fpr[i], tpr[i], _ = roc_curve(one_hot_labels_validation[:, i], probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(len(intent_map)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

