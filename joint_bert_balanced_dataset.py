from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
import pandas as pd

seq_in_path = "/content/seq.in"
seq_out_path = "/content/seq.out"
labels_path = "/content/label"


seq_in_data = pd.read_csv(seq_in_path, header=None, names=['seq_in'])
seq_out_data = pd.read_csv(seq_out_path, header=None, names=['seq_out'])
labels_data = pd.read_csv(labels_path, header=None, names=['labels'])


combined_data = pd.concat([seq_in_data, seq_out_data, labels_data], axis=1)


combined_data['combined'] = combined_data.apply(
    lambda row: ' '.join(f"{token}_{label}" for token, label in zip(row['seq_in'].split(), row['seq_out'].split())), axis=1
)
df = combined_data[['combined', 'labels']]
df
df.groupby('labels').describe()

df_no_duplicates = df.drop_duplicates(subset='combined', keep='first')

df_no_duplicates.groupby('labels').describe()


df_SearchScreeningEvent = df_no_duplicates[df_no_duplicates['labels']=='SearchScreeningEvent']
df_SearchScreeningEvent.shape

df_AddToPlaylist = df_no_duplicates[df_no_duplicates['labels']=='AddToPlaylist']
df_RateBook = df_no_duplicates[df_no_duplicates['labels']=='RateBook']
df_SearchCreativeWork = df_no_duplicates[df_no_duplicates['labels']=='SearchCreativeWork']
df_BookRestaurant = df_no_duplicates[df_no_duplicates['labels']=='BookRestaurant']
df_GetWeather = df_no_duplicates[df_no_duplicates['labels']=='GetWeather']
df_PlayMusic = df_no_duplicates[df_no_duplicates['labels']=='PlayMusic']

df_AddToPlaylist_downsize = df_AddToPlaylist.sample(df_SearchScreeningEvent.shape[0])
df_RateBook_downsize = df_RateBook.sample(df_SearchScreeningEvent.shape[0])
df_SearchCreativeWork_downsize = df_SearchCreativeWork.sample(df_SearchScreeningEvent.shape[0])
df_BookRestaurant_downsize = df_BookRestaurant.sample(df_SearchScreeningEvent.shape[0])
df_GetWeather_downsize = df_GetWeather.sample(df_SearchScreeningEvent.shape[0])
df_PlayMusic_downsize = df_PlayMusic.sample(df_SearchScreeningEvent.shape[0])

df_balanced = pd.concat([df_SearchScreeningEvent, df_AddToPlaylist_downsize, df_RateBook_downsize, df_SearchCreativeWork_downsize, df_BookRestaurant_downsize, df_GetWeather_downsize, df_PlayMusic_downsize])
df_balanced['labels'].value_counts()

train_data, val_data = train_test_split(df_balanced, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_batch(batch):
    return tokenizer(batch['combined'].tolist(), padding=True, truncation=True, return_tensors='pt')

train_encodings = tokenize_batch(train_data)
val_encodings = tokenize_batch(val_data)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['labels'])
val_labels = label_encoder.transform(val_data['labels'])


class IntentSlotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentSlotDataset(train_encodings, train_labels)
val_dataset = IntentSlotDataset(val_encodings, val_labels)

from transformers import BertForSequenceClassification

class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = torch.nn.Dropout(0.1)

num_labels = len(label_encoder.classes_)
model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)



from sklearn.metrics import accuracy_score, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

# Evaluation
model.eval()
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy}')


predicted_labels = label_encoder.inverse_transform(all_preds)
true_labels = label_encoder.inverse_transform(all_labels)


print(classification_report(true_labels, predicted_labels))

from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
user_input = " book an ticket for bahubali movie in vijayawada"


prediction = classifier(user_input)

predicted_label = prediction[0]['label']

print(f"Predicted Intent: {predicted_label}")

