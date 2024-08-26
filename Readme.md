# Optimizing Intent Classification Using Joint BERT: A Comparative Study with GPT and LSTM

## Abstract

Intent classification and slot filling are critical tasks in natural language understanding (NLU), particularly within conversational AI systems like chatbots and virtual assistants. This study explores the application of a Joint BERT model for these tasks, utilizing the SNIPS dataset, which includes sequences of user inputs (seq.in), corresponding tags (seq.out), and intent labels. Initially, I compared the performance of BERT, GPT, and LSTM models on intent classification, finding that BERT outperformed the others in both accuracy and efficiency. To further enhance performance, I concatenated the input sequences and corresponding tags, creating a unified input format that BERT processed. This Joint BERT approach resulted in superior accuracy compared to using intent classification alone. The findings demonstrate the effectiveness of combining intent classification with slot filling in a single model, offering a more robust solution for developing efficient and accurate conversational agents.

## Introduction

As conversational AI technologies advance, accurately discerning user intent has become crucial for delivering relevant and precise interactions. Intent classification is a core component of natural language understanding (NLU) systems, tasked with determining the purpose behind a user's message. For example, if a user requests, "Set an alarm for 7 AM," the system must identify that the user intends to set an alarm at a specified time. In addition to intent classification, slot filling is essential for extracting detailed information from the user's input. In this case, slot filling would identify "7 AM" as the time for the alarm.

However, a significant challenge in intent classification arises when words have multiple meanings depending on the context. Consider the word "book" in the sentences "She wants to book a table for dinner" and "He is reading a new book." In the first sentence, "book" refers to making a reservation, while in the second, it refers to a physical object. Simply identifying the intent might not be enough to distinguish between these different meanings. 

To address this challenge, this project explores the use of Joint BERT, a model that simultaneously handles both intent classification and slot filling. By concatenating the input sentences and their corresponding sequence tags(slot tags), we provide the model with richer contextual information.

For example:
* Sentence 1: "Book a table at the new Italian restaurant."
* Sentence 2: "I just finished reading a great book."

In these sentences, the word "book" appears in different contexts, the challenge is that without additional context, the word "book" alone is ambiguous. This is where slot filling becomes essential. By tagging parts of the sentence with their roles, we provide the model with the necessary context to distinguish between these meanings:

* Sentence 1 Tagging: "Book_O a_O table_B-object_type at_O the_O new_O Italian_O restaurant_B-location"

* Sentence 2 Tagging: "I_O just_O finished_O reading_O a_O great_O book_B-object_type"

In the tagged example, "Book_O", "Book_B_Object_type" helps clarify that in Sentence 1, "book" is related to making a reservation, whereas in Sentence 2, it is associated with a physical object.

Through this methodology, we show that integrating intent classification with slot filling into a single model greatly enhances the system’s ability to understand complex user inputs. This approach is particularly beneficial for creating advanced conversational agents capable of managing a wide range of context-sensitive queries effectively. Prior to finalizing our model, we evaluated three different models—BERT, GPT, and LSTM. Among these, BERT delivered the best performance in terms of accuracy and efficiency. Consequently, we selected BERT and adapted it into our proposed Joint BERT model, optimizing it for both intent classification and slot filling.

## Methodology

### Datasets:

For this project, I utilized the [SNIPS dataset](), a well-established benchmark for intent classification and slot filling tasks. The dataset comprises three key components: seq.in (input sentences), seq.out (corresponding sequence tags), and labels (intent labels for each input sentence). Together, these components provide a comprehensive framework for training a model to accurately interpret and respond to user queries.

Due to hardware constraints, specifically an 8GB RAM limitation, I focused solely on the training portion of the dataset. Instead of using the provided test data, I split the training data into training and testing subsets. This approach allowed me to manage computational resources effectively while still enabling both model training and evaluation.

### Preprocessing:

In preparing the SNIPS dataset for model training, I implemented several key preprocessing steps to enhance the quality and consistency of the data. First, I concatenated the seq.in (input sentences) with the corresponding seq.out (sequence tags) to create a unified input format. This step was crucial for providing the model with richer context, particularly in cases where words could have multiple meanings depending on the surrounding text. To further distinguish between concatenated elements, I added a comma after every concatenated word, ensuring clear separation and easier interpretation by the model.

After concatenation, I removed duplicate entries to eliminate redundancy, ensuring that the dataset comprised only unique examples. This step was important for preventing the model from being influenced by repeated data, thereby promoting better generalization during testing.

Finally, I addressed the issue of data imbalance among the different intent labels. Some labels had significantly more unique sentences than others, which could lead to biased model performance. To mitigate this, I balanced the dataset by identifying the label with the fewest unique sentences and then reducing the number of unique sentences for all other labels to match this minimum. This balancing step helped ensure that the model was exposed to an equal number of examples across all intent categories, leading to more consistent and reliable performance across the board.


### Model Architecture:

#### Initial Experiment:
 So the Initial experiment is to find the best model among this three(BERT, GPT and LSTM) so that i can choose the best model and I can develop the model with the proposed idea.

##### BERT:
<br>
BERT, which stands for Bidirectional Encoder Representations from Transformers, is known for its ability to understand the context of words by analyzing them in both directions within a sentence. This capability is especially important for tasks like intent classification, where the meaning of words can depend heavily on their surrounding context.

I used the bert-base-cased model for this project. The first step was to encode the text sequences using a tokenizer, which converts the sentences into numerical token IDs and creates attention masks. These masks help the model distinguish between actual words and padding in the input data. The encoded sequences were then passed through the BERT model to generate context-aware representations of the text.

The model architecture included the BERT base model, with an additional dropout layer to reduce overfitting, and a dense layer for classifying the intents. The model was trained using the Adam optimizer, with a learning rate set at 3e-5, and the loss function used was sparse categorical cross-entropy, which is appropriate for this type of multi-class classification problem.

I trained the model for two epochs with a batch size of 32, and a portion of the training data was set aside for validation. This approach allowed me to monitor the model’s performance and make necessary adjustments during training. BERT showed strong results in accurately identifying user intents, proving to be an effective choice for this task.

##### GPT:
<br>
GPT-2 (Generative Pre-trained Transformer 2) is a powerful language model known for its ability to generate coherent text and understand contextual information based on its pre-training.

For this project, I utilized the gpt2 variant of the model. The first step was to encode the text sequences using the GPT-2 tokenizer, which converts sentences into numerical token IDs. I ensured that all sequences were padded to a maximum length to maintain consistency in input size. The attention masks were created to differentiate between actual tokens and padding.

The model architecture was built using the TFGPT2Model from TensorFlow. I added a dropout layer to help prevent overfitting and a dense layer for the intent classification task. During training, the model outputs from GPT-2 are processed to obtain the last hidden state of the sequences. This state is pooled (by selecting the last token of the sequence) and passed through the dropout layer before being fed into the dense layer for classification.

The model was compiled using the Adam optimizer with a learning rate of 3e-5 and sparse categorical cross-entropy as the loss function, suitable for multi-class classification. The training process included two epochs with a batch size of 32.

Overall, the GPT-2 model demonstrated solid performance in classifying user intents. Its ability to leverage pre-trained contextual representations proved beneficial, although the computational demands were higher compared to some other models.

##### LSTM:    
<br>
LSTM (Long Short-Term Memory) model for intent classification. LSTM networks are well-suited for processing sequential data, such as text, because they can capture dependencies and patterns over time.

The model architecture included an embedding layer, which transformed token indices into dense vectors of 50 dimensions, capturing semantic relationships between words. Following the embedding layer, the model featured two LSTM layers, each with 100 units. The first LSTM layer returned sequences that were fed into the second LSTM layer, allowing the network to capture complex sequential dependencies in the data.

To reduce overfitting, a dropout layer with a rate of 0.5 was incorporated, randomly setting a fraction of input units to zero during training. Finally, a dense layer with a softmax activation function was used to output the probability distribution over the possible intent categories.

The model was compiled with the Adam optimizer, set to a learning rate of 1e-3, and sparse categorical cross-entropy as the loss function. Training was conducted for 5 epochs with a batch size of 32.

The LSTM model demonstrated its capability in handling text sequences effectively, providing valuable insights into intent classification by learning and leveraging sequential patterns in the data.

#### Joint Bert Approch:
<br>
After preprocessing the data, which included concatenating seq.in and seq.out sequences, adding commas between words, removing duplicates, and balancing the dataset, I proceeded with training the model. The preprocessing steps were crucial in preparing the data for effective input into the BERT model. The concatenation of sequences allowed the model to capture both the intent and slot information, while the added commas helped in distinguishing different parts of the input. Removing duplicates and balancing the dataset ensured that the model was trained on a representative and diverse set of examples.

The BertTokenizer was used to tokenize the text sequences, with padding and truncation applied to standardize the input length. The encoded data was then converted into tensors for model input. Labels were processed using LabelEncoder, transforming categorical intent labels into numerical values that the model could utilize during training.

A custom dataset class, IntentSlotDataset, was created to handle the tokenized encodings and labels, providing methods to retrieve individual data points and their corresponding labels. This class facilitated the loading of data into the model.

The model itself was based on BertForSequenceClassification, with a custom extension to include a dropout layer for regularization. This model was initialized with the pre-trained bert-base-uncased and was configured to handle the number of intent labels present in the dataset. The training process employed the AdamW optimizer with a learning rate of 5e-5, and the model was trained for 2 epochs.

By incorporating these preprocessing steps and utilizing BERT's advanced capabilities, the model effectively learned to classify intents with high accuracy, demonstrating its strength in understanding and processing complex text sequences.

### Evaluation Metrics:

The evaluation process involved several steps, including metric calculations and visualizations to understand the model's performance comprehensively.

#### Metrics Calculation:

* **Predictions and True Labels**: The model's predictions were obtained by passing the encoded validation data through the trained model. The output logits were converted to class predictions using the argmax function. These predictions were then compared with the true intent labels from the validation set.
* **Accuracy**: The accuracy of the model was calculated using the accuracy_score function from sklearn.metrics, which measures the proportion of correctly classified instances out of the total instances.
* **Precision, Recall, and F1 Score**: To gain insights into the model's performance beyond overall accuracy, precision, recall, and F1 score were calculated. Precision indicates how many of the predicted positive instances were actually correct, recall measures how many of the actual positive instances were correctly predicted, and F1 score provides a balanced measure of precision and recall. These metrics were computed using the precision_score, recall_score, and f1_score functions with a weighted average to account for class imbalances.
#### ROC Curve Analysis:

* One-Hot Encoding: The true labels were converted to a one-hot encoded format using label_binarize to facilitate ROC curve computation for each class.
* Predicted Probabilities: The model provided predicted probabilities for each class, which were used to compute the ROC curve and area under the curve (AUC) for each class.
* ROC Curve Computation and Plotting: The false positive rate (FPR) and true positive rate (TPR) were computed for each class, and the AUC was calculated to quantify the model's ability to distinguish between classes. The ROC curves were plotted using matplotlib, providing a visual representation of the model's performance across different classes. Each curve showed the trade-off between sensitivity and specificity for a given class, with the AUC value indicating the overall performance.

By analyzing these metrics and visualizations, the evaluation demonstrated the model's effectiveness in classifying intents. The accuracy, precision, recall, and F1 score provided a clear understanding of how well the model performed on the validation data, while the ROC curves and AUC values offered insights into the model's capability to handle different classes.

### Results 

In our evaluation, we compared three models—BERT, GPT, and LSTM—based on their performance in intent classification tasks. The results were as follows:

| Model | Accuracy | Precision | Recall | F1 Score | Time Taken |
|-------|----------|-----------|--------|----------|---------|
| BERT  | 0.9871   | 0.9872    | 0.9871 | 0.9871   | 150.55 mins |
| GPT   | 0.9871   | 0.9879    | 0.9871 | 0.9872   | 203.47 mins |
| LSTM  | 0.9600   | 0.9615    | 0.9600 | 0.9602   | 40.00 mins  |

Among these models, BERT emerged as the most efficient and effective choice. Although GPT showed slightly better precision, recall, and F1 score, it required significantly more computational resources and time compared to BERT. LSTM, while faster, did not perform as well as BERT and GPT in terms of accuracy and F1 score.

**Joint BERT Model Performance**:

The Joint BERT model, which combines BERT with intent classification and slot filling, achieved an impressive performance with the following metrics:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
|Joint B| 0.9962   | 0.9968    | 0.9963 | 0.9962   |

The Joint BERT model demonstrated superior accuracy and overall performance compared to the individual models, validating its effectiveness in handling both intent classification and slot filling tasks.

In summary, BERT was chosen for its balance of performance and efficiency among the initial models. Incorporating BERT into the joint model approach resulted in even better performance, making the Joint BERT model the most effective solution for the task at hand.


