# Binary Sentiment Classification Analysis - BERT for Sentence Classification
Fine-tuning BERT model to identify positive/negative sentiment in Amazon reviews.

# Introduction
This project focuses on the fine-tuning of a BERT pre-trained model for the task of sentiment analysis on customer reviews. BERT (Bidirectional Encoder Representations from Transformers) models are originally trained on a vast corpus of text data to acquire comprehensive language understanding. In this endeavor, I aim to leverage the power of BERT by fine-tuning it specifically for sentiment analysis on reviews with the objective of achieving high accuracy and robust performance.

The primary objective is to assess how effectively the BERT model can be adapted and fine-tuned to accurately classify reviews as either positive or negative. Through this data science endeavor, we will explore the model's ability to capture nuanced sentiment patterns within the context of customer reviews. The project involves tokenization, data preprocessing, model training, and rigorous evaluation to gauge its performance on this specific NLP task.

## BERT model
BERT, or Bidirectional Encoder Representations from Transformers, is a deep learning model for natural language processing (NLP). It was developed by Google AI in 2018 and has since become one of the most popular NLP models, achieving state-of-the-art results on a variety of tasks, including question answering, natural language inference, and sentiment analysis. BERT is based on the transformer architecture, which is a neural network architecture that is particularly well-suited for NLP tasks. Transformer models are able to learn long-range dependencies between words in a sentence, which is essential for understanding the meaning of text. BERT is pre-trained on a massive dataset of text and code, which allows it to learn general language representations. This pre-training makes BERT very efficient at learning new tasks, as it does not need to be trained from scratch on each new dataset.
<p align="center">
  <img src="https://stanford-cs324.github.io/winter2022/lectures/images/bert.png" width="50%">
</p>

## Tokenization

### Introduction
Tokenization is a crucial preprocessing step in natural language processing (NLP) tasks. It involves converting raw text into smaller units called tokens, which can be words, subwords, or characters. In this section, we will discuss the tokenization process used for a binary sentiment classifier based on Amazon reviews.

### Dataset Description
The Amazon reviews polarity dataset used in this project was constructed by Xiang Zhang and is commonly used as a text classification benchmark. It contains reviews from Amazon and is split into positive and negative sentiment classes. For this report, a subset of the dataset provided by Kaggle user 'kritanjalijain' was used. Due to computational constraints, only a random 10% of the data was selected for training and validation.

### Tokenization Process
To prepare the data for training, I followed these tokenization steps:

1. Load the dataset from a CSV file, containing columns 'label', 'title', and 'review'.
2. Filter the dataset to include only reviews with a maximum length of 200 characters.
3. Randomly select 10% of the filtered dataset for training and validation.

To visualize the tokenization process, I took a sample review and demonstrated the following:

- The original sentence.
- The sentence split into tokens.
- The sentence mapped to token IDs.

Additionally, I determined the maximum review length in the dataset, which was found to be **200 tokens**.

## Pre-Training Setup

### Data Preprocessing
After tokenization, the dataset was further processed for model input. The following steps were performed:

1. Truncate reviews longer than 200 tokens.
2. Encode the text using the BERT tokenizer, adding '[CLS]' and '[SEP]' tokens, and ensuring all sentences were padded to a length of 200 tokens.
3. Create PyTorch tensors for input IDs and attention masks.
4. Adjust the labels to convert the dataset to binary sentiment classification (0 for negative, 1 for positive).

### Data Splitting
The dataset was divided into training and validation sets using an 90%-10% split. This resulted in **38,394 training samples** and **4,267 validation samples**.

### Batch Size
A batch size of 20 was chosen for training.

## Training

### Model Training
Training the binary sentiment classifier involved the following steps:

1. Define the optimizer (AdamW) and loss function (Binary Cross-Entropy Loss).
2. Create a learning rate scheduler.
3. Train the model for **3 epochs**.

### Training Metrics
During training, several metrics were tracked:

- Training loss: The average loss calculated over all training batches.
- Training time: The time taken for each epoch of training.
- Validation loss: The average loss calculated over all validation batches.
- Validation accuracy: The accuracy of the model on the validation set.

<b>Training Summary</b>
| epoch | Training Loss | Evaluation Loss | Evaluation Accuracy | Training Time | Validation Time |
|-------|---------------|-----------------|---------------------|---------------|-----------------|
| 1     | 0.21          | 0.18            | 0.94                | 0:12:55       | 0:00:27         |
| 2     | 0.10          | 0.20            | 0.94                | 0:13:30       | 0:00:30         |
| 3     | 0.04          | 0.27            | 0.94                | 0:12:58       | 0:00:28         |

## Model Testing

### Prediction on Test Set
The model was evaluated on the test set using the following steps:

1. Setting the model to evaluation mode.
2. Running predictions on the test dataset, batch by batch.
3. Storing the model's predictions and true labels.

The prediction process resulted in **predicted logits** for each review.

### Calculating Accuracy, Precision, and Recall
To assess the model's performance, accuracy, precision, and recall were calculated using the following definitions:

- Accuracy: The proportion of correctly predicted labels among all test samples.
- Precision: The proportion of true positives among all positive predictions.
- Recall: The proportion of true positives correctly predicted among all actual positives.

For this evaluation, these metrics were calculated individually for each batch of test data and then averaged.

### Results
The evaluation metrics for the model on the test dataset are as follows:

<b>Testing Model</b>
| Metric           | Score    |
|------------------|----------|
| Accuracy         | 0.891    |
| Precision        | 0.9027   |
| Recall           | 0.8784   |

These metrics indicate that the binary sentiment classifier performs well on the test dataset, achieving high accuracy, precision, and recall, which are key indicators of its effectiveness in classifying Amazon reviews into positive and negative sentiments.

## Conclusion
The combination of tokenization, training, and testing has proven the effectiveness of the binary sentiment classifier in classifying Amazon reviews into positive and negative sentiments. It not only performed well during training and validation but also demonstrated its ability to generalize to new data during testing.

This model can be a valuable tool for sentiment analysis tasks on Amazon reviews, providing accurate and reliable results. Further fine-tuning or optimization may be explored based on specific application requirements, but the foundation for sentiment classification is solid and promising.

## Potential Uses
Binary classification of text is extremely powerful and relevant in today's business. One idea I had for binary sentiment classification is to analyze manager reviews of employees in a large company and identifying potential racial or gender bias. Being proactive in identifying potential bias can improve workplace satisfaction, inclusion, retention, and even avoid lawsuits.
