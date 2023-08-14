# Binary Sentiment Classification Analysis - BERT for Sentence Classification
Training BERT model to identify positive/negative sentiment in Amazon reviews

## Potential Uses
Binary classification of text is extremely powerful and relevant in today's business. One idea I had for binary sentiment classification is to analyze manager reviews of employees in a large company and identifying potential racial or gender bias. Being proactive in identifying potential bias can improve workplace satisfaction, inclusion, retention, and even avoid lawsuits.

## BERT model
BERT, or Bidirectional Encoder Representations from Transformers, is a deep learning model for natural language processing (NLP). It was developed by Google AI in 2018 and has since become one of the most popular NLP models, achieving state-of-the-art results on a variety of tasks, including question answering, natural language inference, and sentiment analysis. BERT is based on the transformer architecture, which is a neural network architecture that is particularly well-suited for NLP tasks. Transformer models are able to learn long-range dependencies between words in a sentence, which is essential for understanding the meaning of text. BERT is pre-trained on a massive dataset of text and code, which allows it to learn general language representations. This pre-training makes BERT very efficient at learning new tasks, as it does not need to be trained from scratch on each new dataset.

## Project Scope
I wanted to train the BERT model on my GPU to identify bias on something simple such as Amazon reviews. I utlized a datasource posted by Kaggle user Kritanjalijain and constructed by Xiang Zhangwhich ahd over 1,800,000 training reviews and 400,000 testing reviews. By using this dataset, I would be able to create a trained model that would identify positive and negative bias.

## Application
The original dataset proved to be too big for my computer to handle, so I trimmed it down to randomly selected sample of 5% of the training data and 10% of the testing data and ran 3 epochs with batch sizes of 20. With this, I was able to get an accuracy of 0.94 and validation loss of 0.27. After testing the data, I tested for accuracy, prevision, and recall. I got an accuracy score of 0.891, precision score of 0.9027, and recall score of 0.8784.

## Conclusion
The model performed well, but it is noteable that the recall score is slightly lower than the precision score, suggesting that the model may be lacking positive examples compared to negative ones. This imbalance of the data could be due to the randomly selected samples. Using the full training and testing datasets could correct this issue. Overall, I think this test went well.
