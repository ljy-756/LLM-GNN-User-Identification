# model/bert_classifier.py

from transformers import BertForSequenceClassification

class BertClassifier(BertForSequenceClassification):
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__(BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).config)
