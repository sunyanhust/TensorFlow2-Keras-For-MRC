import logging
import numpy as np
from typing import List
from pytorch_transformers import *
from utils import load_json
from tensorflow.keras import models

logging.basicConfig(level=logging.ERROR)


class QAModel(object):

    def __init__(self, max_sent_len=16, model_path="bert-base-uncased"):
        self.model_path = model_path
        self.tokenizer = BertWordPieceTokenizer("bert_model/bert-base-uncased/vocab.txt", lowercase=True)
        self.bert = model = models.load_model('model.h5')
        self.max_sent_len = max_sent_len
    
    
    def normalize_text(text):
        text = text.lower()

        # Remove punctuations
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)

        # Remove articles
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = re.sub(regex, " ", text)

        # Remove extra white space
        text = " ".join(text.split())
        return text

    def predict(self, batch: List[List[str,str]]):
        """predict masked word"""
        batch_inputs = []
        context_offsets =[]

        for texts in batch:
            context, question = texts
            tokenized_context = self.tokenizer.encode(context)
            tokenized_question = self.tokenizer.encode(question)

            input_ids = tokenized_context.ids + tokenized_question.ids[1:]
            token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
            attention_mask = [1] * len(input_ids)

            padding_length = self.max_sent_len - len(input_ids)
            if padding_length > 0:  # pad
                input_ids = input_ids + ([0] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
            elif padding_length < 0:  # skip
                return 

            batch_inputs.append([np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)])
            context_offsets.append(tokenized_context.offsets) 

        
        pred_start, pred_end = self.model.predict(batch_inputs)
        
        prediction_scores = self.bert(tokens_tensor)[0]
        
        batch_outputs = []

        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            start = np.argmax(start)
            end = np.argmax(end)
            offsets = context_offsets[idx] 
            if start >= len(offsets):
                batch_outputs.append("no answer!")
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = batch_inputs[idx][0][pred_char_start:pred_char_end]
            else:
                pred_ans = batch_inputs[idx][0][pred_char_start:]

            normalized_pred_ans = self.normalize_text(pred_ans)
            batch_outputs.append(normalized_pred_ans)

        return batch_outputs




if __name__ == "__main__":
    batch = ["twinkle twinkle [MASK] star.",
             "Happy birthday to [MASK].",
             'the answer to life, the [MASK], and everything.']
    model = QAModel()
    outputs = model.predict(batch)
    print(outputs)