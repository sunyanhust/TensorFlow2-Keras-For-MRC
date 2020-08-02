"""
modified from https://keras.io/examples/nlp/text_extraction_with_bert/
"""

import json
import re
import string
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel, BertConfig
from utils import tensorflow_defult_setting, SendEmail
from utils import get_lr_metric


class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.input_ids = None
        self.token_type_ids = None
        self.attention_mask = None
        self.start_token_idx = None
        self.end_token_idx = None
        self.context_token_to_char = None

    def preprocess(self, tokenizer, max_len):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(data_path, tokenizer, max_len):
    with open(data_path) as f:
        raw_data = json.load(f)

    squad_examples = []
    for item in tqdm(raw_data["data"]):
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                start_char_idx = qa["answers"][0]["answer_start"]
                squad_eg = SquadExample(
                    question, context, start_char_idx, answer_text, all_answers
                )
                squad_eg.preprocess(tokenizer=tokenizer, max_len=max_len)
                squad_examples.append(squad_eg)
    return squad_examples


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if not item.skip:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


def build_model(model_path):
    # BERT encoder
    encoder = TFBertModel.from_pretrained(model_path)

    # QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    outputs = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )
    # https://github.com/huggingface/transformers/issues/6029
    _, _, hidden_states = outputs[0], outputs[1], outputs[2] 
    sequence_output = layers.Concatenate(axis=-1)([hidden_states[-1], hidden_states[-2], hidden_states[-3]])
    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax, name="start")(start_logits)
    end_probs = layers.Activation(keras.activations.softmax, name="end")(end_logits)

    bert_model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
        name="BERTForQuestionAnswer"
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    bert_model.compile(optimizer=optimizer, loss=[loss, loss], metrics=['acc',
                                                                        get_lr_metric(optimizer)])
    return bert_model


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


class ExactMatch(keras.callbacks.Callback):
    """
    Each `SquadExample` object contains the character level offsets for each token
    in its input paragraph. We use them to get back the span of text corresponding
    to the tokens between our predicted start and end tokens.
    All the ground-truth answers are also present in each `SquadExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, eval_x, eval_y):
        super().__init__()
        self.x_eval = eval_x
        self.y_eval = eval_y

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip is False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch + 1}, exact match score={acc:.3f}")


class F1Score(keras.callbacks.Callback):

    def __init__(self, eval_x, eval_y):
        super().__init__()
        self.x_eval = eval_x
        self.y_eval = eval_y

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip is False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            prediction_tokens = normalized_pred_ans.split()

            scores_for_ground_truths = []
            for ground_truth in normalized_true_ans:
                ground_truth_tokens = ground_truth.split()
                common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    f1 = 0
                else:
                    precision = 1.0 * num_same / len(prediction_tokens)
                    recall = 1.0 * num_same / len(ground_truth_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
                scores_for_ground_truths.append(f1)

            f1 = max(scores_for_ground_truths)
            count += f1
        f1 = count / len(self.y_eval[0])
        print(f"epoch={epoch + 1}, f1 score={f1:.3f}")


if __name__ == '__main__':
    tensorflow_defult_setting()

    train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    train_path = keras.utils.get_file("train-v1.1.json", train_data_url, cache_dir='.')
    eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    eval_path = keras.utils.get_file("dev-v1.1.json", eval_data_url, cache_dir='.')

    max_len = 384
    bert_model_path = "bert_model/bert-base-uncased"
    configuration = BertConfig.from_pretrained(bert_model_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    tokenizer.save_pretrained(bert_model_path)

    tokenizer = BertWordPieceTokenizer("bert_model/bert-base-uncased/vocab.txt", lowercase=True)
    train_squad_examples = create_squad_examples(data_path=train_path,
                                                 tokenizer=tokenizer,
                                                 max_len=max_len)
    x_train, y_train = create_inputs_targets(train_squad_examples)
    print(f"{len(train_squad_examples)} training points created.")

    eval_squad_examples = create_squad_examples(data_path=eval_path,
                                                tokenizer=tokenizer,
                                                max_len=max_len)
    x_eval, y_eval = create_inputs_targets(eval_squad_examples)
    print(f"{len(eval_squad_examples)} evaluation points created.")

    model = build_model(bert_model_path)
    model.summary()
    
    checkpoint = keras.callbacks.ModelCheckpoint("./", 
                                                 monitor='val_loss', verbose=True, 
                                                 save_best_only=True, save_weights_only=True,
                                                 mode='auto')
    model.fit(
        x_train,
        y_train,
        epochs=3,  # For demonstration, 3 epochs are recommended
        verbose=1,
        batch_size=6,
        callbacks=[ExactMatch(x_eval, y_eval),
                   F1Score(x_eval, y_eval),
                   SendEmail(), 
                   checkpoint],
    )

