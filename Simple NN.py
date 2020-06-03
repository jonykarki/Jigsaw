# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings("ignore")

import os
import re

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt

import transformers
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.callbacks as cb
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from sklearn import metrics


INPUT_PATH = "/content"
TRAIN_PATH = os.path.join(INPUT_PATH, "jigsaw-toxic-comment-train.csv")
VAL_PATH = os.path.join(INPUT_PATH, "validation.csv")
TEST_PATH = os.path.join(INPUT_PATH, "test.csv")

train = pd.read_csv(TRAIN_PATH)
valid = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)


def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub("\\n", " ", str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*", "", str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)", "", str(x)))
    return text


valid["comment_text"] = clean(valid["comment_text"])
test["content"] = clean(test["content"])
train["comment_text"] = clean(train["comment_text"])


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))


def fast_encode(texts, tokenizer, chunk_size=240, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i : i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


AUTO = tf.data.experimental.AUTOTUNE

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

EPOCHS = 2
BATCH_SIZE = 32 * strategy.num_replicas_in_sync

tokenizer = transformers.DistilBertTokenizer.from_pretrained(
    "distilbert-base-multilingual-cased"
)

save_path = "/content"
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)

fast_tokenizer = BertWordPieceTokenizer("/content/vocab.txt", lowercase=True)

x_train = fast_encode(train.comment_text.astype(str), fast_tokenizer, maxlen=512)
x_valid = fast_encode(valid.comment_text.astype(str).values, fast_tokenizer, maxlen=512)
x_test = fast_encode(test.content.astype(str).values, fast_tokenizer, maxlen=512)

y_valid = valid.toxic.values
y_train = train.toxic.values

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)


def build_vnn_model(transformer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    embed = transformer.weights[0].numpy()
    embedding = Embedding(
        np.shape(embed)[0],
        np.shape(embed)[1],
        input_length=max_len,
        weights=[embed],
        trainable=False,
    )(input_word_ids)

    conc = K.sum(embedding, axis=2)
    conc = Dense(128, activation="relu")(conc)
    conc = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=input_word_ids, outputs=conc)
    model.compile(Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])
    return model


with strategy.scope():
    transformer_layer = transformers.TFDistilBertModel.from_pretrained(
        "distilbert-base-multilingual-cased"
    )
    model_vnn = build_vnn_model(transformer_layer, max_len=512)

model_vnn.summary()


def callbacks():
    calls = []

    reduceLROnPlat = cb.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        verbose=1,
        mode="auto",
        epsilon=0.0001,
        cooldown=1,
        min_lr=0.000001,
    )
    log = cb.CSVLogger("log.csv")
    RocAuc = RocAucEvaluation(validation_data=(x_valid, y_valid), interval=1)
    calls.append(reduceLROnPlat)
    calls.append(log)
    calls.append(RocAuc)
    return calls


N_STEPS = x_train.shape[0] // BATCH_SIZE
calls = callbacks()

train_history = model_vnn.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    callbacks=calls,
    epochs=EPOCHS,
)
