import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import string
import re, random, os
import missingno as msno
import tensorflow as tf
import tensorflow_hub as hub

#%matplotlib inline
INPUT_PATH = ""

train1 = pd.read_csv(os.path.join(INPUT_PATH, "jigsaw-toxic-comment-train.csv"))
train2 = pd.read_csv(os.path.join(INPUT_PATH, "jigsaw-unintended-bias-train.csv"))

train2.toxic = train2.toxic.round().astype(int)

valid = pd.read_csv(os.path.join(INPUT_PATH, "validation.csv"))
test = pd.read_csv(os.path.join("test.csv"))
sub = pd.read_csv(os.path.join("sample_submission.csv"))

train = pd.concat(
    [train1[["comment_text", "toxic"]], train2[["comment_text", "toxic"]]]
)


def plot(train):
    plt.figure(figsize=(15, 8))
    plt.title("0 vs 1")
    plt.xlabel("Toxic")
    plt.ylabel("Count")
    sns.countplot(x="toxic", data=train)


plot(train)

# balancing the data
train = pd.concat(
    [
        train1[["comment_text", "toxic"]],
        train2[["comment_text", "toxic"]].query("toxic==1"),
        train2[["comment_text", "toxic"]]
        .query("toxic==0")
        .sample(n=150000, random_state=0),
    ]
)

plot(train)

plt.figure(figsize=(12, 8))
msno.bar(train)

stopwords = set(STOPWORDS)


def cloud(data, title=None):
    cloud = WordCloud(
        background_color="black",
        stopwords=stopwords,
        max_words=400,
        max_font_size=40,
        scale=3,
    ).generate(str(data))

    fig = plt.figure(figsize=(15, 15))
    plt.axis("off")
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.25)

    plt.imshow(cloud)
    plt.show()


cloud(train["comment_text"], "Train data word cloud")
