import os
import json
import re
import numpy as np
import torch
import random

from flair.data import Sentence
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, FlairEmbeddings, BytePairEmbeddings, TransformerWordEmbeddings

from typing import List

from definitions import ROOT_DIR

def softmax(x):
    return np.exp(x) / sum(np.exp(x))


embedding_types: List[TokenEmbeddings] = [

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),
    # comment in these lines to use flair embeddings

    BytePairEmbeddings(language='en', dim=300, syllables=200000),

    TransformerWordEmbeddings('bert-base-cased', layers='-1', pooling_operation='mean')

]


embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


f1 = open("/media/janzz11/Backup_Drive/rs_3_fetuared_embedded_bert_bpemb_train.jsonl", "w")


def check_overlapping_spans(sample):

    flag = 1

    for i in range(len(sample["entities"])):
        for j in range(len(sample["entities"])):
            if i != j:
                if sample["entities"][i][1] > sample["entities"][j][0] and sample["entities"][i][0] <= sample["entities"][j][0]:
                    flag = 0
                    break

                if sample["entities"][i][0] < sample["entities"][j][1] and sample["entities"][i][1] >= sample["entities"][j][1]:
                    flag = 0
                    break

                if sample["entities"][i][0] >= sample["entities"][j][0] and sample["entities"][i][1] <= sample["entities"][j][1]:
                    flag = 0
                    break
        if flag == 0:
            break
    return flag

tmp = []

c = 0

s = 0

with open(ROOT_DIR + "/datasets/HAnDS/RS_fner_3.json", "r", encoding="utf-8") as file:
    for line in file:

        x = json.loads(line)

        flag = 1

        sample = {'tokens' : x['tokens'], "entities" :[]}

        training_sample = {'encoded_entity': [], "labels": []}

        for link in x["links"]:

            entity = " ".join(x["tokens"][link["start"]:link["end"]])

            if entity == "The" or entity == "It":
                flag = 0
                break

            if entity == "In" or entity == "He" or entity == 'On':
                flag = 0
                break

            if len(link["labels"]) == 1:
                type = link["labels"][0]

            elif len(link["labels"]) == 2:

                if link["labels"][0].split("/")[1] != link["labels"][1].split("/")[1]:
                    flag = 0
            else:
                flag = 0

        if flag == 1 and check_overlapping_spans(sample) == 1 and len(x["links"]) > 0:

            c+=1

            if c%10000 == 0:
                print(c)

            text = " ".join(x["tokens"])
            sentence = Sentence(text, use_tokenizer=False)
            embeddings.embed(sentence)

            for link in x["links"]:

                if link["start"]<10:

                    l_b = 0
                else:
                    l_b = link["start"] - 10

                if link["end"] + 9 <= len(x["tokens"]):

                    r_b = link["end"] + 9
                else:
                    r_b = len(x["tokens"])

                left_context = []

                left_weigths = []

                for i in range(l_b,link["start"]):

                    if x["pos"][i] in ["NN","NNS","NNP","NNPS"]:

                        left_weigths.append(4)
                    else:
                        left_weigths.append(1)

                    left_context.append(sentence[i].embedding.detach().numpy())

                if left_context == []:
                    left_context = np.zeros(1068, dtype=np.float16)

                else:

                    left_weigths = softmax(np.array(left_weigths))

                    context_length = len(left_context)

                    for i in range(0,context_length):

                        left_context[i] = left_weigths[i] * left_context[i]

                    left_context = np.mean(left_context, axis=0, dtype=np.float16)

                entity_embedding = []

                weigths_entity = []

                for i in range(link["start"],link["end"]):

                    if x["pos"][i] in ["NN","NNS","NNP","NNPS"]:

                        weigths_entity.append(4)
                    else:
                        weigths_entity.append(1)

                    entity_embedding.append(sentence[i].embedding.detach().numpy())

                weigths_entity = softmax(np.array(weigths_entity))

                for i in range(0,len(entity_embedding)):

                    entity_embedding[i] = weigths_entity[i] * entity_embedding[i]

                entity_embedding = np.mean(entity_embedding, axis=0, dtype=np.float16)

                rigth_context = []

                rigth_weigths = []

                for i in range(link["end"], r_b):

                    if x["pos"][i] in ["NN","NNS","NNP","NNPS"]:

                        rigth_weigths.append(4)
                    else:
                        rigth_weigths.append(1)

                    rigth_context.append(sentence[i].embedding.detach().numpy())

                tmp_flag = 1

                if rigth_context == [] or (len(rigth_context) == 1 and x["tokens"][-1] == "."):

                    if rigth_context == []:

                        rigth_context = np.zeros(1068,dtype=np.float16)

                        tmp_flag = 0

                    else:
                        flag = random.sample(range(0, 2), 1)

                        if flag[0] == 0:
                            rigth_context = np.zeros(1068, dtype=np.float16)
                            tmp_flag = 0

                if tmp_flag == 1:

                    rigth_weigths = softmax(np.array(rigth_weigths))

                    context_length = len(rigth_context)

                    for i in range(0,context_length):

                        rigth_context[i] = rigth_weigths[i] * rigth_context[i]

                    rigth_context = np.mean(rigth_context, axis=0, dtype=np.float16)

                features_vector = np.concatenate((left_context, entity_embedding, rigth_context)).tolist()

                training_sample['encoded_entity'] = features_vector

                training_sample["labels"] = link["labels"]

                #print(" ".join(x["tokens"][link["start"]:link["end"]]), training_sample["labels"], len(training_sample['encoded_entity']))

                json.dump(training_sample, f1, ensure_ascii=False)
                f1.write("\n")

                s += 1

print("amount of samples:",s)
print()
