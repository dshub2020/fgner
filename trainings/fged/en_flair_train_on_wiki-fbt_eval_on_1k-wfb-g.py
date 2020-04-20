from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, FlairEmbeddings, BytePairEmbeddings
from typing import List
import flair
import numpy as np
import random
import torch
from definitions import ROOT_DIR
from datetime import datetime

#manualSeed = 1

#np.random.seed(manualSeed)
#random.seed(manualSeed)
#torch.manual_seed(manualSeed)

#torch.cuda.manual_seed(manualSeed)
#torch.cuda.manual_seed_all(manualSeed)

#torch.backends.cudnn.deterministic = True

print(flair.device)

data_folder = ROOT_DIR + '/datasets/HAnDS/'

columns = {0: 'text', 1: 'ner'}

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='fed_wiki-fbt_train_downsampled.txt',
                                                   test_file='fed_1k-wfb-g_test.txt',
                                                   dev_file='fed_1k-wfb-g_dev.txt')
print(len(corpus.train))
print(len(corpus.test))
print(len(corpus.dev))
print(corpus.obtain_statistics())

tag_type = 'ner'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

print(tag_dictionary)

embedding_types: List[TokenEmbeddings] = [

    FlairEmbeddings("news-forward-fast"),

    BytePairEmbeddings(language='en', dim=300, syllables=200000),

    FlairEmbeddings("news-backward-fast"),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type, use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer
from datetime import datetime

now = datetime.now()

dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# start training
trainer.train( ROOT_DIR + '/models/fged/en_flair+bpemb_emb_'+ dt_string,
              learning_rate=0.1,
              mini_batch_size=256,
              max_epochs=30,checkpoint=True, patience=3, embeddings_storage_mode='none', num_workers=12, use_amp=True)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves(ROOT_DIR + '/models/fged/en_flair+bpemb_emb_'+ dt_string+'/loss.tsv')
plotter.plot_weights(ROOT_DIR + '/models/fged/en_flair+bpemb_emb_'+ dt_string+'/weights.txt')
