from flair.data import Sentence
from flair.models import SequenceTagger

from definitions import ROOT_DIR

en_flair_fged = SequenceTagger.load(ROOT_DIR + "/models/fged/flair_en_fed/best-model.pt")

text = """Paul Norell (born 11 February 1952) is an English actor residing in Auckland, New Zealand. He is known for his portrayal as the King of the Dead in Peter Jackson's The Lord of the Rings: The Return of the King. Some of his other credits include Hercules: The Legendary Journeys playing the traveling food merchant Falafel and Power Rangers: SPD."""
sentence = Sentence(text)

# predict NER tags
en_flair_fged.predict(sentence)

# print sentence with predicted tags
for entity in sentence.get_spans('ner'):
    print(entity)
