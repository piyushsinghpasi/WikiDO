from flair.data import Sentence
from flair.models import SequenceTagger
import time
# load tagger
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

tag_mappings = {
    'CARDINAL': 'CARDINAL',
    'DATE': 'DATE',
    'EVENT': 'EVENT',
    'FAC': 'BUILDING',
    'GPE': 'GEO-POLITICAL ENTITY',
    'LANGUAGE': 'LANGUAGE',
    'LAW': 'LAW',
    'LOC': 'LOCATION',
    'MONEY': 'MONEY',
    'NORP': 'AFFILIATION',
    'ORDINAL': 'ORDINAL',
    'ORG': 'ORGANIZATION',
    'PERCENT': 'PERCENT',
    'PERSON': 'PERSON',
    'PRODUCT': 'PRODUCT',
    'QUANTITY': 'QUANTITY',
    'TIME': 'TIME',
    'WORK_OF_ART': 'WORK OF ART',
}

texts =  ['the holbrook house facade features a row of windows with red shutters', 'a red and white building with a red roof and white trim', 'a topographic map of berryville, virginia from 1944 shows a road crossing snickers gap on blue ridge mountain', 'a yellow and black blackburn skua plane with the number 307 on the tail', 'a mountain peak rises above the porchabella glacier', 'a man with a beard and a black shirt is holding a guitar he is smiling and has a star wars logo on his shirt', 'a painting of a large building with a tree in front of it the building has a coat of arms on it, which is the ferrers family from derby', 'a street sign on the side of the road that says south 27', 'a man running with a baton during a relay race', 'the bower house is a large red brick building with a circular driveway', 'a view of a river with a lock and dam 7, and a highway in the background', 'the courtyard of the basilica is a large open area with a statue in the center the statue is surrounded by a potted plant and flowers, adding a touch of', 'the dublin guitar quartet poses for a photo with philip glass', 'a gondola lift at a winery, carrying passengers over a valley', 'a woman in a blue dress and a red headband is standing in front of a museum', 'two large orange francis turbine rotors from the söse dam power station', 'a portrait of george berkeley, an irish philosopher and church of ireland bishop, wearing a black hat and a white collar', 'a wooden support structure in a corner of the horyuji temple', 'the national m k ciurlionis school of art is a large building with a prominent clock tower the building is made of stone and has a large staircase leading up', 'a man in a yellow robe, possibly a priest, is performing a ritual he is holding a book and two blue cloths, possibly garments, while standing in front of a', 'two female wrestlers, one holding a belt, celebrating their victory in a wrestling match', 'a view of batu caves, outside of the temple complex, with a mountain in the background', 'two brown signs on a pole designating the california trail and pony express trail', 'a map of launceston town and castle from 1611 by john speed', 'a group of bats hanging from a tree in a park']

start_time = time.time()
for text in texts:
    # make example sentence
    # text = "On September 1st George won 1 dollar while watching Game of Thrones."
    sentence = Sentence(text)

    # predict NER tags
    tagger.predict(sentence)


    #  |  end_position
    #  |  
    #  |  start_position
    #  |  
    #  |  text unlabeled_identifier get_label labels tag score


    last_index = 0
    replaced_sentence = ""

    for entity in sentence.get_spans('ner'):

        if entity.tag in tag_mappings:
            replaced_sentence += text[last_index:entity.start_position] + tag_mappings[entity.tag] + " " + entity.text
            last_index = entity.end_position

    replaced_sentence += text[last_index:]

    print(text)  
    print("-"*100)
    print(replaced_sentence)
    print("="*100)

print(f"Ends in {time.time()-start_time:.3f} s")