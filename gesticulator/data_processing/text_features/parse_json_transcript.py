"""
This file contains script to parse json transcript file and encode it by BERT model
"""
import json
import numpy as np
from argparse import ArgumentParser

from bert_embedding import BertEmbedding

from gesticulator.data_processing.text_features.syllable_count import sylco


def check_json_transcript(json_file, printout=False):
    """
    Check if everything is fine with the json transcript file

    Args:
        json_file: file with speech transcript
        printout:  weather we need to print for debugging

    Returns:
        nothing, can raise errors
    """


    with open(json_file, 'r') as file:
        datastore = json.load(file)

    prev_time = 0

    for segment in datastore:

        if printout:
            print('New segment')

        words = segment['alternatives'][0]['words']

        for word in words:

            # Get the word timing
            start_time = float(word['start_time'][0:-1])  # remove s
            end_time = float(word['end_time'][0:-1])  # remove s

            print(start_time)
            assert start_time <= end_time
            assert prev_time <= start_time

            prev_time = end_time


            # ToDo: remove that break
            if end_time > 270:
                break

            if printout:
                print(prev_time)

            # Check if we have "'" symbol
            if "'" in word["word"]:
                print(start_time)
                print(word["word"])

    print("Transcript file is alright\n")


def encode_json_transcript(json_file, bert_embedding, printout=False):
    """
    Parse json file and encode every word by BERT

    First, I am separating the text into sentences (because I believe that BERT works the best
     if applied for sentences) Then for each sentence, I collect timing information: which word
      lasted how long, then encode the whole sentence and finally I combine encodings and timings

    Example

    file = {"start_time": "0s", "end_time": "0.500s", "word": "I"},
           {"start_time": "0.5s", "end_time": "0.800s", "word": "love"},
           {"start_time": "0.800s", "end_time": "1s", "word": "you"}
    words = ["I", "love", "you"]
    timing = [1,1,1,1,1, 2,2,2,3,3]

    embed_words = [ [1,2,3] ,[2,3,4] ,[3,4,5] ]
    embed_final = [ [1,2,3] ,[1,2,3] ,[1,2,3] ,[1,2,3] ,[1,2,3],
                    [2,3,4] ,[2,3,4] ,[2,3,4] ,[3,4,5] ,[3,4,5] ]

    Args:
        json_file:       json of transcript of the speech signal by Google ASR
        bert_embedding:  BERT model
        printout:        weather we want to do printout for debugging

    Returns:
        all_embedding:   resulting embedding for every time frame with 10 fps

    """

    with open(json_file, 'r') as file:
        datastore = json.load(file)

    fillers = ["eh", "ah", "like", "kind of"]
    delimiters = ['.', '!', '?']

    # Define special embeddings
    fillers_sentence = "eh, ah, like, kind of".split('\n')
    filler_embedding = bert_embedding(fillers_sentence)[0][1][0]
    silence_embedding = np.array([-15 for i in range(768)]) # encodings seems to be in (-10,10)

    all_the_words = []
    all_embedding = []

    curr_words = []  # will be used to store only one what was said no matter how long it lasted
    curr_ids = []    # will be used to store ids of words for temporal allignment

    word_id = 0

    # Also track additional info about the word and sentence
    t = 0 # in 0.1 ms

    curr_times = []      # for storing word timings: how far we are in the current word
    curr_end_times = []  # for storing word timings: how much is left in the current word
    curr_words_durations = [] # for storing word durations
    curr_words_progress = []  # for storing progress within a word
    curr_words_speed = []           # for storing speed of talking in syllables per minute

    for segment in datastore:

        if printout:
            print("Start a new segment")

        words = segment['alternatives'][0]['words']

        for word in words:

            # Get the word timing
            start_time = float(word['start_time'][0:-1])   # remove s letter
            end_time = float(word['end_time'][0:-1])       # remove s

            # Make it integer in 0.1s
            start_time = int(start_time*10)
            end_time = int(end_time*10)

            # calculate duration
            duration = end_time - start_time

            # Cope with the silence at the beginning
            while t < start_time:
                t += 1 # in 0.1s
                curr_ids.append(0)  # sil
                curr_times.append(0)
                curr_words_durations.append(0)
                curr_end_times.append(0)
                curr_words_progress.append(0)
                curr_words_speed.append(0)

            # Check if that's not a filler
            if word['word'] not in fillers and word['word'][:-1] not in fillers:
                # normal case
                word_id = word_id + 1
                curr_words.append(word['word'])

                # store current word ids
                while t < end_time:
                    t += 1
                    curr_ids.append(word_id)

            else:
                # if the filler contains delimiter - we want to keep the delimiter
                if word['word'][-1] in delimiters:
                        curr_words.append(word['word'][-1])

                # set separate id for the filler words
                while t < end_time:
                    t += 1
                    curr_ids.append(-1)

            # put the timer back to encode additional features
            t = start_time
            word_time= 0

            # Calculate speed of the speech
            numb_syl = sylco(word['word'])
            if duration>0:
                syl_per_second = numb_syl / duration
            else:
                syl_per_second = 10 # since the frequency is 10 fsp

            # Encode additional word features
            while t < end_time:
                curr_end_times.append(duration - word_time)
                t += 1
                word_time += 1
                word_progress = word_time / duration if duration>0 else 1
                curr_times.append(word_time)
                curr_words_durations.append(duration)
                curr_words_progress.append(word_progress)
                curr_words_speed.append(syl_per_second)

            if word['word'][-1] in delimiters:
                # sentence is finished
                curr_sentence = ' '.join(word for word in curr_words) #[:-1] # remove the dot

                sentence = curr_sentence.split('\n')
                embed = bert_embedding(sentence)[0]

                # check if BERT can encode the whole sentence
                if embed[0][-1] not in delimiters:
                    print("ERROR! Embedding has too short context length."
                          "It didn't encode the whole sentence!")
                    print("Sentence: ", sentence)
                    print("Embedded: ", embed[0])
                    exit(1)

                embedding = embed[1]

                # Add silence and filler embeddings
                embedding = [silence_embedding] + embedding + [filler_embedding]

                # Combine embedding with the additional info about the words
                curr_embed = [list(embedding[ind]) + [curr_times[j]] + [curr_end_times[j]] +
                                                   [curr_words_durations[j]] + [curr_words_progress[j]] +
                                                   [curr_words_speed[j]] for j,ind in  enumerate(curr_ids)]

                all_embedding = all_embedding + curr_embed

                # Add it to the global set
                all_the_words = all_the_words + curr_words

                # restart the current sentence
                curr_words = []
                curr_ids = []

                curr_times = []
                curr_end_times = []
                curr_words_durations = []
                curr_words_progress = []
                curr_words_speed = []

                word_id = 0

        if printout:
            print(t)
            print(len(all_embedding))
            print(embed[0])

    # check if Some sentence end delimiter was not transcribed
    if t != len(all_embedding):
        print("ERROR! Encoding is too short! Possibly because some delimiters were missing")
        #print(embed[0])
        print(t)
        print(len(all_embedding))
        exit(1)

    recording = ' '.join(word for word in all_the_words)

    if printout:
        print(recording)

    return np.array(all_embedding)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", default="/home/taras/Documents/Datasets/SpeechToMotion/Irish/processed/WithTextV2/test/inputs/NaturalTalking_04.json",
        help="Filename to parse", metavar="FILE")

    args = parser.parse_args()

    bert_embedding = BertEmbedding(max_seq_length=100, model='bert_12_768_12',
                                   dataset_name='book_corpus_wiki_en_cased')

    json_file = args.filename  # 'NaturalTalking_004.json'

    check_json_transcript(json_file)

    encoding = encode_json_transcript(json_file, bert_embedding, printout=False)

    print("Done")
