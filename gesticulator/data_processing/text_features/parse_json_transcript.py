"""
This file contains script to parse json transcript file and encode it by BERT model
"""
import json
import numpy as np
from argparse import ArgumentParser
from bert_embedding import BertEmbedding
from torchnlp.word_to_vector import FastText
from gesticulator.data_processing.text_features.syllable_count import count_syllables
from dataclasses import dataclass

def encode_json_transcript_with_bert(json_file, bert_model):
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
        embedding:       text embedding model
        printout:        weather we want to do printout for debugging

    Returns:
        all_embedding:   resulting embedding for every time frame with 10 fps

    """
    fillers = ["eh", "ah", "like", "kind of"]
    filler_encoding = bert_model(["eh, ah, like, kind of"])[0][1][0]
    delimiters = ['.', '!', '?']

    silence_encoding = np.array([-15 for i in range(768)]) # BERT has 768-dimensional features
    silent_frame_features = [0, 0, 0, 0, 0]
    
    elapsed_deciseconds = 0   
    output_features = []
    
    # The JSON files contain about a minute long segments
    with open(json_file, 'r') as file:
        transcription_segments = json.load(file)

    # BERT requires the entire sentence instead of singular words
    non_filler_words_in_sentence = [] # Filler words are encoded separately
    frame_word_idx_list = [] # The index of the current word in each frame
    frame_features_list = [] # The corresponding extra features
    
    for segment in transcription_segments: 
        segment_words = segment['alternatives'][0]['words']    

        for word_data in segment_words:                
            word = word_data['word']   

            # Word-level features: duration, speed, start time, end time 
            word_features = extract_word_features(word_data)

            # Get the index of the current word
            if word in fillers:
                # Fillers have word_idx 1
                curr_word_idx = 1 
            elif word[:-1] in fillers: # The last character of the word might be a delimiter
                curr_word_idx = 1

                # Here we explicitly check whether the delimiter signals the end of the sentence
                # For example, commas are not added to the word list
                if word[-1] in delimiters: 
                    non_filler_words_in_sentence.append(word[-1])
            else:
                # Not a filler word
                # The first two indices are reserved for silence and fillers, so increase by 2
                curr_word_idx = len(non_filler_words_in_sentence) + 2
                non_filler_words_in_sentence.append(word)

            # Process the silent frames before the word starts
            while elapsed_deciseconds < word_features['start_time']:
                elapsed_deciseconds += 1
                frame_word_idx_list.append(0) # The idx 0 is reserved for silence
                frame_features_list.append(silent_frame_features)

            # Process the voiced frames           
            while elapsed_deciseconds < word_features['end_time']:
                elapsed_deciseconds += 1
                
                frame_features = extract_voiced_frame_features(
                                    word_features, elapsed_deciseconds)

                frame_word_idx_list.append(curr_word_idx)
                frame_features_list.append(frame_features)

            # If the sentence is over, use bert to embed the words
            is_sentence_over = any([word[-1] == delimiter for delimiter in delimiters]) 

            if is_sentence_over:
                sentence = [' '.join(non_filler_words_in_sentence)] # Concatenate the words using space as a separator

                input_to_bert, encoded_words = bert_model(sentence)[0]

                if input_to_bert[-1] not in delimiters:
                    print("ERROR: missing delimiter in input to BERT!")
                    print("The current sentence:", sentence)
                    print("The input to BERT:", bert_input)
                    exit(-1)

                # Add the silence/filler encodings at the reserved indices
                encoded_words = [silence_encoding] + [filler_encoding] + encoded_words

                # Frame-by-frame features of the entire sentence
                sentence_features = \
                    [ list(encoded_words[word_idx]) + frame_features_list[i]
                      for i, word_idx in enumerate(frame_word_idx_list) ]

                # Add the sentence to the final feature list
                output_features.extend(sentence_features)

                # Reset the sentence-level variables
                non_filler_words_in_sentence = []
                frame_word_idx_list = []
                frame_features_list = []

    if len(output_features) != elapsed_deciseconds-1:
        print(f"ERROR: The number of frames in the encoded transcript ({len(output_features)})") 
        print(f"       does not match the number of frames in the input ({elapsed_deciseconds})!")
        
        exit(-1)

    return np.array(output_features)

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

# -------- Private functions --------

def json_time_to_deciseconds(time_in_text):
    """Convert timestamps from text representation to tenths of a second (e.g. '1.500s' to 15 deciseconds)."""
    # Remove the unit ('s' as seconds) from the representation
    time_in_seconds = float(time_in_text.rstrip('s')) 

    return int(time_in_seconds * 10)

def extract_word_features(word_data):
    start_time = json_time_to_deciseconds(word_data['start_time'])
    end_time   = json_time_to_deciseconds(word_data['end_time'])
    duration   = end_time - start_time
    
    word = word_data['word']

    # Syllables per decisecond
    speed = count_syllables(word) / duration if duration > 0 else 10 # Because the text freq. is 10FPS

    feature_dict = { 'start_time': start_time, 'end_time': end_time,
                     'duration': duration, 'speed': speed }
    
    return feature_dict

def extract_voiced_frame_features(word_features, total_elapsed_time):
    """Return the word encoding and the additional features for the given frame as a list.
  
    Args:
        word_features:       A dictionary with word-level features. See extract_word_features() for details.
        total_elapsed_time:  The elapsed time since the beginning of the entire input sequence
    
    Returns: 
        frame_features:  A list that contains the 5 additional features.
    """
    word_elapsed_time = total_elapsed_time - word_features['start_time']
    # The remaining time is calculated starting from the beginning of the frame - that's why we add 1
    word_remaining_time = word_features['duration'] - word_elapsed_time + 1 
    word_progress = word_elapsed_time / word_features['duration']        
  
    frame_features = [ word_elapsed_time, 
                       word_remaining_time,
                       word_features['duration'], 
                       word_progress, 
                       word_features['speed'] ]

    return frame_features
