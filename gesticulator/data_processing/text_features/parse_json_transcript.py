"""
This file contains the functions that encode the words in the JSON transcriptions using BERT or FastText.

The main differences between the two embeddings are:
    - the dimensionalities (768 vs 300)
    - BERT encodes the words using the entire sentence as a context, 
      while FastText encodes the words individually

We expect that the JSONs were create with 10 FPS. 

For each frame, we calculate 5 extra features in addition to the word encodings. See extract_extra_features() for details.
"""
import json
import numpy as np
from gesticulator.data_processing.text_features.syllable_count import count_syllables

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
        json_file:        json of transcript of the speech signal by Google ASR
        embedding:        text embedding model

    Returns:
        feature_array:  an array of shape (n_frames, 773), where n_frames is the number of timeframes.
    
        NOTE: The transcription is processed at 10 FPS, so for a 60 second input 
              there will be 600 frames. 
        
        NOTE: The feature dimensionality is 773 because we add 5 extra features
              on top of the BERT dimensionality (768).
    """
    fillers = ["eh", "ah", "like", "kind of"]
    filler_encoding = bert_model(["eh, ah, like, kind of"])[0][1][0]
    delimiters = ['.', '!', '?']

    silence_encoding = np.array([-15 for i in range(768)]) # BERT has 768-dimensional features
    silent_frame_features = [0, 0, 0, 0, 0]
    
    elapsed_deciseconds = 0   
    feature_array = []
    
    # BERT requires the entire sentence instead of singular words

    # NOTE: The index 0 is reserved for silence, and the index 1 is reserved for filler words
    non_filler_words_in_sentence = [] 
    sentence_word_indices_list = [] # The index of the current word in the above vector for each frame
    sentence_extra_features_list = [] # The corresponding extra features
    
    with open(json_file, 'r') as file:
        transcription_segments = json.load(file)

    # The JSON files contain about a minute long segments
    for segment in transcription_segments: 
        segment_words = segment['alternatives'][0]['words']    

        for word_data in segment_words:                
            word = word_data['word']   

            # Word attributes: duration, speed, start time, end time 
            word_attributes = extract_word_attributes(word_data)

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
                # -> not a filler word

                # The first two indices are reserved for silence and fillers, 
                # therefore we start indexing from 2
                curr_word_idx = len(non_filler_words_in_sentence) + 2
                non_filler_words_in_sentence.append(word)

            # Process the silent frames before the word starts
            while elapsed_deciseconds < word_attributes['start_time']:
                elapsed_deciseconds += 1
                sentence_word_indices_list.append(0) # The idx 0 is reserved for silence
                sentence_extra_features_list.append(silent_frame_features)

            # Process the voiced frames           
            while elapsed_deciseconds < word_attributes['end_time']:
                elapsed_deciseconds += 1
                
                frame_features = extract_extra_features(
                                    word_attributes, elapsed_deciseconds)

                sentence_word_indices_list.append(curr_word_idx)
                sentence_extra_features_list.append(frame_features)

            # If the sentence is over, use bert to embed the words
            is_sentence_over = any([word[-1] == delimiter for delimiter in delimiters]) 

            if is_sentence_over:
                # Concatenate the words using space as a separator
                sentence = [' '.join(non_filler_words_in_sentence)]

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
                    [ list(encoded_words[word_idx]) + sentence_extra_features_list[i]
                      for i, word_idx in enumerate(sentence_word_indices_list) ]

                # Add the sentence to the final feature list
                feature_array.extend(sentence_features)

                # Reset the sentence-level variables
                non_filler_words_in_sentence = []
                sentence_word_indices_list = []
                sentence_extra_features_list = []

    if len(feature_array) != elapsed_deciseconds:
        print(f"ERROR: The number of frames in the encoded transcript ({len(feature_array)})") 
        print(f"       does not match the number of frames in the input ({elapsed_deciseconds})!")
        
        exit(-1)

    return np.array(feature_array)

def encode_json_transcript_with_fasttext(json_file, fasttext_model):
    """Create a feature array by encoding the given JSON file with the given FastText model
    and adding the 5 extra features for each frame.

    Args:
        json_file:       the path to the JSON transcription
        fasttext_model:  the FastText model to encode the words with
    
    Returns:
        feature_array:   an array of shape (n_frames, 305), where n_frames is the number of timeframes.
    
    NOTE: The transcription is processed at 10 FPS, so for a 60 second input 
          there will be 600 frames. 

    NOTE: The feature dimensionality is 305 because we add 5 extra features
          on top of the FastText dimensionality (300).
    """
    fillers = ["eh", "ah", "like", "kind of"]
    delimiters = ['.', '!', '?']
    
    silence_encoding = np.array([-15 for i in range(300)]) # Fasttext has 300-dimensional features
    silence_extra_features = [0, 0, 0, 0, 0] # The extra text features for silence are all zeros
   
    filler_encoding  = fasttext_model["ah"] # The fillers will be encoded with the same vector

    elapsed_deciseconds = 0   
    feature_array = []
    
    # The JSON files contain about a minute long segments
    with open(json_file, 'r') as file:
        transcription_segments = json.load(file)
    
    for segment in transcription_segments: 
        segment_words = segment['alternatives'][0]['words']    

        for word_data in segment_words:   
            word = word_data['word']

            # The delimiters are not relevant to FastText
            for d in delimiters: 
                word.replace(d, '') 

            # Word attributes: duration, speed, start time, end time 
            word_attributes = extract_word_attributes(word_data)

            # NOTE: Each frame below is one decisecond long
            
            # Process the silent frames before the word starts
            while elapsed_deciseconds < word_attributes['start_time']:
                elapsed_deciseconds += 1
                # The silent frames always have the same features and encoding
                feature_array.append(list(silence_encoding) + silence_extra_features)
            
            word_encoding = filler_encoding if word in fillers else fasttext_model[word]

            # Process the voiced frames             
            while elapsed_deciseconds < word_attributes['end_time']:
                elapsed_deciseconds += 1

                frame_features = extract_extra_features(
                                    word_attributes, elapsed_deciseconds)

                feature_array.append(list(word_encoding) + frame_features)

    if len(feature_array) != elapsed_deciseconds:
        print(f"ERROR: The number of frames in the encoded transcript ({len(feature_array)})") 
        print(f"       does not match the number of frames in the input ({elapsed_deciseconds})!")
        
        exit(-1)

    return np.array(feature_array)

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
    # Remove the unit ('s' as in seconds) from the representation
    time_in_seconds = float(time_in_text.rstrip('s')) 

    return int(time_in_seconds * 10)

def extract_word_attributes(word_data):
    start_time = json_time_to_deciseconds(word_data['start_time'])
    end_time   = json_time_to_deciseconds(word_data['end_time'])
    duration   = end_time - start_time
    
    word = word_data['word']

    # Syllables per decisecond
    speed = count_syllables(word) / duration if duration > 0 else 10 # Because the text freq. is 10FPS

    feature_dict = { 'start_time': start_time, 'end_time': end_time,
                     'duration': duration, 'speed': speed }
    
    return feature_dict

def extract_extra_features(word_attributes, total_elapsed_time):
    """Return the word encoding and the additional features for the current frame as a list.
    The five additional features are: 
        
        1) elapsed time since the beginning of the current word 
        2) remaining time from the current word
        3) the duration of the current word
        4) the progress as the ratio 'elapsed_time / duration'
        5) the pronunciation speed of the current word (number of syllables per decisecond)

    Args:
        word_attributes:     A dictionary with word-level attributes. See extract_word_attributes() for details.
        total_elapsed_time:  The elapsed time since the beginning of the entire input sequence
    
    Returns: 
        frame_extra_features:  A list that contains the 5 additional features.
    """
    word_elapsed_time = total_elapsed_time - word_attributes['start_time']
    # The remaining time is calculated starting from the beginning of the frame - that's why we add 1
    word_remaining_time = word_attributes['duration'] - word_elapsed_time + 1 
    word_progress = word_elapsed_time / word_attributes['duration']        
  
    frame_extra_features = [ word_elapsed_time, 
                             word_remaining_time,
                             word_attributes['duration'], 
                             word_progress, 
                             word_attributes['speed'] ]

    return frame_extra_features
