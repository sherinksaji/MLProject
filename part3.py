#!/usr/bin/env python
# coding: utf-8

# # Part 3

# In[14]:


# Import necessary packages
import pandas as pd
import re
import numpy as np
import math
import copy

def load_dev_in_data(file_path):
    """Specific function to load dev.in data, which only contains words."""
    
    # Open the file for reading using utf-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the content of the file, remove leading/trailing whitespace,
        # and split the content into separate sentences based on empty lines.
        data = file.read().strip().split('\n\n')
    
    # Process each sentence and split it into words while removing any empty lines
    # to create a list of sentences, where each sentence is represented as a list of words.
    return [[word for word in sentence.split('\n') if word.strip()] for sentence in data]


def load_data_modified_v7(file_path):
    
    # Open the file for reading using utf-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the content of the file, remove leading/trailing whitespace,
        # and split the content into separate sentences based on empty lines.
        data = file.read().strip().split('\n\n')
    
    # Initialize an empty list to store the processed data
    processed_data = []
    
    # Iterate over each sentence in the data
    for sentence in data:
        
        # Initialize an empty list to store the processed words and tags for the current sentence
        processed_sentence = []
        
        # Iterate over each line in the sentence
        for line in sentence.split('\n'):
            
            # Check if line is not empty
            if line.strip():  
                
                # Use regular expression to match a word and a tag at the end of the line
                match = re.search(r'^(.*)\s(\S+)$', line)
                
                # If match is true
                if match:
                    # Extract the word and tag using groups() method of the match object
                    word, tag = match.groups()
                    
                    # Combine the word and tag with a space and append to the processed_sentence list
                    processed_sentence.append(f"{word} {tag}")
        
        # Add the processed sentence to the data only if it's not empty
        if processed_sentence:
            processed_data.append(processed_sentence)
    
    return processed_data


def compute_probabilities_v2(data, state_list):
    
    # Create a dictionary to store the count of starting transitions, count of transitions from one state to other
    # coutn of emissions for each state and count of occurences for each state and count of occurences for each state
    start_transition_count = {state: 0 for state in state_list}
    transition_count = {state: {state2: 0 for state2 in state_list} for state in state_list}
    emission_count = {state: {} for state in state_list}
    state_count = {state: 0 for state in state_list}
    
    # Iterate over each sentence in the provided data
    for sentence in data:
        # Initialize a variable to keep track of the previous state in the sentence
        prev_state = None
        # Iterate over each line (word and tag pair) in the sentence
        for line in sentence:
            # Use regular expression to extract word and state from the line
            match = re.search(r'^(.*)\s(\S+)$', line.strip())
            # Check if the regular expression match was successful
            if match:
                # Extract word and state using groups() method of the match object
                word, state = match.groups()
                # Check if this is the beginning of a sentence (no previous state)
                if prev_state is None:
                    # Increment the count of starting transitions for the current state
                    start_transition_count[state] += 1
                else:
                    # Increment the count of transitions from the previous state to the current state
                    transition_count[prev_state][state] += 1
                    # Increment the emission count for the previous state and the current word
                    # If the word has not been encountered before for the current state, initialize the count to 0
                    emission_count[prev_state][word] = emission_count[prev_state].get(word, 0) + 1
                # Increment the count of occurrences for the current state
                state_count[state] += 1
                # Update the previous state for the next iteration
                prev_state = state
        # After processing all lines in the sentence, check if there is a previous state
        if prev_state:
            # Increment the emission count for the last state and word
            # If the word has not been encountered before for the last state, initialize the count to 0
            emission_count[prev_state][word] = emission_count[prev_state].get(word, 0) + 1
            
    # Calculate the total number of sentences in the data
    total_sentences = len(data)
    # Calculate the probabilities for start transitions based on the counts
    start_transition_prob = {state: count / total_sentences for state, count in start_transition_count.items()}
    # Calculate the probabilities for transitions based on the counts
    transition_prob = {state: {state2: count2 / state_count[state] for state2, count2 in count.items()} for state, count in transition_count.items()}
    # Calculate the probabilities for emissions (words) based on the counts
    emission_prob = {state: {word: count / state_count[state] for word, count in state_emission_count.items()} for state, state_emission_count in emission_count.items()}
    # Return the calculated probabilities for start transitions, transitions, and emissions
    return start_transition_prob, transition_prob, emission_prob

def extract_entities_from_tags(tags):
    
    # Initialize a list to store extracted entities
    entities = []
    
    # Initialize a list to store the current entity being processed
    entity = []
    
    # Iterate over each tag in the provided list of tags
    for tag in tags:
        # Checking for "B-" tag
        if tag.startswith("B-"):
            if entity:
                # If there was an ongoing entity, append it to the entities list
                entities.append(tuple(entity))
                # Clear the entity list for the new entity
                entity = []
            
            # Add the tag to the current entity
            entity.append(tag)
            
        # Checking entity starts with I- tag
        elif tag.startswith("I-"):
            # Add the tag to the current entity
            entity.append(tag)
        
        
        else:
            if entity:
                # If there was an ongoing entity, append it to the entities list
                entities.append(tuple(entity))
                # Clear the entity list
                entity = []
    
    # Check if there is any remaining entity after processing all tags
    if entity:
        # Append the last entity to the entities list
        entities.append(tuple(entity))
    
    # Convert the list of entities into a set to remove duplicates
    return set(entities)


# In[15]:


def process_dataset_final_v5(dataset_type):
    
    # Adjusting the paths dynamically based on dataset type
    train_path = f"Data/{dataset_type}/train"
    dev_in_path = f"Data/{dataset_type}/dev.in"
    dev_out_path = f"Data/{dataset_type}/dev.out"
    
    # Load training data using the modified version 7 of the data loading function
    train_data = load_data_modified_v7(train_path)
    
    # Load dev.in data using the specialized data loading function
    dev_in_data = load_dev_in_data(dev_in_path)
    
    # Load dev.out data (actual tags) from the corresponding file
    with open(dev_out_path, 'r', encoding='utf-8') as f:
        # Read the entire content of the file, remove leading/trailing whitespace,
        # and split the content into separate sentences based on double newline characters.
        dev_tags_actual = [sentence.split() for sentence in f.read().strip().split('\n\n')]
    
    # Initialize state and observations dictionary
    states = {}
    observations = {}

    # Iterate over each sentence in the training data
    for sentence in train_data:
        
        # Iterate over each line (word and tag pair) in the sentence
        for line in sentence:
            
            # Use regular expression to extract word and tag from the line
            match = re.search(r'^(.*)\s(\S+)$', line.strip())
            
            if match:
                # Extract word and tag using groups() method of the match object
                word, tag = match.groups()
                
                # Increment the count of occurrences for the current tag in the 'states' dictionary
                # If the tag has not been encountered before, initialize the count to 0
                states[tag] = states.get(tag, 0) + 1
                
                # Check if the tag is not already in the 'observations' dictionary
                if tag not in observations:
                    # Initialize a sub-dictionary for the tag
                    observations[tag] = {}
                    
                # Increment the count of occurrences for the current word under the current tag
                # If the word has not been encountered before for the current tag, initialize the count to 0
                observations[tag][word] = observations[tag].get(word, 0) + 1

    # Create a list of states by extracting keys from the 'states' dictionary
    state_list = list(states.keys())
    
    # Compute probabilities for the Hidden Markov Model (HMM) using the training data and state list
    # The resulting probabilities include start transition, transition, and emission probabilities
    start_transition_prob, transition_prob, emission_prob = compute_probabilities_v2(train_data, state_list)

    return (start_transition_prob, transition_prob)


# Process the "ES" dataset and retrieve start probabilities and transition probabilities
es_tuple = process_dataset_final_v5("ES")
sr_start_probabilities_es = es_tuple[0]
df_transition_es = es_tuple[1]

# Process the "RU" dataset and retrieve start probabilities and transition probabilities
ru_tuple = process_dataset_final_v5("RU")
sr_start_probabilities_ru = ru_tuple[0]
df_transition_ru = ru_tuple[1]


# Convert start probabilities for the "ES" dataset into a Pandas Series
sr_start_probabilities_es = pd.Series(sr_start_probabilities_es)

# Convert transition probabilities for the "ES" dataset into a transposed Pandas DataFrame
df_transition_es = pd.DataFrame(df_transition_es).T

# Read emission probabilities for the "ES" dataset from a CSV file
df_emission_es = pd.read_csv('Data/ES/csv_dev_in_es_test_e_x_y.csv')

# Drop duplicates based on columns 'x' and 'y', keeping the last occurrence
df_emission_es = df_emission_es.drop_duplicates(subset=['x','y'], keep='last')

# Pivot the emission DataFrame to create a matrix with 'y' as index and 'x' as columns
# Fill any missing values with 0
df_emission_es = df_emission_es.pivot(index='y', columns='x', values='e(x|y)').fillna(0)

# Display the emission DataFrame for the "ES" dataset
# display(df_emission_es)

# Convert start probabilities for the "RU" dataset into a Pandas Series
sr_start_probabilities_ru = pd.Series(sr_start_probabilities_ru)

# Convert transition probabilities for the "RU" dataset into a transposed Pandas DataFrame
df_transition_ru = pd.DataFrame(df_transition_ru).T

# Read emission probabilities for the "RU" dataset from a CSV file
df_emission_ru = pd.read_csv('Data/RU/csv_dev_in_ru_test_e_x_y.csv')

# Drop duplicates based on columns 'x' and 'y', keeping the last occurrence
df_emission_ru = df_emission_ru.drop_duplicates(subset=['x','y'], keep='last')

# Pivot the emission DataFrame to create a matrix with 'y' as index and 'x' as columns
# Fill any missing values with 0
df_emission_ru = df_emission_ru.pivot(index='y', columns='x', values='e(x|y)').fillna(0)

# Display the emission DataFrame for the "RU" dataset
# display(df_emission_ru)


# In[ ]:


def modified_viterbi(observation, transition, emission, start_probabilities, kth_best):
    k_best = math.ceil(kth_best/3)
    states = transition.index.tolist()
    step_count = len(observation)

    preceding = {}

    preceding = {state: [(0, [])] * k_best for state in states}

    for state, sequences in preceding.items():
        
        try:
            preceding[state][0] = (emission[observation[0]][state] * start_probabilities[state], [state])
        except:
            preceding[state][0] = (emission["#UNK#"][state] * start_probabilities[state], [state])

    for step in range(1, step_count):

        # refresh current at the beginning of each step
        current = {state: [(0, [])] * k_best for state in states}

        for current_state in states:

            for previous_state in states:

                for sequence in preceding[previous_state]:

                    # getting the transition probability from preceding state to current state from one of the tuples in the "preceding" table
                    prev_probability = sequence[0]

                    try:
                        emission_param = emission.loc[current_state, observation[step]]
                    except:
                        emission_param = emission.loc[current_state, "#UNK#"]
                    prev_to_cur_probability = prev_probability * transition.loc[previous_state, current_state] * emission_param

                    # sort the tuples in ascending order so that the first tuple has lowest probability
                    current[current_state] = sorted(current[current_state], key=lambda x: x[0])
                    
                    if prev_to_cur_probability >= current[current_state][0][0]:
                        sequence_list = copy.deepcopy(sequence[1])
                        sequence_list.append(current_state)
                        current[current_state][0] = (prev_to_cur_probability, sequence_list)
        
        # we are either entering the next step or leaving the loop, so preceding becomes current
        preceding = copy.deepcopy(current)

    combined_list = []
    for sequences in preceding.values():
        combined_list.extend(sequences)

    combined_list = sorted(combined_list, key=lambda x: x[0])[::-1]

    return combined_list[kth_best-1]


# In[ ]:

print("Creating files. Please wait. This might take a while ...")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip().split('\n\n')
    return [sentence.split('\n') for sentence in data]

es_observations = load_data('Data/ES/dev.in')
ru_observations = load_data('Data/RU/dev.in')

predicted_tags_viterbi_ES2 = [modified_viterbi([word.split()[0] for word in sentence], df_transition_es, df_emission_es, sr_start_probabilities_es, 2)[1] for sentence in es_observations]
predicted_tags_viterbi_ES8 = [modified_viterbi([word.split()[0] for word in sentence], df_transition_es, df_emission_es, sr_start_probabilities_es, 8)[1] for sentence in es_observations]
predicted_tags_viterbi_RU2 = [modified_viterbi([word.split()[0] for word in sentence], df_transition_ru, df_emission_ru, sr_start_probabilities_ru, 2)[1] for sentence in ru_observations]
predicted_tags_viterbi_RU8 = [modified_viterbi([word.split()[0] for word in sentence], df_transition_ru, df_emission_ru, sr_start_probabilities_ru, 8)[1] for sentence in ru_observations]

predicted_ES2 = []
for s in range(len(predicted_tags_viterbi_ES2)):
    for i in range(len(predicted_tags_viterbi_ES2[s])):
        predicted_ES2.append(es_observations[s][i] + " "+ predicted_tags_viterbi_ES2[s][i])
    predicted_ES2.append('\n')

predicted_ES8 = []
for s in range(len(predicted_tags_viterbi_ES8)):
    for i in range(len(predicted_tags_viterbi_ES8[s])):
        predicted_ES8.append(es_observations[s][i] + " "+ predicted_tags_viterbi_ES8[s][i])
    predicted_ES8.append('\n')

predicted_RU2 = []
for s in range(len(predicted_tags_viterbi_RU2)):
    for i in range(len(predicted_tags_viterbi_RU2[s])):
        predicted_RU2.append(ru_observations[s][i] + " "+ predicted_tags_viterbi_RU2[s][i])
    predicted_RU2.append('\n')

predicted_RU8 = []
for s in range(len(predicted_tags_viterbi_RU8)):
    for i in range(len(predicted_tags_viterbi_RU8[s])):
        predicted_RU8.append(ru_observations[s][i] + " "+ predicted_tags_viterbi_RU8[s][i])
    predicted_RU8.append('\n')

with open('Data/ES/dev.p3.2nd.out', 'w', encoding='utf-8') as file:
    # Write each modified line back to the output file
    for line in predicted_ES2:
        if line != "\n":
            file.write(line + '\n')
        else:
            file.write(line)

with open('Data/ES/dev.p3.8th.out', 'w', encoding='utf-8') as file:
    # Write each modified line back to the output file
    for line in predicted_ES8:
        if line != "\n":
            file.write(line + '\n')
        else:
            file.write(line)

with open('Data/RU/dev.p3.2nd.out', 'w', encoding='utf-8') as file:
    # Write each modified line back to the output file
    for line in predicted_RU2:
        if line != "\n":
            file.write(line + '\n')
        else:
            file.write(line)

with open('Data/RU/dev.p3.8th.out', 'w', encoding='utf-8') as file:
    # Write each modified line back to the output file
    for line in predicted_RU8:
        if line != "\n":
            file.write(line + '\n')
        else:
            file.write(line)

print("Files created successfully")
