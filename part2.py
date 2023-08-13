#!/usr/bin/env python
# coding: utf-8

# # Part 2

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
# print(df_emission_es)

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

# Display transition parameters - ES Dataset
print("Display transition parameters - ES Dataset")
print(df_transition_es)

# Display transition parameters - RU Dataset
print("Display transition parameters - RU Dataset")
print(df_transition_ru)

# Display starting probabilities - ES Dataset
print("Display starting probabilities - ES Dataset")
print(sr_start_probabilities_es)

# Display starting probabilities - RU Dataset
print("Display starting probabilities - RU Dataset")
print(sr_start_probabilities_ru)

def viterbi(observation, transition, emission, start_probabilities):
    
    # Extract states from the transition DataFrame
    states = transition.index.tolist()
    state_count = transition.shape[0]
    step_count = len(observation)
    
    # Create a DataFrame to memoize the incoming edges for each node in the Viterbi trellis diagram
    df_incoming = np.zeros((state_count , step_count), dtype=int)
    df_incoming = pd.DataFrame(df_incoming, index = states)
    
    # Create a series to memoize the overall probabilities of arriving at each state at a particular step
    # Calculate the step probabilities using emission and start probabilities
    try:
        step_probabilities = emission[observation[0]] * start_probabilities
    except:
        step_probabilities = emission["#UNK#"] * start_probabilities
    step_probabilities_temp = pd.Series({state: 0 for state in states})
    
    # Iterate over each step in the observation sequence (excluding the first step)
    for step in range(1, step_count):
        # Iterate over each current state
        for current_state in states:
            # Create a Series to track probabilities of choosing the maximum path from previous states to the current state
            choosing_max = pd.Series({state: 0 for state in states})
            # Iterate over each previous state
            for previous_state in states:
                try:
                    # Calculate the probability of choosing the maximum path using Viterbi algorithm formula
                    # Multiply step probability, transition probability, and emission probability
                    choosing_max.loc[previous_state] = step_probabilities.loc[previous_state] * transition.loc[previous_state, current_state] * emission.loc[current_state, observation[step]]
                except:
                    # If emission probability for the current observation is not available, use emission for "#UNK#"
                    choosing_max.loc[previous_state] = step_probabilities.loc[previous_state] * transition.loc[previous_state, current_state] * emission.loc[current_state, "#UNK#"]
            # Find the state with the maximum calculated probability
            max_state = choosing_max.idxmax()
            # Update the step_probabilities_temp Series with the maximum probability for the current state
            step_probabilities_temp.loc[current_state] = choosing_max.loc[max_state]
            # Memoize the incoming state that leads to the maximum probability in the df_incoming DataFrame
            df_incoming.loc[current_state, step] = max_state
        # Update the step_probabilities Series for the next step using the updated probabilities from step_probabilities_temp
        step_probabilities = step_probabilities_temp.copy()
    
    # get sequences in descending order of likelihood
    descending_final_probability_states = step_probabilities.sort_values(ascending=False).index.to_numpy()    
    # Create a DataFrame to store sequences of states
    df_sequences = np.zeros((state_count , step_count), dtype=int)
    df_sequences = pd.DataFrame(df_sequences)
    # Initialize the last step of each sequence with the most likely states
    for sequence_rank in range(state_count):
        df_sequences.loc[sequence_rank, step_count-1] = descending_final_probability_states[sequence_rank]
    # Iterate through steps and fill in the rest of the sequences using memoized incoming edges
    for sequence_rank in range(state_count):
        for step in range(step_count-2, -1, -1):
            # Get the previous state of the current step from the memoized incoming edges
            previous_state = df_sequences.loc[sequence_rank, step+1]
            # Assign the previous state to the current step in the sequence
            df_sequences.loc[sequence_rank, step] = df_incoming.loc[previous_state, step+1]
    # Return the sequence of states with the highest likelihood (most likely sequence)
    return df_sequences.loc[0].values


def load_data(file_path):
    # Open the file for reading using UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire content of the file, remove leading/trailing whitespace,
        # and split the content into separate sentences based on double newline characters.
        data = file.read().strip().split('\n\n')
    
    # Split each sentence into lines using newline characters
    # The result is a list of lists, where each inner list represents a sentence
    # and contains lines of text from the file.
    return [sentence.split('\n') for sentence in data]

# Load observations (words) for the "ES" dataset from the 'dev.in' file
es_observations = load_data('Data/ES/dev.in')

# Apply the Viterbi algorithm to predict the most likely sequence of tags (hidden states)
# for each sentence in the "ES" dataset using the precomputed transition and emission probabilities,
# as well as the start probabilities for the Hidden Markov Model (HMM)

# Display Viterbi Output calcualation - ES Dataset
print("Calculating Viterbi Output - ES Dataset")

# Display Viterbi Output calcualation - RU Dataset
print("Calculating Viterbi Output - RU Dataset")


predicted_tags_viterbi_ES1 = [viterbi([word.split()[0] for word in sentence], df_transition_es, df_emission_es, sr_start_probabilities_es) for sentence in es_observations]

# Create an empty list to store the combined predictions for the "ES" dataset
predicted_ES1 = []

# Iterate through each sentence in the predicted Viterbi tags for the "ES" dataset
for s in range(len(predicted_tags_viterbi_ES1)):
    
    # Iterate through each word's index in the current sentence
    for i in range(len(predicted_tags_viterbi_ES1[s])):
        
        # Combine the observed word with the predicted tag from Viterbi and append it to predicted_ES1 list
        predicted_ES1.append(es_observations[s][i] + " "+ predicted_tags_viterbi_ES1[s][i])
    
    # Append a newline character to separate predictions for different sentences
    predicted_ES1.append('\n')


# Open the 'dev.p2.out' file for writing using UTF-8 encoding
with open('Data/ES/dev.p2.out', 'w', encoding='utf-8') as file:
    # Write each modified line back to the output file
    for line in predicted_ES1:
        if line != "\n":
            # Write the line followed by a newline character
            file.write(line + '\n')
        else:
            # Write the newline character without an additional newline
            file.write(line)
            
            es_observations = load_data('Data/ES/dev.in')
            
# Load observations (words) for the "ES" dataset from the 'dev.in' file
ru_observations = load_data('Data/RU/dev.in')

# Apply the Viterbi algorithm to predict the most likely sequence of tags (hidden states)
# for each sentence in the "RU" dataset using the precomputed transition and emission probabilities,
# as well as the start probabilities for the Hidden Markov Model (HMM)
predicted_tags_viterbi_RU1 = [viterbi([word.split()[0] for word in sentence], df_transition_ru, df_emission_ru, sr_start_probabilities_ru) for sentence in ru_observations]

# Create an empty list to store the combined predictions for the "RU" dataset
predicted_RU1 = []

# Iterate through each sentence in the predicted Viterbi tags for the "RU" dataset
for s in range(len(predicted_tags_viterbi_RU1)):
    
    # Iterate through each word's index in the current sentence
    for i in range(len(predicted_tags_viterbi_RU1[s])):
        
        # Combine the observed word with the predicted tag from Viterbi and append it to predicted_RU list
        predicted_RU1.append(ru_observations[s][i] + " "+ predicted_tags_viterbi_RU1[s][i])
    
    # Append a newline character to separate predictions for different sentences
    predicted_RU1.append('\n')

# Open the 'dev.p2.out' file for writing using UTF-8 encoding

with open('Data/RU/dev.p2.out', 'w', encoding='utf-8') as file:
    
    # Write each modified line back to the output file
    for line in predicted_RU1:
        if line != "\n":
            # Write the line followed by a newline character
            file.write(line + '\n')
        
        else:
            # Write the newline character without an additional newline
            file.write(line)

print("Files created successfully")