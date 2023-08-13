#!/usr/bin/env python
# coding: utf-8

# # Part 4

# We can modify the approach by using an n-th order HMM, whereby the HMM can depend on not only the current state but also the previous n-states. In this case, we have adopted n=2.

# In[1]:


#Import libraries
import pandas as pd
import numpy as np


# In[2]:


# 1. Extracting data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip().split('\n\n')
    return [sentence.split('\n') for sentence in data]


# In[3]


#set to RU/ES
lang = 'ES'
#set to test/dev
flag = 'test'

# Adjusting the paths for ES and RU datasets
train_data = load_data(f'Data/{lang}/train')
with open(f'Data/{lang}/dev.out', 'r', encoding='utf-8') as f:
    dev_tags_actual = [sentence.split() for sentence in f.read().strip().split('\n\n')]


if(flag=='test'):
    dev_in_data = load_data(f'Test/{lang}/{flag}.in')
    dev_out_data = f'Test/{lang}/{flag}.p4.out'
    

elif(flag=='dev'):
    dev_in_data = load_data(f'Data/{lang}/{flag}.in')
    dev_out_data = f'Data/{lang}/{flag}.p4.out'
    with open(f'Data/{lang}/dev.out', 'r', encoding='utf-8') as f:
        dev_tags_actual = [sentence.split() for sentence in f.read().strip().split('\n\n')]

# Adjusting the paths for ES and RU datasets
train_data = load_data(f'Data/{lang}/train')




# In[4]:


states = {}
observations = {}

for sentence in train_data:
    for line in sentence:
        word, tag = line.strip().split(maxsplit=1)
        states[tag] = states.get(tag, 0) + 1
        if tag not in observations:
            observations[tag] = {}
        observations[tag][word] = observations[tag].get(word, 0) + 1

state_list = list(states.keys())


# In[5]:


from collections import defaultdict

def compute_probabilities(data, state_list):
    # Initialize counts
    start_count = defaultdict(int)
    transition_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    emission_count = defaultdict(lambda: defaultdict(int))

    # Populate counts
    for sentence in data:
        if len(sentence) < 3:  # Skip sentences that are too short
            continue

        # Splitting and extracting the word and state for first two words/states
        word1, state1 = sentence[0].strip().split()
        word2, state2 = sentence[1].strip().split()
        start_count[state1] += 1
        emission_count[state1][word1] += 1
        emission_count[state2][word2] += 1

        # Rest of the sentence
        for i in range(2, len(sentence)):
            word3, state3 = sentence[i].strip().split(maxsplit=1)
            transition_count[state1][state2][state3] += 1
            emission_count[state3][word3] += 1
            state1, state2 = state2, state3

    # Convert counts to probabilities
    # Start transition probabilities
    start_transition_prob = {state: count / sum(start_count.values()) for state, count in start_count.items()}

    # Transition probabilities
    transition_prob = {}
    for s1, s1_dict in transition_count.items():
        transition_prob[s1] = {}
        for s2, s2_dict in s1_dict.items():
            transition_prob[s1][s2] = {}
            for s3, count in s2_dict.items():
                total = sum(s2_dict.values())
                transition_prob[s1][s2][s3] = count / total if total != 0 else 0.0

    # Emission probabilities
    emission_prob = {}
    for state, word_count in emission_count.items():
        emission_prob[state] = {}
        total = sum(word_count.values())
        for word, count in word_count.items():
            emission_prob[state][word] = count / total

    return start_transition_prob, transition_prob, emission_prob


# In[6]:


state_list = list(states.keys())
start_transition_prob, transition_prob, emission_prob = compute_probabilities(train_data, state_list)


# In[7]:


transition_prob


# In[8]:


def viterbi(obs, states, start_p, trans_p, emit_p):
    # Initialize the Viterbi matrix. 
    V = [{}]

    # Initialize the first column of the matrix with the start probabilities
    for st in states:
        V[0][st] = {"prob": start_p.get(st, 0) * emit_p[st].get(obs[0], 0), "prev": None}

    # Main loop through the observations updating the Viterbi matrix
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            # For each state, find the maximum transition probability 
            # considering all possible previous state combinations.
            max_trans_prob, prev_st1_max, prev_st2_max = max(
                (V[t-1][prev_st1]["prob"] * trans_p[prev_st1].get(prev_st2, {}).get(st, 0), prev_st1, prev_st2)
                for prev_st1 in states for prev_st2 in states
            )

            # Multiply the max transition probability with emission probability
            max_prob = max_trans_prob * emit_p[st].get(obs[t], 0)

            # Store the maximum probability and previous state information
            V[t][st] = {"prob": max_prob, "prev": (prev_st1_max, prev_st2_max)}

    # Now, backtrack to find the most probable sequence of states
    opt = []

    # Find the state with the maximum probability for the last observation
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None

    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break

    # Backtrack through the Viterbi matrix to find the sequence of states
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"][1])
        previous = V[t + 1][previous]["prev"][1]

    # Return the most probable sequence of states
    return opt


# In[9]:


predicted_tags_viterbi = [viterbi([word.split()[0] for word in sentence], state_list, start_transition_prob, transition_prob, emission_prob) for sentence in dev_in_data]


# In[10]:


def extract_entities_from_tags(tags):
    entities = []
    entity = []
    for tag in tags:
        if tag.startswith("B-"):
            if entity:
                entities.append(tuple(entity))
                entity = []
            entity.append(tag)
        elif tag.startswith("I-"):
            entity.append(tag)
        else:
            if entity:
                entities.append(tuple(entity))
                entity = []
    if entity:
        entities.append(tuple(entity))
    return set(entities)

TP = 0
FP = 0
FN = 0


if(flag=='dev'):
    for pred, actual in zip(predicted_tags_viterbi, dev_tags_actual):
        predicted_entities = extract_entities_from_tags(pred)
        actual_entities = extract_entities_from_tags(actual)
        TP += len(predicted_entities.intersection(actual_entities))
        FP += len(predicted_entities - actual_entities)
        FN += len(actual_entities - predicted_entities)

    if TP + FP == 0:
        precision = 1.0  # or 0.0, depending on how you want to define it in this case
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 1.0  # or 0.0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * precision * recall / (precision + recall)


    print("Precision:", precision)
    print("Recall:", recall)
    print("F-score:", f_score)


# In[11]:


## predicted = []

# Open the output file for writing using UTF-8 encoding
with open(dev_out_data, 'w', encoding='utf-8') as file:
    for s in range(len(dev_in_data)):
        for wi in range(len(dev_in_data[s])):
            line = dev_in_data[s][wi]+" "+predicted_tags_viterbi[s][wi]
            if line != "\n":
                file.write(line + '\n')
            else:
                file.write(line)
        file.write('\n')


# In[ ]:




