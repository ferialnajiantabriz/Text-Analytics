import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import pickle

### Dialogue act label encoding, SWDA
# {...}  # [unchanged comments about label mappings]

### Dialogue act label encoding, MRDA
# {...}

### Dialogue act label encoding, DyDA
# {...}

### Topic label encoding, DyDA
# {...}

class DialogueActData(Dataset):
    def __init__(self, corpus, phase, chunk_size=0):
        os.makedirs('processed_data', exist_ok=True)
        if phase == 'train':
            data_path = f'processed_data/{corpus}_{chunk_size}_{phase}.pkl'
        else:
            data_path = f'processed_data/{corpus}_{phase}.pkl'

        print(f'Tokenizing {phase}....')
        if os.path.exists(data_path):
            # Load cached data if it already exists
            input_ids_, attention_mask_, labels_, chunk_lens_, speaker_ids_, topic_labels_ = \
                pickle.load(open(data_path, 'rb'))
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('roberta-base')

            # 1) Read CSV
            df = pd.read_csv(f'data/{corpus}/{phase}.csv')

            # 2) Create a turn_type column: 0 = continued speaker, 1 = switched speaker
            #    For the first utterance in each conversation, we'll default to 1 (i.e. "switch").
            df['turn_type'] = 0
            conv_ids_unique = df['conv_id'].unique()
            for c_id in conv_ids_unique:
                mask_conv = (df['conv_id'] == c_id)
                df_conv = df[mask_conv].copy()
                df_conv = df_conv.sort_index()  # make sure in chronological order if needed

                prev_speaker = None
                local_turn_types = []
                for idx, row in df_conv.iterrows():
                    current_speaker = row['speaker']
                    if prev_speaker is None:
                        # first utterance => 1 indicates "start or switched"
                        local_turn_types.append(1)
                    else:
                        local_turn_types.append(0 if current_speaker == prev_speaker else 1)
                    prev_speaker = current_speaker

                df.loc[mask_conv, 'turn_type'] = local_turn_types

            # 3) Figure out chunk_size if user set 0 for training or for other phases
            max_conv_len = df['conv_id'].value_counts().max()
            if (chunk_size == 0 and phase == 'train') or phase != 'train':
                chunk_size = max_conv_len

            # 4) Tokenize the text
            texts_all = df['text'].tolist()
            encodings_all = tokenizer(texts_all, truncation=True, padding=True)
            input_ids_all = np.array(encodings_all['input_ids'])
            attention_mask_all = np.array(encodings_all['attention_mask'])

            # Prepare arrays to hold chunked data
            input_ids_ = []
            attention_mask_ = []
            labels_ = []
            chunk_lens_ = []
            speaker_ids_ = []   # <-- This will store turn_type now
            topic_labels_ = []

            # 5) Process each conversation
            conv_ids = df['conv_id'].unique()
            for conv_id in conv_ids:
                mask_conv = (df['conv_id'] == conv_id)
                df_conv = df[mask_conv]

                # Extract tokenized slices for utterances in this conversation
                conv_input_ids = input_ids_all[mask_conv]
                conv_attention_mask = attention_mask_all[mask_conv]

                # Instead of original speaker, we'll use the turn_type
                conv_turn_type = df_conv['turn_type'].values

                # The rest is the same as before
                conv_labels = df_conv['act'].values
                conv_topic = df_conv['topic'].values

                # Break this conversation into chunks
                chunk_indices = list(range(0, df_conv.shape[0], chunk_size)) + [df_conv.shape[0]]
                for i in range(len(chunk_indices) - 1):
                    idx1, idx2 = chunk_indices[i], chunk_indices[i + 1]

                    chunk_input_ids = conv_input_ids[idx1: idx2].tolist()
                    chunk_attention_mask = conv_attention_mask[idx1: idx2].tolist()
                    chunk_labels = conv_labels[idx1: idx2].tolist()
                    chunk_turn_type_ids = conv_turn_type[idx1: idx2].tolist()
                    chunk_topic_labels = conv_topic[idx1: idx2].tolist()
                    chunk_len = idx2 - idx1

                    # 6) Handle padding if chunk < chunk_size
                    if chunk_len < chunk_size:
                        length1 = chunk_len
                        length2 = chunk_size - length1
                        # pad input_ids + attention_mask
                        encodings_pad = [[0] * len(input_ids_all[0])] * length2
                        chunk_input_ids.extend(encodings_pad)
                        chunk_attention_mask.extend(encodings_pad)
                        # pad labels with -1
                        labels_padding = np.array([-1] * length2)
                        chunk_labels = np.concatenate((chunk_labels, labels_padding), axis=0)
                        # pad turn_type with 2 (since 0=continued, 1=switched, 2=pad)
                        turn_type_padding = np.array([2] * length2)
                        chunk_turn_type_ids = np.concatenate((chunk_turn_type_ids, turn_type_padding), axis=0)
                        # pad topic label with 99
                        topic_labels_padding = np.array([99] * length2)
                        chunk_topic_labels = np.concatenate((chunk_topic_labels, topic_labels_padding), axis=0)

                    # 7) Append chunk to lists
                    input_ids_.append(chunk_input_ids)
                    attention_mask_.append(chunk_attention_mask)
                    labels_.append(chunk_labels)
                    chunk_lens_.append(chunk_len)
                    speaker_ids_.append(chunk_turn_type_ids)  # <--- saving turn_type as "speaker_ids"
                    topic_labels_.append(chunk_topic_labels)

            # 8) Save processed data to disk
            pickle.dump(
                (input_ids_, attention_mask_, labels_, chunk_lens_, speaker_ids_, topic_labels_),
                open(data_path, 'wb')
            )

        print('Done')

        # 9) Assign final lists to self.*
        self.input_ids = input_ids_
        self.attention_mask = attention_mask_
        self.labels = labels_
        self.chunk_lens = chunk_lens_
        self.speaker_ids = speaker_ids_  # these are turn_type IDs now
        self.topic_labels = topic_labels_

    def __getitem__(self, index):
        item = {
            'input_ids': torch.tensor(self.input_ids[index]),
            'attention_mask': torch.tensor(self.attention_mask[index]),
            'labels': torch.tensor(self.labels[index]),
            'chunk_lens': torch.tensor(self.chunk_lens[index]),
            # "speaker_ids" is actually "turn_type_ids" in the new logic
            'speaker_ids': torch.tensor(self.speaker_ids[index], dtype=torch.long),
            'topic_labels': torch.tensor(self.topic_labels[index], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.labels)


def data_loader(corpus, phase, batch_size, chunk_size=0, shuffle=False):
    dataset = DialogueActData(corpus, phase, chunk_size=chunk_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
