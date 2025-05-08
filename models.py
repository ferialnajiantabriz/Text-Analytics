import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader


class BertTransformer(nn.Module):
    def __init__(
        self,
        nlayer,
        nclass,
        dropout=0.5,
        nfinetune=0,
        speaker_info='none',
        topic_info='none',
        emb_batch=0
    ):
        super(BertTransformer, self).__init__()

        # 1) Load RoBERTa (or BERT)
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        # 2) Freeze all layers first
        for param in self.bert.parameters():
            param.requires_grad = False

        n_layers = 12
        if nfinetune > 0:
            # Unfreeze pooler
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            # Unfreeze last `nfinetune` layers
            for i in range(n_layers - 1, n_layers - 1 - nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        # 4) Define a Transformer encoder to replace the GRU
        #    You can tune nhead, dim_feedforward, activation.
        encoder_layer = TransformerEncoderLayer(
            d_model=nhid,
            nhead=8,                  # Number of attention heads
            dim_feedforward=4 * nhid, # Size of the feedforward layer
            dropout=dropout,
            activation='relu'
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=nlayer)

        # 5) Final classification layer
        self.fc = nn.Linear(nhid, nclass)

        # 6) Speaker embedding (still used if speaker_info='emb_cls')
        #    This can embed 0,1,2 (or “continued, switched, pad”)
        self.speaker_emb = nn.Embedding(3, nhid)

        # 7) Topic embedding (still used if topic_info='emb_cls')
        self.topic_emb = nn.Embedding(100, nhid)

        self.dropout = nn.Dropout(p=dropout)
        self.nclass = nclass
        self.speaker_info = speaker_info
        self.topic_info = topic_info
        self.emb_batch = emb_batch

    def forward(
        self,
        input_ids,
        attention_mask,
        chunk_lens,
        speaker_ids,
        topic_labels
    ):
        """
        Args:
          input_ids: (batch_size, chunk_size, seq_len)
          attention_mask: (batch_size, chunk_size, seq_len)
          chunk_lens: (batch_size, chunk_size)   -- used for RNN packing, we won't use it here unless we do advanced masking.
          speaker_ids: (batch_size, chunk_size)
          topic_labels: (batch_size, chunk_size)
        """
        # 1) Flatten the batch for RoBERTa
        batch_size, chunk_size, seq_len = input_ids.shape

        # Flatten to (batch_size * chunk_size, seq_len)
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        # 2) Run BERT (or RoBERTa) - get the [CLS] token
        if self.training or self.emb_batch == 0:
            # Full batch
            bert_out = self.bert(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )[0]  # shape: (bs * chunk_size, seq_len, hidden_dim)
            embeddings = bert_out[:, 0]  # take the [CLS] vector => shape (bs*chunk_size, hidden_dim)
        else:
            # If emb_batch > 0, do smaller-batch forward passes
            embeddings_list = []
            dataset2 = TensorDataset(input_ids, attention_mask)
            loader = DataLoader(dataset2, batch_size=self.emb_batch)
            for _, small_batch in enumerate(loader):
                sb_input_ids, sb_mask = small_batch
                sb_bert_out = self.bert(
                    sb_input_ids,
                    attention_mask=sb_mask,
                    output_hidden_states=True
                )[0]
                sb_embeddings = sb_bert_out[:, 0]  # [CLS]
                embeddings_list.append(sb_embeddings)
            embeddings = torch.cat(embeddings_list, dim=0)

        # 3) Optionally add speaker + topic embeddings
        nhid = embeddings.shape[-1]
        # Flatten speaker_ids & topic_labels for embedding lookups
        speaker_ids = speaker_ids.reshape(-1)   # shape (bs * chunk_size)
        topic_labels = topic_labels.reshape(-1) # shape (bs * chunk_size)

        if self.speaker_info == 'emb_cls':
            sp_emb = self.speaker_emb(speaker_ids)  # (bs*chunk_size, nhid)
            embeddings = embeddings + sp_emb

        if self.topic_info == 'emb_cls':
            tp_emb = self.topic_emb(topic_labels)
            embeddings = embeddings + tp_emb

        # 4) Reshape to (chunk_size, batch_size, nhid)
      
        embeddings = embeddings.reshape(batch_size, chunk_size, nhid)
        embeddings = embeddings.permute(1, 0, 2)  # (chunk_size, batch_size, nhid)

        #  Minimal approach: no mask 
        # 5) Pass through the Transformer
        outputs = self.encoder(embeddings)  # shape => (chunk_size, batch_size, nhid)
        
        # 6) Dropout + FC
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)  # => (chunk_size, batch_size, nclass)

        # 7) Reshape to (batch_size * chunk_size, nclass)
        outputs = outputs.permute(1, 0, 2)   # => (batch_size, chunk_size, nclass)
        outputs = outputs.reshape(-1, self.nclass)

        return outputs
