import torch
from torch import nn
import numpy as np

from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
  def __init__(self, config, opt):
    super(SelfAttention, self).__init__()
    self.opt = opt
    self.config = config
    self.SA = BertSelfAttention(config)
    self.tanh = torch.nn.Tanh()

  def forward(self, inputs):
    zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt['max_seq_len']),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt['device'])
    SA_out = self.SA(inputs, zero_tensor)
    return self.tanh(SA_out[0])


# Local Context Focus
class LFC_BERT(nn.Module):
  def __init__(self, bert, opt):
    super(LFC_BERT, self).__init__()

    self.bert_spc = bert
    self.opt = opt

    self.bert_local = bert
    self.dropout = nn.Dropout(opt['dropout'])
    self.bert_SA = SelfAttention(bert.config, opt)
    self.linear_double = nn.Linear(opt['bert_dim'] * 2, opt['bert_dim'])
    self.linear_single = nn.Linear(opt['bert_dim'], opt['bert_dim'])
    self.bert_pooler = BertPooler(bert.config)
    self.dense = nn.Linear(opt['bert_dim'], opt['polarities_dim'])

  def feature_dymanic_mask(self, text_local_indexes, aspect_indexes):
    texts = text_local_indexes.cpu().numpy()
    asps = aspect_indexes.cpu().numpy()
    mask_len = self.opt['SRD']
    masked_text_raw_indexes = np.ones((text_local_indexes.size(0), self.opt['max_seq_len'], self.opt['bert_dim']), dtype=np.float32)

    for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
      asp_len = np.count_nonzero(asps[asp_i]) - 2
      try:
        asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
      except:
        continue
      if asp_begin >= mask_len:
        mask_begin = asp_begin - mask_len
      else:
        mask_begin = 0
      for i in range(mask_begin):
        masked_text_raw_indexes[text_i][i] = np.zeros((self.opt['bert_dim']), dtype=np.float)
      for j in range(asp_begin + asp_len + mask_len, self.opt['max_seq_len']):
        masked_text_raw_indexes[text_i][j] = np.zeros((self.opt['bert_dim']), dtype=np.float)
    masked_text_raw_indexes = torch.from_numpy(masked_text_raw_indexes)
    return masked_text_raw_indexes.to(self.opt['device'])

  def forward(self, inputs):
    text_bert_indexes = inputs[0]
    bert_segments_ids = inputs[1]
    text_local_indexes = inputs[2]
    aspect_indexes = inputs[3]

    bert_spc_out = self.bert_spc(text_bert_indexes, token_type_ids=bert_segments_ids)
    bert_spc_out = self.dropout(bert_spc_out.last_hidden_state)

    bert_local_out = self.bert_local(text_local_indexes)
    bert_local_out = self.dropout(bert_local_out.last_hidden_state)

    masked_local_text_vec = self.feature_dymanic_mask(text_local_indexes, aspect_indexes)
    bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

    out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
    mean_pool = self.linear_double(out_cat)
    self_attention_out = self.bert_SA(mean_pool)
    pooled_out = self.bert_pooler(self_attention_out)
    dense_out = self.dense(pooled_out)

    return dense_out