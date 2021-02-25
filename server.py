from flask import Flask, request, jsonify

import torch
import torch.nn.functional as F
from lcf_bert import LFC_BERT
from transformers import BertModel
from data_util import Tokenizer4Bert, preprocess_text
import numpy as np

PORT = 8080
app = Flask('ABSC lsc-bert')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPTIONS = {
    'max_seq_len': 85,
    'bert_dim': 768,
    'dropout': 0.1,
    'device': DEVICE,
    'SRD': 3,
    'polarities_dim': 3,
    'pretrained_bert_name': 'bert-base-uncased'
}


def create_model():
    bert = BertModel.from_pretrained(OPTIONS['pretrained_bert_name'])

    model = LFC_BERT(bert, OPTIONS).to(OPTIONS['device'])
    model.load_state_dict(torch.load('model path', map_location=DEVICE))
    return model


tokenizer = Tokenizer4Bert(OPTIONS['max_seq_len'], OPTIONS['pretrained_bert_name'])
model = create_model()


@app.route('/')
def root():
    return 'ABSC lsc-bert'


@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json().get('text')
    aspect = request.get_json().get('aspect')

    data = preprocess_text(text, aspect, tokenizer, OPTIONS['device'])
    result = model(data)

    response = {
        'result': int(np.argmax(F.softmax(result, dim=-1).cpu().detach().numpy()))
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)