'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import logging
import torch

logger = logging.getLogger(__name__)

def load_weight(model, state_dict):
    state_dict_new = {}
    def map_keys(state_dict, prefix):
        for key in state_dict.keys():
            if isinstance(state_dict[key], dict):
                if key == 'task_embedding':
                    map_keys(state_dict[key], 'embedding' if prefix == '' else prefix + '.' + 'embedding')
                elif key == 'model' or key == 'language_model':
                    map_keys(state_dict[key], prefix)
                else:
                    map_keys(state_dict[key], key if prefix == '' else prefix + '.' + key)
            else:
                state_dict_new[key if prefix == '' else prefix + '.' + key] = state_dict[key]
    map_keys(state_dict, prefix='')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    local_metadata = {}
    prefix = ''
    vocab_size = model.state_dict()['embedding.word_embeddings.weight'].shape[0]
    state_dict_new['embedding.word_embeddings.weight'] = state_dict_new['embedding.word_embeddings.weight'][:vocab_size]
    model.load_state_dict(state_dict_new,strict=False)
    state_dict_model = model.state_dict()
    return model