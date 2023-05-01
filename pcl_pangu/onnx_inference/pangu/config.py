'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import pcl_pangu
from pcl_pangu.model.alpha import MODEL_CONFIG

class PanguConfig(object):
    def __init__(
            self,
            model_config,
            vocab_size_or_config_json_file=40_000,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        config = model_config.model_config
        self.n_ctx = config['max_position_embeddings']
        self.n_positions = config['seq_length']
        self.n_embd = config['hidden_size']
        self.n_layer = config['num_layers']
        self.n_head = config['num_attention_heads']

#
# class PanguConfig(object):
#     def __init__(
#             self,
#             vocab_size_or_config_json_file=40_000,
#             model_size='2B6',
#             layer_norm_epsilon=1e-5,
#             initializer_range=0.02,
#     ):
#         self.vocab_size = vocab_size_or_config_json_file
#         self.layer_norm_epsilon = layer_norm_epsilon
#         self.initializer_range = initializer_range
#         if model_size == '2B6':
#             self.model_2b6()
#         if model_size == '13B':
#             self.model_13b()
#         if model_size == '350M':
#             self.model_350m()
#
#     def model_350m(self):
#         self.n_ctx = 1024
#         self.n_positions = 1024
#         self.n_embd = 2560
#         self.n_layer = 24
#         self.n_head = 16
#
#     def model_2b6(self):
#         self.n_ctx = 1024
#         self.n_positions = 1024
#         self.n_embd = 2560
#         self.n_layer = 32
#         self.n_head = 32
#
#     def model_13b(self):
#         self.n_ctx = 1024
#         self.n_positions = 1024
#         self.n_embd = 5120
#         self.n_layer = 40
#         self.n_head = 40