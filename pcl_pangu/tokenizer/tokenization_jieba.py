from io import open
import os,sys

TOKENIZER = None

class JIEBATokenizer(object):

    def __init__(self, vocab_file, max_len=None):
        import sentencepiece as spm
        self.max_len = max_len if max_len is not None else int(1e12)
        # self.encoder = json.load(open(vocab_file))
        model_file = vocab_file + ".model"
        vocab_file = vocab_file + ".vocab"
        f = open(vocab_file,'r', encoding='utf-8')
        lines = f.readlines()
        self.encoder = {}
        for line in enumerate(lines):
            key = line[1].split('\t')[0]
            self.encoder[key] = line[0]

        self.decoder = {v:k for k,v in self.encoder.items()}

        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        self.eod_id = self.encoder['<eot>']
        self.eot_id = self.encoder['<eot>']
        self.pad_id = self.encoder['<pad>']
        self.unk = 0

    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    @property
    def eod(self):
        return self.eot_id

    def tokenize(self, text):
        """ Tokenize a string. """
        import jieba
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

def get_tokenizer():
    global TOKENIZER
    current_dir = os.path.dirname(os.path.dirname(__file__))
    vocab = os.path.join(current_dir+'/tokenizer/bpe_4w_pcl/vocab')
    if TOKENIZER is None:
        TOKENIZER = JIEBATokenizer(vocab_file=vocab)

    return TOKENIZER

if __name__ == '__main__':
    ids = [26,   80,  373, 5440, 2527, 1663,   11, 1268, 4896, 7902]
    vocab_file = '/userhome/pclproject/gpt/Megatron-LM-1.1-Pangu/megatron/tokenizer/bpe_4w_pcl/vocab'
    tokenizer = JIEBATokenizer(vocab_file)
    print(tokenizer.convert_ids_to_tokens(ids))
    pass