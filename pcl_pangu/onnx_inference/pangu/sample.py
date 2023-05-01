'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from scipy.special import softmax
import time

def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None,
                    context=None, temperature=1, top_k=0, device='cuda',
                    sample=True, past = None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past, position_ids=None, lm_labels=None, token_type_ids=None)
            logits = torch.tensor(logits).to(device)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        res, num = func(*args, **kwargs)
        num += 1e-10
        end = time.time()
        print(f'Generating {int(num)} tokens costs {end - start}s')
        print(f'Per token costs {(end - start)/num}s')
        return res
    return wrapper


def generate(
        model,
        text,
        tokenizer,
        past,
        max_len = 100,
        temperature = 1.0,
        top_p = 0.95,
        top_k = 50,
        eod=None,
        additional_eod=[],
        ban = []
):
    if eod is None:
        eod = [tokenizer.eod_id, tokenizer.eot_id]
    ids = tokenizer.encode(text)
    len_input = len(ids)
    predict_ids = []
    kv_cache = None
    next_token = np.array([ids], dtype=np.int32)

    start_time = time.time()
    for i in range(max_len):
        logits, past = model(next_token, past=past, position_ids=None, lm_labels=None, token_type_ids=None)

        for x in ban:
            logits[:, -1, x] = -9999

        logits = logits / temperature
        scores = softmax(logits[:, -1, :])
        next_probs = np.sort(scores)[:, ::-1]
        if top_p > 0.0 and top_p < 1.0:
            next_probs = next_probs[:, :int(next_probs.shape[1] * (1 - top_p))]
        if top_k > 0 and top_k < next_probs.shape[1]:
            next_probs = next_probs[:, :top_k]
        next_probs_1 = next_probs / next_probs.sum(axis=1).reshape((-1, 1))

        next_tokens = np.argsort(scores)[:, ::-1]
        if top_p > 0.0 and top_p < 1.0:
            next_tokens = next_tokens[:, :int(next_tokens.shape[1] * (1 - top_p))]
        if top_k > 0 and top_k < next_tokens.shape[1]:
            next_tokens = next_tokens[:, :top_k]

        next_token = np.random.choice(next_tokens[0], p=next_probs_1[0])
        if eod is not None and next_token in eod:
            break
        if next_token in additional_eod or tokenizer.decode([int(next_token)]) in additional_eod:
            break
        predict_ids.append(next_token)
        next_token = np.array([[next_token]], dtype=np.int32)
    end_time = time.time()
    inference_tokens_number = len(predict_ids)
    print(">>> inference speed: {:.4f}ms/token >>>\n".format(1000.0 * (end_time - start_time)/float(inference_tokens_number)))
    return text + tokenizer.decode([int(x) for x in predict_ids])
