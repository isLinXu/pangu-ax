# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))


from megatron.text_generation_utils import pad_batch, get_batch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
import torch.nn.functional as F
# from megatron.model.transformer import LayerNorm


def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=False)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--isEvolution", type=bool, default=False,
                       help='if using Evolution model')
    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top-p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=5,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--generate_max_tokens", type=int, default=128,
                       help='generate_max_tokens from (0, 800]')
    group.add_argument("--sample-input", type=str, default=None,
                       help='Input one sample to generate.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')

    return parser


def top_k_logits(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits = logits.view(batch_size, -1).contiguous()
        for logit in logits:
            sorted_logits, sorted_indices = torch.sort(logit, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logit[indices_to_remove] = filter_value

        logits = logits.view(batch_size, -1).contiguous()

    return logits


def generate(model, context_tokens, args, tokenizer, max_num=50):

    valid_length = len(context_tokens)
    context_tokens_, context_lengths = pad_batch([context_tokens],
                                                 tokenizer.pad_id, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens_)
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
    type_ids = None
    bs,_  = tokens.shape
    cnt = 0
    while valid_length < args.seq_length:
        with torch.no_grad():
            logits = model(tokens,
                           position_ids,
                           attention_mask,
                           tokentype_ids=type_ids,
                           forward_method_parallel_output=False)
        logits = logits[:,:,:tokenizer.vocab_size].cpu()
        logits = logits.numpy()
        logits = logits.reshape(bs, args.seq_length, -1)
        probs = logits[0, valid_length-1, :]
        p_args = probs.argsort()[::-1][:args.top_k]

        p = probs[p_args]
        p = p / sum(p)
        for i in range(1000):
            target_index = np.random.choice(len(p), p=p)
            if p_args[target_index] != tokenizer.unk:
                break

        if p_args[target_index] == tokenizer.eod or \
                valid_length == args.seq_length-1 or cnt>=max_num:
            outputs = tokens.cpu().numpy()
            break
        tokens[0][valid_length] = p_args[target_index]
        valid_length += 1
        cnt += 1

    length = np.sum(outputs != tokenizer.pad_id)
    outputs = outputs[0][:length]
    return outputs

def generate_samples_cftpd(model, context_tokens, args, tokenizer, max_num=50, top_k=0, top_p=0.9, temperature=1.0):
    valid_length = len(context_tokens)
    if valid_length > args.seq_length - max_num:
        valid_length = args.seq_length - max_num
        context_tokens = context_tokens[-valid_length:]

    context_tokens_, context_lengths = pad_batch([context_tokens],
                                                 tokenizer.pad_id, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens_)
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
    type_ids = None
    bs, _ = tokens.shape
    cnt = 0
    with torch.no_grad():
        while valid_length < args.seq_length:
            logits = model(tokens,
                           position_ids,
                           attention_mask,
                           tokentype_ids=type_ids,
                           forward_method_parallel_output=False)
            probs = logits[:, valid_length - 1, :] / temperature
            probs = top_k_logits(probs, top_k=top_k, top_p=top_p)
            probs = F.softmax(probs, dim=-1)
            target_index = torch.multinomial(probs.float(), num_samples=1).squeeze(1)

            if target_index == tokenizer.eod or valid_length == args.seq_length - 1 or cnt >= max_num:
                outputs = tokens
                break
            tokens[0][valid_length] = target_index
            valid_length += 1
            cnt += 1

    length = torch.sum(outputs != tokenizer.pad_id)
    outputs = outputs[0][:length]
    return outputs


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    args = get_args()
    # Set up model and load checkpoint.
    model = get_model(model_provider)
    model.eval()

    assert args.top_k >= 1, "> reset top_k from [1, 512], bigger the value, richer the generation diversity!"
    assert args.top_p > 0.0, "> reset top_p from [0.1, 1), bigger the value, richer the generation diversity!"
    assert args.top_p < 1.0, "> reset top_p from [0.1, 1), bigger the value, richer the generation diversity!"

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    if args.sample_input is not None:
        raw_text = bytes.fromhex(args.sample_input).decode('utf-8')
        tokenizer = get_tokenizer()
        context_tokens = tokenizer.tokenize(raw_text)

        # output_ids = generate(model, context_tokens, args, tokenizer)
        output_ids = generate_samples_cftpd(model, context_tokens, args, tokenizer,
                                            max_num=args.generate_max_tokens, top_k=args.top_k, top_p=args.top_p)
        output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
        print('Input is:', raw_text)
        print('Output is:', output_samples[len(raw_text):], flush=True)
        print()
    if args.sample_input_file is not None:
        raw_texts = open(args.sample_input_file, 'r').read().split('\n\n')
        write_output = print
        if args.sample_output_file is not None:
            output_file = open(args.sample_output_file, 'w')
            write_output = lambda x: output_file.write(x + '\n')
        tokenizer = get_tokenizer()
        for raw_text in raw_texts:
            context_tokens = tokenizer.tokenize(raw_text)
            # output_ids = generate(model, context_tokens, args, tokenizer)
            output_ids = generate_samples_cftpd(model, context_tokens, args, tokenizer,
                                                max_num=args.generate_max_tokens, top_k=args.top_k, top_p=args.top_p)
            output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
            write_output('Input is: ' + raw_text)
            write_output('Output is: ' + ' '.join(output_samples[len(raw_text):]))
            write_output()


if __name__ == "__main__":

    main()
