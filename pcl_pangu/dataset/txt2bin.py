import time
import sys
from pcl_pangu.tokenizer import vocab_4w

def txt2bin(input_glob, output_prefix, vocab_file="vocab_4w"):
    from pcl_pangu.context import check_context
    BACKEND = check_context()
    if not BACKEND == 'pytorch':
        raise ImportError("> txt2bin only support 'pytorch' backend, "
                          "U need to set pcl_pangu.context.set_context backend correctly")
    import torch
    from .preprocess_data_pangu import get_args, nltk_available, Encoder, package_file, indexed_dataset
    import glob
    import nltk
    import multiprocessing

    args = get_args()
    startup_start = time.time()

    if not isinstance(input_glob, str) or input_glob == "":
        raise ImportError("> You need to set input_glob[str] correctly! absolute path is recommended...")
    if not isinstance(output_prefix, str) or output_prefix == "":
        raise ImportError("> You need to set output_prefix[str] correctly! absolute path is recommended...")
    if not isinstance(vocab_file, str) or vocab_file == "":
        raise ImportError("> You need to set vocab_file[str] correctly! absolute path is recommended...")

    if vocab_file == 'vocab_4w':
        args.vocab_file = vocab_4w
    else:
        args.vocab_file = vocab_file

    args.input = input_glob
    args.output_prefix = output_prefix

    print("Opening", args.input)
    file_iter = glob.iglob(args.input)

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    # tokenizer = JIEBATokenizer(vocab_path, tokenizer_path)
    pool = multiprocessing.Pool(args.workers)
    encoded_docs = pool.imap(encoder.encode, package_file(file_iter, 128))  # , all_lens))
    # encoded_docs = map(encoder.encode, fin)
    print('encoded_docs', encoded_docs)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    # print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}{}_{}.bin".format(args.output_prefix,
                                                     key, level)
        output_idx_files[key] = "{}{}_{}.idx".format(args.output_prefix,
                                                     key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                     impl=args.dataset_impl,
                                                     vocab_size=encoder.tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i / elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

    end_time = time.time()
    print('Preprocess data using {}s'.format(end_time - startup_end))


if __name__ == '__main__':
    txt2bin(input_glob="", output_prefix="", vocab_file="")