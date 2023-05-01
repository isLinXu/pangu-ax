import logging
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

def txt2mindrecord(input_glob, output_prefix, vocab_file='vocab_4w'):
    from pcl_pangu.context import check_context
    BACKEND = check_context()
    print("-------------------------------------------------------------------------------------------"
          "\n preprocess txt2mindrecord using {}. ['vocab_4w' / 'vocab_13w' / 'your_vocab_file_path']."
          "\n # 'vocab_4w' for [alpha and evolution], "
          "\n # 'vocab_13w' for [mPangu],"
          "\n # or manual setting 'your_vocab_file_path', which will using JIEBATokenizer!"
          "\n-----------------------------------------------------------------------------------------"
          .format(vocab_file))

    if not BACKEND == 'mindspore':
        raise ImportError("> txt2bin only support 'mindspore' backend, "
                          "U need to set pcl_pangu.context.set_context backend correctly")
    from .pre_process_chinese import tokenizer, modelarts_flag, args, JIEBATokenizer, \
        FileWriter, tokenize_lambada, tokenize_wiki, divideIntoNstrand, task_unit, task_unit_mPangu, \
        package_file, setup_tokenizer, setup_writer
    try:
        import moxing as mox
    except:
        logging.INFO('> Using NPU Machine, not in modelarts ENV !!')
    import glob
    import random
    from multiprocessing import Pool


    global tokenizer

    if not isinstance(input_glob, str) or input_glob == "":
        raise ImportError("> You need to set input_glob[str] correctly! absolute path is recommended...")
    if not isinstance(output_prefix, str) or output_prefix == "":
        raise ImportError("> You need to set output_prefix[str] correctly! absolute path is recommended...")
    if not isinstance(vocab_file, str) or vocab_file == "":
        raise ImportError("> You need to set vocab_file[str] correctly! absolute path is recommended...")

    if "s3:" in input_glob or "obs:" in input_glob:
        input_txt_from_obs_glob = mox.file.glob(input_glob)
        assert not len(input_txt_from_obs_glob) == 0, "> OBS findError: cannot find *.txt in your input_glob" \
                                                      ", check again"
        print('> Copy *.txt from OBS')
        if modelarts_flag:
            copy_dir = './cache/dataset/'
            if not os.path.exists(copy_dir):
                os.makedirs(copy_dir)
            for item in input_txt_from_obs_glob:
                file_name = item.split('/')[-1]
                mox.file.copy(item, copy_dir + file_name)
            print('#### Moxing copy dataset succsseed! ####')
            args.input_glob = './cache/dataset/*'
        else:
            raise ImportError("> 1. check you input_glob path if start with ['s3:'] or ['obs:'],"
                              "> 2. check you container env if in [modelarts]")
    else:
        args.input_glob = input_glob

    if "s3:" in vocab_file or "obs:" in vocab_file:
        if modelarts_flag:
            copy_dir = './cache/bpe_vocab/'
            if not os.path.exists(copy_dir):
                os.makedirs(copy_dir)
            mox.file.copy(vocab_file + '.vocab', copy_dir + 'vocab.vocab')
            mox.file.copy(vocab_file + '.model', copy_dir + 'vocab.model')
            print('#### Moxing copy vocab files succsseed! ####')
            vocab_file = './cache/bpe_vocab/vocab'
            args.vocab_file = vocab_file
        else:
            raise ImportError("> 1. check you vocab_file path if start with ['s3:'] or ['obs:'],"
                              "> 2. check you container env if in [modelarts]")
    else:
        args.vocab_file = vocab_file
    #######################################
    tokenizer = setup_tokenizer()
    #######################################
    PAD = tokenizer.pad_id
    EOT = tokenizer.eot_id
    print('pad id :', PAD)
    print('eot id :', EOT)
    print('vocab size :', tokenizer.vocab_size)

    if "s3:" in output_prefix or "obs:" in output_prefix:
        if modelarts_flag:
            upload2obs_dir, upload2obs_file_name = os.path.split(os.path.abspath(output_prefix))
            output_mindrecord_dir = './cache/dataMindrecord/'
            if not os.path.exists(output_mindrecord_dir):
                os.makedirs(output_mindrecord_dir)
            args.output_dir = output_mindrecord_dir + upload2obs_file_name
        else:
            raise ImportError("> 1. check you input_glob path if start with ['s3:'] or ['obs:'],"
                              "> 2. check you container env if in [modelarts]")
    else:
        args.output_dir = output_prefix

    out_dir, out_file = os.path.split(os.path.abspath(args.output_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    writer = setup_writer(args)
    ###
    transforms_count = 0
    if args.dataset_type == 'wiki':
        for x in tokenize_wiki(args.input_glob):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'lambada':
        for x in tokenize_lambada(args.input_glob):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'openwebtext':
        # file_iter = glob.iglob(args.input_glob)
        input_files = list(glob.iglob(args.input_glob))
        # input_files = input_files*2
        input_files.sort()
        random.seed(10)
        random.shuffle(input_files)

        all = int(args.rankOfCluster.split('of')[1])
        order = int(args.rankOfCluster.split('of')[0])
        print(order, '  of   ', all)
        print('num files of cluster : ', len(input_files))
        input_files = divideIntoNstrand(input_files, all)[order]
        print('num files of this machine : ', len(input_files))

        file_iter = (x for x in input_files)
        if vocab_file == 'vocab_13w':
            with Pool(processes=args.num_process) as pool:
                pool.map(task_unit_mPangu, package_file(file_iter, args.file_batch_size))
        else:
            with Pool(processes=args.num_process) as pool:
                pool.map(task_unit, package_file(file_iter, args.file_batch_size))
    else:
        raise ValueError(
            "Not support dataset type: {}".format(args.dataset_type))

    writer.commit()

    if "s3:" in output_prefix or "obs:" in output_prefix:
        if modelarts_flag:
            upload2obs_dir, _ = os.path.split(output_prefix)
            mox.file.copy_parallel(out_dir, upload2obs_dir)
            print('#### Moxing upload output mindrecord to [{}] succsseed! ####'.format(upload2obs_dir))
            print("Transform finished, output files refer: {}".format(output_prefix))
    else:
        print("Transform finished, output files refer: {}".format(args.output_dir))

if __name__ == '__main__':
    txt2mindrecord(input_glob="", output_prefix="")

