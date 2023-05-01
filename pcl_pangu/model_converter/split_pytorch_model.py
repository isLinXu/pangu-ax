from pcl_pangu.model import alpha
from pcl_pangu.context import set_context
from pcl_pangu.model.launcher_torch import launch
from loguru import logger
import os, sys


DISTRUBUTED_CONFIG = {
    'nnodes':1,
    'node_rank':0,
    'nproc_per_node':1,
    'master_addr':"localhost",
    'master_port':29503
}

def run_pt(script_args, py_script, **kwargs):
    current_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(current_dir+'/panguAlpha_pytorch')
    py_script = os.path.join(current_dir, py_script)
    logger.info("> Running {} with args: {}".format(py_script, script_args))
    launch(training_script=py_script,
           training_script_args=script_args,
           **DISTRUBUTED_CONFIG,
           **kwargs)

def split_pt(config, merged_path, num_ranks):
    config.save = merged_path
    config.load = merged_path
    script_args = config._get_training_script_args()

    model_save = os.path.join(merged_path, f'{num_ranks}_partition_model')
    if os.path.exists(model_save):
        os.system('rm -rf {}'.format(model_save))
    os.system('mkdir {}'.format(model_save))
    script_args.append('--num-mp-model={}'.format(num_ranks))
    script_args.append('--model-type={}'.format('Pangu'))
    script_args.append('--mp-model-save={}'.format(model_save))
    py_script = 'model/panguAlpha_pytorch/tools/split_full_model_into_mp_model.py'
    run_pt(script_args, py_script)

if __name__ == '__main__':
    from pcl_pangu.model_converter import split_pt
    merged_path = '/home/yands/tmp/numpy_ckpt_2.6b/merged'
    split_pt(alpha.model_config_gpu(model='2B6'),merged_path,2)


    pass