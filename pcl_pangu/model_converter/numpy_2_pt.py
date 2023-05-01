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
    current_dir = os.path.join(current_dir, 'model_converter')
    sys.path.append(current_dir+'/panguAlpha_pytorch')
    py_script = os.path.join(current_dir, py_script)
    logger.info("> Running {} with args: {}".format(py_script, script_args))
    launch(training_script=py_script,
           training_script_args=script_args,
           **DISTRUBUTED_CONFIG,
           **kwargs)

def numpy_2_pt(config,numpy_ckpt):

    script_args = config._get_training_script_args()
    script_args.append('--npy-ckpt-path={}'.format(numpy_ckpt))
    script_args.append('--model-type={}'.format('Pangu'))
    py_script = 'numpy_ckpt_2_pytorch.py'
    run_pt(script_args, py_script)

if __name__ == '__main__':

    numpy_ckpt = '/home/yands/tmp/numpy_ckpt_2.6b'
    numpy_2_pt(alpha.model_config_gpu(model='2B6'),
               numpy_ckpt)
    pass