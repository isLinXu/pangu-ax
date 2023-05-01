from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu
from pcl_pangu.model_converter import pt_2_onnx8

set_context(backend='onnx-cpu')
config = alpha.model_config_onnx(model='2B6')
pt_2_onnx8(model_name='alpha-2b6',
           pt_path='/Users/gatilin/Pan/ckpts/盘古α_2.6B/',
           model_config=config,
           onnx_ckpt_root_dir='/Users/gatilin/Pan/ckpts/onnx')

