from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu

set_context(backend='onnx-cpu')

config = alpha.model_config_onnx(model='2B6',load='/Users/gatilin/Pan/ckpts/onnx_int8_pangu_alpha_2b6/')

# config = alpha.model_config_onnx(model='2B6'/'13B',load='onnx/mode/path')
output_file = "output/output.txt"
alpha.inference(config,top_k=1, top_p=0.9,input='四川的省会是?',
                input_file=None,
                generate_max_tokens=800, output_file=output_file, oneCardInference=True
                )

