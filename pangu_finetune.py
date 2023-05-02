from pcl_pangu.context import set_context
from pcl_pangu.dataset import txt2mindrecord
from pcl_pangu.model import alpha, evolution, mPangu

set_context(backend='mindspore')
data_path = 'path/of/training/dataset'
txt2mindrecord(input_glob='your/txt/path/*.txt', output_prefix=data_path)
config = alpha.model_config_npu(model='350M',
                                load='path/of/your/existing/ckpt',
                                data_path=data_path)
alpha.fine_tune(config)
