from pcl_pangu.context import set_context
from pcl_pangu.model import alpha, evolution, mPangu

import argparse


class PanguInference():
    def __init__(self, input, model='2B6', load='ckpts/pretrained/onnx_int8_pangu_alpha_2b6/', backend='onnx-cpu', batch_size=8):
        '''
        PanguInference
        :param model:
        :param load:
        :param backend:
        :param batch_size:
        '''
        self.input = input
        self.model = model
        self.load = load
        self.backend = backend
        self.batch_size = batch_size
        self.input_file = None
        self.output = None
        self.top_k = 2
        self.top_p = 0.9
        self.gmax_tokens = 800

    def alpha_inference(self):
        '''
        alpha_inference
        :param input: prompt
        :param model: model='350M'/'2B6'/'13B'
        :param load: 'onnx/mode/path'
        :return: output
        '''
        set_context(backend=self.backend)
        if self.backend == 'onnx-cpu':
            # onnx
            alpha_config = alpha.model_config_onnx(model=self.model, load=self.load)
        elif self.backend == 'mindspore':
            # mindspore
            alpha_config = alpha.model_config_npu(model=self.model, model_parallel_size=1,
                                                  batch_size=self.batch_size, load=None, save=None)
        else:
            # pytorch
            alpha_config = alpha.model_config_gpu(model=self.model, model_parallel_size=1,
                                                  batch_size=8, load=None, save=None)

        return alpha.inference(alpha_config, top_k=self.top_k, top_p=self.top_p, input=self.input,
                               input_file=self.input_file, generate_max_tokens=self.gmax_tokens,
                               output_file=self.output)

    def alpha_evolution_inference(self):
        '''
        alpha_evolution_inference
        :param input: prompt
        :param model: model='2B6'/'13B'
        :param load: 'onnx/mode/path'
        :return: output
        '''
        set_context(backend=self.backend)
        if self.backend == 'onnx-cpu':
            # onnx
            evolution_config = evolution.model_config_onnx(model=self.model, load=self.load)
        else:
            # pytorch
            evolution_config = evolution.model_config_gpu(model=self.model, model_parallel_size=1,
                                                          batch_size=self.batch_size, load=None, save=None)

        return evolution.inference(evolution_config, top_k=self.top_k, top_p=self.top_p, input=self.input,
                                   input_file=self.input_file, generate_max_tokens=self.gmax_tokens,
                                   output_file=self.output)

    def alpha_mPangu_inference(self):
        '''
        alpha_mPangu_inference
        :param input: prompt
        :param model: model='2B6'
        :param load: 'onnx/mode/path'
        :return: output
        '''
        set_context(backend=self.backend)
        if self.backend == 'mindspore':
            # mindspore
            mPangu_config = mPangu.model_config_npu(model=self.model, load=self.load)
        else:
            # pytorch
            mPangu_config = mPangu.model_config_gpu(model=self.model, load=self.load)

        return mPangu.inference(mPangu_config, top_k=self.top_k, top_p=self.top_p, input=self.input,
                                input_file=self.input_file, generate_max_tokens=self.gmax_tokens,
                                output_file=self.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using PCL-Pangu LLM.')
    parser.add_argument('--prompt', '-p', default="西红柿炒蛋怎么做？",
                        help='The prompt to generate text from')
    parser.add_argument('--model', '-m', default='2B6', help='The GPT-3 model to use')
    parser.add_argument('--ckpt', '-c', default='ckpts/pretrained/onnx_int8_pangu_alpha_2b6/',
                        help='The checkpoint to use')
    parser.add_argument('--backend', '-b', default='onnx-cpu', help='The backend to use')
    parser.add_argument('--batch_size', '-k', default=8, help='batch size')
    parser.add_argument('--input_file', '-i', default=None, help='input file')
    parser.add_argument('--output', '-o', default=None, help='output file')

    args = parser.parse_args()

    # inference
    PanguInference = PanguInference(input=args.prompt, model=args.model, load=args.ckpt, backend=args.backend)
    output = PanguInference.alpha_inference()
    print("output: ", output)
