import requests
import os
# from utils.download import get_path_from_url
import utils.download as download
# from utils.model import HuggingfaceDownloader

HOME = os.getcwd()
# If the default HOME dir does not support writing, we
# will create a temporary folder to store the cache files.

DATA_HOME = os.path.join(HOME, 'ckpt')

def urldownload(url,filename=None):
    """
    下载文件到指定目录
    :param url: 文件下载的url
    :param filename: 要存放的目录及文件名，例如：./test.xls
    :return:
    """
    down_res = requests.get(url)
    with open(filename,'wb') as file:
        file.write(down_res.content)

def download_hf_model(model_name, output_dir='.'):
    downloader = HuggingfaceDownloader(model_name=model_name, output_dir=output_dir)
    downloader.run()

if __name__ == '__main__':
    # urldownload('https://openi.pcl.ac.cn/attachments/59fac5b4-ec54-493f-95ba-d0b724f39ed6','2.6B_part0.rar')
    ckpt_path = '2.6B_part0.rar'
    # if not os.path.isdir(os.path.join(DATA_HOME, ckpt_path)):
    #     download.get_path_from_url(
    #         url="https://openi.pcl.ac.cn/attachments/59fac5b4-ec54-493f-95ba-d0b724f39ed6",
    #         DATA_HOME="/Users/gatilin/PycharmProjects/pangu-ax/ckpt",
    #         md5="3904bb0e551f36206cfca96ab0f63cba",
    #         decompress=True,
    #     )
    # download._download(url="https://openi.pcl.ac.cn/attachments/59fac5b4-ec54-493f-95ba-d0b724f39ed6",
    #                    path="/Users/gatilin/PycharmProjects/pangu-ax/ckpt/2.6B_part0.rar",
    #                    md5sum="3904bb0e551f36206cfca96ab0f63cba",
    #                    method='get')

    # https://huggingface.co/imone/pangu_2_6B/blob/main/pytorch_model.bin
    download._download(url="https://openi.pcl.ac.cn/attachments/452d8bf4-54d9-4e11-a008-654b7e3d0382?type=0%26%2334%3B",
                       path="/Users/gatilin/PycharmProjects/pangu-ax/ckpt/2.6B/",
                       md5sum=None,
                       method='get')


