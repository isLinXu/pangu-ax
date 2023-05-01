import threading
import requests

# 定义下载函数
def download_model(url, output_file):
    """Download a model from Huggingface and save it to disk"""
    """从Huggingface下载模型并将其保存到磁盘"""
    with open(output_file, 'wb') as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # 确认文件大小
            f.write(response.content)
        else:
            downloaded = 0
            total_length = int(total_length)
        for data in response.iter_content(chunk_size=4096):
            downloaded += len(data)
            f.write(data)
            progress = downloaded / total_length * 100
            print(f'\rDownloaded {downloaded}/{total_length} bytes ({progress:.2f}%)', end='')

# 定义下载列表
models = {
    # 'gpt2': 'https://cdn.huggingface.co/gpt2-pytorch_model.bin',
    # 'bert-base-uncased': 'https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin'
    # 'pangu_2_6B': 'https://cdn.huggingface.co/pangu-2_6B-pytorch_model.bin'
    'pangu_2_6B': 'https://huggingface.co/imone/pangu_2_6B/blob/main/pytorch_model.bin'
}

# 定义多线程函数
def run_threads():
    """Run the download function in multiple threads"""
    """在多个线程中运行下载函数"""
    threads = []
    for model in models:
        print(f'Downloading {model}')
        url = models[model]
        output_file = f'{model}.bin'
        t = threading.Thread(target=download_model,
                             args=(url, output_file))
        threads.append(t)
        t.start()
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    run_threads()
