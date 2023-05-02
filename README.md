# pangu-ax

![banner](./figure/banner.png)

![GitHub watchers](https://img.shields.io/github/watchers/isLinXu/pangu-ax.svg?style=social) ![GitHub stars](https://img.shields.io/github/stars/isLinXu/pangu-ax.svg?style=social) ![GitHub forks](https://img.shields.io/github/forks/isLinXu/pangu-ax.svg?style=social) ![GitHub followers](https://img.shields.io/github/followers/isLinXu.svg?style=social)
 [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/isLinXu/pangu-ax)![img](https://badgen.net/badge/icon/learning?icon=deepscan&label)![GitHub repo size](https://img.shields.io/github/repo-size/isLinXu/pangu-ax.svg?style=flat-square) ![GitHub language count](https://img.shields.io/github/languages/count/isLinXu/pangu-ax)  ![GitHub last commit](https://img.shields.io/github/last-commit/isLinXu/pangu-ax) ![GitHub](https://img.shields.io/github/license/isLinXu/pangu-ax.svg?style=flat-square)![img](https://hits.dwyl.com/isLinXu/pangu-ax.svg)

---

## 介绍

### 项目介绍

### 效果展示

![test1](./figure/test1.png)

![test1](./figure/test2.png)



## 使用说明





### 环境配置



### ckpts

```
sudo ln -s /Users/gatilin/Pan/ckpts/ ckpts/pretrained
```

### 运行

#### pcl-pangu sdk推理

```shell
python pangu_infernce.py
```

![infer_demo1](./figure/infer_demo1.png)

```shell
python pangu_infernce.py -p "请简单介绍一下盘古" -m "2B6" -c "ckpts/pretrained/onnx_int8_pangu_alpha_2b6/" -b "onnx-cpu" -k 1 
```

![infer_demo2](./figure/infer_demo2.png)

#### gradio web启动

#### 文本推理

```shell
python web/run_gradio_web.py
```



#### chatbot对话机器人

```shell
python web/run_pangu_bot.py
```



### 测试数据



## 参考



