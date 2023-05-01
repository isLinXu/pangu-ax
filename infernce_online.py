

from pcl_pangu.online import Infer
model = "pangu-alpha-evolution-2B6-pt"
# prompt_input = "文本分类：\n基本上可以说是诈骗\n选项：积极，消极\n答案："
prompt_input = "四川的省会是?"
api_key = "2d9553d8374871a78200b2a18eccb7c2d39079fa"
result = Infer.generate(model, prompt_input, api_key)

print(result)

