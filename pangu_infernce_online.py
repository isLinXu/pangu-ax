from pcl_pangu.online import Infer


def infernce_online(model="pangu-alpha-evolution-2B6-pt",
                    prompt_input="四川的省会是?",
                    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"):
    result = Infer.generate(model, prompt_input, api_key)
    return result

if __name__ == '__main__':
    output = infernce_online(model="pangu-alpha-evolution-2B6-pt",
                    prompt_input="四川的省会是?",
                    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("output:", output)
