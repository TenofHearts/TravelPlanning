from openai import OpenAI
import warnings
from json_repair import repair_json

class model_GPT:
    def __init__(self, model_name="chatgpt-4o-latest"):
        self.llm = OpenAI(
            api_key="sk-CaNxBZNilnFQDexn7eCd8c36Db714b9795D89f7f307a6316",
            base_url="https://api.lqqq.cc/v1",
        )

    def __call__(self, messages):
        try:
            res_str = self.llm.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=messages,
                max_tokens=4095,
                temperature=0,
                top_p=0.01,
                stop=['\n'],
            ).choices[0].message.content
        except Exception as e:
            res_str = "调用失败，错误信息：" + str(e)
        res_str = res_str.split("\n")[0]
        res_str = res_str.strip()
        return res_str

class model_GPT_poi:
    def __init__(self, model_name="chatgpt-4o-latest"):
        self.llm = OpenAI(
            api_key="sk-CaNxBZNilnFQDexn7eCd8c36Db714b9795D89f7f307a6316",
            base_url="https://api.lqqq.cc/v1",
        )

    def __call__(self, messages):
        try:
            res_str = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=4095,
                # stop=['\n'],
                temperature=0,
                top_p=0.01,
            ).choices[0].message.content
        except Exception as e:
            res_str = "调用失败，错误信息：" + str(e)
        res_str = res_str.strip()
        return res_str

class model_GPT_json:
    def __init__(self, model_name="chatgpt-4o-latest"):
        self.llm = OpenAI(
            api_key="sk-CaNxBZNilnFQDexn7eCd8c36Db714b9795D89f7f307a6316",
            base_url="https://api.lqqq.cc/v1",
        )

    def __call__(self, messages):
        try:
            res_str = self.llm.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=messages,
                max_tokens=4095,
                temperature=0,
                top_p=0.01,
                # stop=['\n'],
                response_format={
                    'type': 'json_object'
                }
            ).choices[0].message.content
        except Exception as e:
            res_str = "调用失败，错误信息：" + str(e)
        res_str = res_str.replace("```json", "")
        res_str = res_str.replace("```", "")
        # res_str = res_str.replace("'", '"')
        res_str = res_str.strip()
        res_str = repair_json(res_str, ensure_ascii=False)
        return res_str

class model_glm4:
    def __init__(self):
        self.llm = OpenAI(
            api_key="9c4c7ab24e2b4f0d9b9c621e7d74d805.92p8egnHGVlhkvX9",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )

    def __call__(self, messages):
        try:
            res_str = self.llm.chat.completions.create(
                model="glm-4-plus",
                messages=messages,
                max_tokens=4095,
                temperature=0,
                top_p=0.01,
                stop=['<STOP>'],
            ).choices[0].message.content
        except Exception as e:
            res_str = "调用失败，错误信息：" + str(e)
        res_str = res_str.strip()
        return res_str

class model_glm4_poi:
    def __init__(self):
        self.llm = OpenAI(
            api_key="9c4c7ab24e2b4f0d9b9c621e7d74d805.92p8egnHGVlhkvX9",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )

    def __call__(self, messages):
        try:
            res_str = self.llm.chat.completions.create(
                model="glm-4-plus",
                messages=messages,
                max_tokens=4095,
                temperature=0,
                top_p=0.01,
                stop=['<STOP>'],
            ).choices[0].message.content
        except Exception as e:
            res_str = "调用失败，错误信息：" + str(e)
        res_str = res_str.strip()
        return res_str

class model_glm4_json:
    def __init__(self):
        self.llm = OpenAI(
            api_key="9c4c7ab24e2b4f0d9b9c621e7d74d805.92p8egnHGVlhkvX9",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )

    def __call__(self, messages):
        try:
            res_str = self.llm.chat.completions.create(
                model="glm-4-plus",
                messages=messages,
                max_tokens=4095,
                temperature=0,
                top_p=0.01,
            ).choices[0].message.content
        except Exception as e:
            res_str = "调用失败，错误信息：" + str(e)
        res_str = res_str.replace("```json", "")
        res_str = res_str.replace("```", "")
        # res_str = res_str.replace("'", '"')
        res_str = res_str.strip()
        res_str = repair_json(res_str, ensure_ascii=False)
        return res_str

class deepseek:
    def __init__(self):
        self.llm = OpenAI(
            base_url='https://api.deepseek.com/beta',
            # api_key='sk-500b3107741d49c38f2a945082d04598'
            api_key='sk-66ed89bddd664c8eb4c31cfd2b0efb58'
        )

    def __call__(self, messages):
        try:
            messages[-1]["prefix"] = True
            res_str = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=8192,
                stop=['\n'],
                temperature=0,
                top_p=0.00000001,
            ).choices[0].message.content
        except Exception as e:
            res_str = "调用失败，错误信息：" + str(e)
        res_str = res_str.split("\n")[0]
        res_str = res_str.strip()
        return res_str


class deepseek_poi:
    def __init__(self):
        self.llm = OpenAI(
            base_url='https://api.deepseek.com/beta',
            api_key='sk-66ed89bddd664c8eb4c31cfd2b0efb58',
            timeout=60.0  # 设置60秒超时
        )

    def __call__(self, messages, max_retries=3):
        for attempt in range(max_retries):
            try:
                messages[-1]["prefix"] = True
                res_str = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    max_tokens=4096,  # 减少token数量
                    # stop=['\n'],
                    temperature=0,
                    top_p=0.00000001,
                ).choices[0].message.content
                # res_str = res_str.split("\n")[0]
                res_str = res_str.strip()
                return res_str
            except Exception as e:
                error_msg = str(e)
                print(f"第 {attempt + 1} 次调用失败: {error_msg}")
                
                if attempt == max_retries - 1:  # 最后一次尝试
                    res_str = "调用失败，错误信息：" + error_msg
                    return res_str
                
                # 如果是超时错误，等待一下再重试
                if "timeout" in error_msg.lower():
                    import time
                    time.sleep(2)  # 等待2秒后重试
                    continue


class deepseek_json:
    def __init__(self):
        self.llm = OpenAI(
            base_url='https://api.deepseek.com',
            # api_key='sk-66ed89bddd664c8eb4c31cfd2b0efb58',
            api_key='sk-66ed89bddd664c8eb4c31cfd2b0efb58',
            timeout=60.0  # 设置60秒超时
        )

    def __call__(self, messages, max_retries=3):
        for attempt in range(max_retries):
            try:
                res_str = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0,
                    top_p=0.01,
                    max_tokens=4096,  # 减少token数量，加快响应
                    response_format={
                        'type': 'json_object'
                    }
                ).choices[0].message.content
                res_str = res_str.strip()
                # res_str = res_str.replace("'", '"')
                res_str = repair_json(res_str, ensure_ascii=False)
                return res_str
            except Exception as e:
                error_msg = str(e)
                print(f"第 {attempt + 1} 次调用失败: {error_msg}")
                
                if attempt == max_retries - 1:  # 最后一次尝试
                    res_str = "调用失败，错误信息：" + error_msg
                    return res_str
                
                # 如果是超时错误，等待一下再重试
                if "timeout" in error_msg.lower():
                    import time
                    time.sleep(2)  # 等待2秒后重试
                    continue


class human:
    def __init__(self):
        pass

    def __call__(self, prompt):
        with open("tmp_prompt.txt", "w", encoding="utf-8") as f:
            for i in prompt:
                f.write(str(i))
        print(prompt[-1]["content"], end="")
        res = input()
        return res


def test():
    # model = deepseek_json()
    # from test_prompts import DIRECT_PROMPT
    # res = model([{"role": "user", "content": DIRECT_PROMPT}, {
    #             "role": "user", "content": "请安排一下南京的行程"}])
    # print(res)
    model = deepseek()
    res = model([{"role": "user", "content": "你好"}])
    print(res)


if __name__ == "__main__":
    test()
