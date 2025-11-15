import json
from PIL import Image
from fire import Fire
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel
import httpx               # ← 新增
from typing import Optional, List, Any
from data_loading import convert_image_to_text, load_image
from google import genai
from google.genai import types


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    temperature: float = 0.0
    max_image_size: int = 1024

    def resize_image(self, image: Image) -> Image:
        h, w = image.size
        if h <= self.max_image_size and w <= self.max_image_size:
            return image

        factor = self.max_image_size / max(h, w)
        h = round(h * factor)
        w = round(w * factor)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize((h, w), Image.LANCZOS)
        return image

    def run(self, prompt: str, image: Image = None) -> dict:
        raise NotImplementedError


# import os
# from openai import OpenAI, AzureOpenAI

# client = OpenAI(
#     api_key = os.environ.get("ARK_API_KEY"),
#     base_url = "https://ark.cn-beijing.volces.com/api/v3",
# )

# # Non-streaming:
# print("----- standard request -----")
# completion = client.chat.completions.create(
#     model = "deepseek-r1-250120",  # your model endpoint ID
#     messages = [
#         {"role": "system", "content": "你是人工智能助手"},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
# )
# print(completion.choices[0].message.content)

# # Streaming:
# print("----- streaming request -----")
# stream = client.chat.completions.create(
#     model = "deepseek-r1-250120",  # your model endpoint ID
#     messages = [
#         {"role": "system", "content": "你是人工智能助手"},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
#     stream=True
# )

# for chunk in stream:
#     if not chunk.choices:
#         continue
#     print(chunk.choices[0].delta.content, end="")
# print()

class GPTModel(EvalModel):
    model_path: str = ""
    timeout: int = 150  # 默认超时时间300秒
    engine: str = ""
    client: Optional[OpenAI] = None
    max_retries: int = 1  # 最大重试次数
    max_tokens: int = 2048  # 默认最大token数

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["model"]
            self.client = OpenAI(
                api_key=info["api_key"],
                organization=info.get("organization", None),
                base_url=info.get("base_url", None),
                timeout=self.timeout,  # 设置API调用超时时间
            )
            # 从配置文件中读取参数（如果存在）
            self.temperature = info.get("temperature", self.temperature)
            self.max_tokens = info.get("max_tokens", self.max_tokens)

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        # 有图片 → lmdeploy 专用格式
        if image:
            if isinstance(image, str):
                image = load_image(image)
            img_base64 = convert_image_to_text(self.resize_image(image))
            # 不需要添加 data:image/jpeg;base64, 前缀
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "data": img_base64},  # lmdeploy 格式
                    {"type": "text", "text": prompt}
                ]
            }]
        # 无图片 → 纯文本字符串
        return [{"role": "user", "content": prompt}]

    def run(self, prompt: str, image: Image = None) -> dict:
        import time
        self.load()
        output = ""
        error_message = "The response was filtered"
        total_tokens = 0
        
        start_time = time.time()
        retry_count = 0

        while not output:
            # 检查是否超时
            if time.time() - start_time > self.timeout:
                return {"content": f"请求超时，已达到设定的{self.timeout}秒限制。任务终止。", "total_tokens": 0}
                
            # 检查重试次数
            if retry_count >= self.max_retries:
                return {"content": f"请求失败，已达到最大重试次数{self.max_retries}。任务终止。", "total_tokens": 0}
                
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.make_messages(prompt, image),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                
                # 获取内容
                output = response.choices[0].message.content or ""
                
                # 获取total_tokens
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    total_tokens = response.usage.completion_tokens
                
                # 简单地将响应内容保存到文件
                # self._save_output_to_file(output)
                
                # 打印token使用情况
                if hasattr(response, 'usage'):
                    print(f"\nToken使用情况: 输入={response.usage.prompt_tokens}, "
                          f"输出={response.usage.completion_tokens}, "
                          f"总计={response.usage.prompt_tokens}")

            except Exception as e:
                print(f"错误: {e}")
                retry_count += 1
                if error_message in str(e):
                    output = error_message
                # 短暂休眠后重试
                time.sleep(min(2 ** retry_count, 10))  # 指数退避策略，最大10秒

            if not output:
                print(f"OpenAI API请求失败，正在进行第{retry_count}次重试...")

        return {
            "content": output,
            "total_tokens": total_tokens
        }
    
    def _save_output_to_file(self, content):
        """将模型输出内容保存到文件"""
        import os
        from datetime import datetime
        
        # 创建输出目录
        output_dir = "model_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{self.engine}_{timestamp}.txt"
        
        # 保存内容到文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n输出已保存到文件: {filename}")


class DeepseekModel(GPTModel):
    model_path: str = "deepseek.json"
    
    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info.get("model", "deepseek-r1-250120")  # 默认使用deepseek-r1-250120
            self.client = OpenAI(
                api_key=info["api_key"],
                base_url=info.get("base_url", "https://ark.cn-beijing.volces.com/api/v3"),
                timeout=self.timeout,  # 设置API调用超时时间
            )
            # 直接从配置文件中读取参数
            self.temperature = info.get("temperature", self.temperature)
            self.max_tokens = info.get("max_tokens", self.max_tokens)
    
    def run(self, prompt: str, image: Image = None) -> dict:
        import time
        self.load()
        output = ""
        reasoning_content = ""
        reasoning_tokens = 0
        total_tokens = 0
        error_message = "The response was filtered"
        
        start_time = time.time()
        retry_count = 0

        while not output:
            # 检查是否超时
            if time.time() - start_time > self.timeout:
                return {
                    "content": f"请求超时，已达到设定的{self.timeout}秒限制。任务终止。",
                    "reasoning_content": "",
                    "reasoning_tokens": 0,
                    "total_tokens": 0
                }
                
            # 检查重试次数
            if retry_count >= self.max_retries:
                return {
                    "content": f"请求失败，已达到最大重试次数{self.max_retries}。任务终止。",
                    "reasoning_content": "",
                    "reasoning_tokens": 0,
                    "total_tokens": 0
                }
                
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=[{"role": "user", "content": prompt}],  # 直接构建消息数组
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                
                # 获取内容
                output = response.choices[0].message.content or ""
                
                # 获取reasoning_content（Deepseek模型特有）
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning_content = response.choices[0].message.reasoning_content or ""
                
                # 获取reasoning_tokens和total_tokens
                if hasattr(response, 'usage'):
                    total_tokens = response.usage.completion_tokens
                    
                    if hasattr(response.usage, 'completion_tokens_details'):
                        if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                
                # 保存完整的response对象到文件
                # self._save_output_to_file(output, response)
                
                # 打印token使用情况
                if hasattr(response, 'usage'):
                    print(f"\nToken使用情况: 输入={response.usage.prompt_tokens}, "
                          f"输出={response.usage.completion_tokens}, "
                          f"总计={response.usage.total_tokens}")
                    if reasoning_tokens > 0:
                        print(f"推理Token数: {reasoning_tokens}")

            except Exception as e:
                print(f"错误: {e}")
                retry_count += 1
                if error_message in str(e):
                    output = error_message
                # 短暂休眠后重试
                time.sleep(min(2 ** retry_count, 10))  # 指数退避策略，最大10秒

            if not output:
                print(f"OpenAI API请求失败，正在进行第{retry_count}次重试...")

        return {
            "content": output,
            "reasoning_content": reasoning_content,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens
        }

class GPT4oModel(GPTModel):
    model_path: str = "gpt4o.json"


class GPT4tModel(GPTModel):
    model_path: str = "gpt4t.json"


   
class GLM4Model(GPTModel):
    model_path: str = "glm4.json"


class O1Model(GPTModel):
    model_path: str = "o1-full-high.json"
    reasoning_effort: str = ""

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["model"]
            self.reasoning_effort = info["reasoning_effort"]
            self.client = OpenAI(
                api_key=info["api_key"],
                organization=info.get("organization", None),
                base_url=info.get("base_url", None),
                timeout=self.timeout,  # 设置API调用超时时间
            )

    def run(self, prompt: str, image: Image = None) -> dict:
        import time
        self.load()
        output = ""
        reasoning_tokens = 0
        total_tokens = 0
        error_message = "The response was filtered"
        
        start_time = time.time()
        retry_count = 0

        while not output:
            # 检查是否超时
            if time.time() - start_time > self.timeout:
                return {
                    "content": f"请求超时，已达到设定的{self.timeout}秒限制。任务终止。",
                    "reasoning_tokens": 0,
                    "total_tokens": 0
                }
                
            # 检查重试次数
            if retry_count >= self.max_retries:
                return {
                    "content": f"请求失败，已达到最大重试次数{self.max_retries}。任务终止。",
                    "reasoning_tokens": 0,
                    "total_tokens": 0
                }
                
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.make_messages(prompt, image),
                    reasoning_effort=self.reasoning_effort,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                
                # 获取内容
                output = response.choices[0].message.content or ""
                
                # O1模型: 获取reasoning_tokens和total_tokens
                if hasattr(response, 'usage'):
                    total_tokens = response.usage.completion_tokens
                    
                    # 从completion_tokens_details中获取reasoning_tokens
                    if hasattr(response.usage, 'completion_tokens_details'):
                        if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                
                # 保存完整的response对象到文件
                # self._save_output_to_file(output, response)
                
                # 打印token使用情况
                if hasattr(response, 'usage'):
                    print(f"\nToken使用情况: 输入={response.usage.prompt_tokens}, "
                          f"输出={response.usage.completion_tokens}, "
                          f"总计={response.usage.total_tokens}")
                    if reasoning_tokens > 0:
                        print(f"推理Token数: {reasoning_tokens}")

            except Exception as e:
                print(f"错误: {e}")
                retry_count += 1
                if error_message in str(e):
                    output = error_message
                # 短暂休眠后重试
                time.sleep(min(2 ** retry_count, 10))  # 指数退避策略，最大10秒

            if not output:
                print(f"OpenAI API请求失败，正在进行第{retry_count}次重试...")

        return {
            "content": output,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens
        }
    
    def _save_output_to_file(self, content, response=None):
        """将模型输出内容保存到文件"""
        import os
        import json
        from datetime import datetime
        
        # 创建输出目录
        output_dir = "model_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{self.engine}_{timestamp}.txt"
        
        # 保存内容到文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"输出内容已保存到文件: {filename}")
        
        # 保存完整的response对象
        if response:
            json_filename = f"{output_dir}/{self.engine}_{timestamp}_full_response.json"
            try:
                # 将response对象转换为字典
                response_dict = response.model_dump()
                
                # 保存为JSON文件
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(response_dict, f, ensure_ascii=False, indent=2)
                    
                print(f"完整的响应对象已保存到文件: {json_filename}")
            except Exception as e:
                print(f"保存完整响应对象时出错: {e}")
                
                # 尝试使用model_dump_json方法
                try:
                    json_content = response.model_dump_json(indent=2)
                    with open(json_filename, 'w', encoding='utf-8') as f:
                        f.write(json_content)
                    print(f"完整的响应对象已保存到文件: {json_filename}")
                except Exception as e2:
                    print(f"使用model_dump_json方法保存时出错: {e2}")
                    
                    # 最后尝试使用__dict__属性
                    try:
                        import inspect
                        response_attrs = {}
                        
                        # 记录对象的所有公共属性
                        for attr_name in dir(response):
                            if not attr_name.startswith('_'):  # 排除私有属性
                                try:
                                    attr_value = getattr(response, attr_name)
                                    # 排除方法和内置函数
                                    if not inspect.ismethod(attr_value) and not inspect.isbuiltin(attr_value):
                                        response_attrs[attr_name] = str(attr_value)
                                except Exception:
                                    response_attrs[attr_name] = "无法序列化的值"
                        
                        with open(json_filename, 'w', encoding='utf-8') as f:
                            json.dump(response_attrs, f, ensure_ascii=False, indent=2)
                            
                        print(f"响应对象的属性已保存到文件: {json_filename}")
                    except Exception as e3:
                        print(f"尝试所有方法保存响应对象均失败: {e3}")


class GeminiModel(EvalModel):
    model_path: str = "gemini.json"
    timeout: int = 300  # 默认超时时间300秒
    engine: str = ""
    client: Optional[Any] = None
    max_retries: int = 2  # 最大重试次数
    max_tokens: int = 8096  # 默认最大token数

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["model"]
            
            # 导入Gemini SDK
            from google import genai
            
            # 初始化Gemini客户端
            self.client = genai.Client(api_key=info["api_key"])
            
            # 从配置文件中读取参数（如果存在）
            self.temperature = info.get("temperature", self.temperature)
            self.max_tokens = info.get("max_output_tokens", self.max_tokens)

    def make_contents(self, prompt: str, image: Image = None) -> List:
        """将提示文本和图像（如果有）组合成Gemini API的contents格式"""
        contents = []
        
        # 添加图像
        if image:
            contents.append(image)
        
        # 添加文本提示
        contents.append(prompt)
        
        return contents

    def run(self, prompt: str, image: Image = None) -> dict:
        import time
        from google.genai import types
        
        self.load()
        output = ""
        error_message = "The response was filtered"
        total_tokens = 0
        
        # 如果image是字符串路径，则加载图像
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                return {"content": f"图像加载失败: {str(e)}", "total_tokens": 0}
        
        start_time = time.time()
        retry_count = 0
        last_status_time = start_time  # 上次状态输出时间

        while not output:
            # 检查是否超时
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 每30秒输出一次等待状态
            if current_time - last_status_time >= 30:
                print(f"当前等待响应时间: {int(elapsed_time)}秒...")
                last_status_time = current_time
            
            if elapsed_time > self.timeout:
                return {"content": f"请求超时，已达到设定的{self.timeout}秒限制。任务终止。", "total_tokens": 0}
                
            # 检查重试次数
            if retry_count >= self.max_retries:
                return {"content": f"请求失败，已达到最大重试次数{self.max_retries}。任务终止。", "total_tokens": 0}
                
            try:
                # 创建生成内容的配置
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                # 使用Gemini SDK生成内容
                response = self.client.models.generate_content(
                    model=self.engine,
                    contents=self.make_contents(prompt, image),
                    config=config
                )
                
                # 检查内容过滤
                if (hasattr(response, 'prompt_feedback') and 
                    response.prompt_feedback is not None and 
                    hasattr(response.prompt_feedback, 'block_reason') and 
                    response.prompt_feedback.block_reason):
                    raise ValueError(error_message)
                
                # 获取内容
                output = response.text or ""
                
                # 获取token使用情况
                if hasattr(response, 'usage_metadata'):
                    # Gemini API使用usage_metadata属性获取token统计
                    prompt_token_count = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    candidates_token_count = getattr(response.usage_metadata, 'candidates_token_count', 0)
                    total_tokens = getattr(response.usage_metadata, 'total_token_count', 0)
                    output_total_tokens = total_tokens - prompt_token_count
                    
                    # 打印token使用情况
                    print(f"\nToken使用情况: 输入={prompt_token_count}, "
                          f"输出={candidates_token_count}, "
                          f"总计={total_tokens}")

            except Exception as e:
                print(f"错误: {e}")
                retry_count += 1
                if error_message in str(e):
                    output = error_message
                # 短暂休眠后重试
                time.sleep(min(2 ** retry_count, 10))  # 指数退避策略，最大10秒
                print(f"Gemini API请求失败，正在进行第{retry_count}/{self.max_retries}次重试...")
                continue  # 立即进入下一次循环

            # 如果API返回了响应但内容为空，继续等待而不是立即重试
            if not output:
                print(f"Gemini API返回空响应，继续等待中...已等待{int(elapsed_time)}秒，总超时限制为{self.timeout}秒")
                # 较长休眠后继续尝试
                time.sleep(90)  # 每次空响应后等待90秒再次尝试
                continue

            break

        return {
            "content": output,
            "total_tokens": output_total_tokens
        }
    
    def _save_output_to_file(self, content):
        """将模型输出内容保存到文件"""
        import os
        from datetime import datetime
        
        # 创建输出目录
        output_dir = "model_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{self.engine}_{timestamp}.txt"
        
        # 保存内容到文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n输出已保存到文件: {filename}")
        
    def count_tokens(self, prompt: str, image: Image = None) -> int:
        """预估输入内容的token数量"""
        from google.genai import types
        
        self.load()
        
        try:
            # 使用Gemini SDK的count_tokens功能
            contents = self.make_contents(prompt, image)
            
            # 从客户端获取模型实例
            model = self.client.models.get_model(self.engine)
            
            # 计算token数量
            token_count = model.count_tokens(contents)
            
            return token_count.total_tokens
        except Exception as e:
            print(f"计算token时出错: {e}")
            return 0


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        o1=O1Model,
        gpt4o=GPT4oModel,
        gpt4t=GPT4tModel,
        glm4=GLM4Model,
        deepseek=DeepseekModel,
        gemini=GeminiModel,
        # 新增Azure模型支持
        azure_o4_mini=Azure_o4_miniModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    model_name: str = "gpt4o",
    prompt: str = "How to solve the wood_slide puzzle? give me some guide ",
    image_path: str = "D:\Projects\P3_CLER_Framework\Data_generation\evaluation\data\wood_slide\solution_198_easy.png",
    timeout: int = 300,  # 添加超时参数
    max_retries: int = 3,  # 添加最大重试次数参数
    **kwargs,
):
    model = select_model(model_name, timeout=timeout, max_retries=max_retries, **kwargs)
    # 先打印函数参数
    print("函数参数:", locals())
    
    # 手动调用load加载配置
    model.load()
    
    # 打印加载配置后的模型参数
    print(f"模型参数加载后: temperature={model.temperature}, max_tokens={model.max_tokens}, engine={model.engine}")
    
    result = model.run(prompt, image_path)
    # result = model.run(prompt)
    
    # 打印主要结果
    print(f"\n内容: {result['content']}")
    
    # 根据模型类型打印其他信息
    if model_name == "o1":
        print(f"\n推理Token数: {result.get('reasoning_tokens', 0)}")
    elif model_name == "deepseek":
        print(f"\n推理内容: {result.get('reasoning_content', '')}")
        print(f"\n推理Token数: {result.get('reasoning_tokens', 0)}")
    
    print(f"\n总Token数: {result.get('total_tokens', 0)}")
    
    return result


"""
python modeling.py test_model --model_name o1
python modeling.py test_model --model_name gpt4t
python modeling.py test_model --model_name gpt4v
python modeling.py test_model --model_name gpt4o
python modeling.py test_model --model_name deepseek
python modeling.py test_model --model_name gemini
python modeling.py test_model --model_name azure_o4_mini
"""


class AzureGPTModel(GPTModel):
    """
    使用Azure OpenAI API的模型类
    相比普通OpenAI API，Azure API需要额外的api_version和azure_endpoint参数
    """
    model_path: str = ""
    api_version: str = "2025-01-01-preview"  # 更新为最新预览版API

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["model"]
            
            # 使用AzureOpenAI客户端替代OpenAI
            self.client = AzureOpenAI(
                api_key=info["api_key"],
                api_version=info.get("api_version", self.api_version),
                azure_endpoint=info["azure_endpoint"],  # Azure OpenAI需要此参数
                timeout=self.timeout,
            )
            
            # 从配置文件中读取参数（如果存在）
            self.temperature = info.get("temperature", self.temperature)
            self.max_tokens = info.get("max_tokens", self.max_tokens)
            
            print(f"Azure配置: endpoint={info['azure_endpoint']}, api_version={info.get('api_version', self.api_version)}, deployment={self.engine}")

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        """
        根据Azure OpenAI API要求构建消息格式
        图片格式：{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xxx"}}
        """
        inputs = []
        
        # 处理图片输入
        if image:
            if isinstance(image, str):
                image = load_image(image)
            img_base64 = convert_image_to_text(self.resize_image(image))
            # Azure需要添加data:image/jpeg;base64前缀
            url = f"data:image/jpeg;base64,{img_base64}"
            inputs.append(
                {"type": "image_url", "image_url": {"url": url}}
            )
        
        # 添加文本输入
        inputs.append({"type": "text", "text": prompt})
        
        # 返回完整消息格式 - 使用user角色而非developer
        return [{"role": "user", "content": inputs}]
        
    def run(self, prompt: str, image: Image = None) -> dict:
        """
        使用Azure OpenAI API运行模型，特别针对Azure API格式优化
        """
        import time
        self.load()
        output = ""
        error_message = "The response was filtered"
        total_tokens = 0
        
        start_time = time.time()
        retry_count = 0
        
        print(f"模型参数加载后: temperature={self.temperature}, max_tokens={self.max_tokens}, engine={self.engine}")

        while not output:
            # 检查是否超时
            if time.time() - start_time > self.timeout:
                return {"content": f"请求超时，已达到设定的{self.timeout}秒限制。任务终止。", "total_tokens": 0}
                
            # 检查重试次数
            if retry_count >= self.max_retries:
                return {"content": f"请求失败，已达到最大重试次数{self.max_retries}。任务终止。", "total_tokens": 0}
                
            try:
                # 准备API调用参数
                api_params = {
                    "model": self.engine,  # 部署名称
                    "messages": self.make_messages(prompt, image),
                }
                
                # o4-mini模型不支持temperature参数，只能使用默认值1
                if "o4-mini" not in self.engine:
                    api_params["temperature"] = self.temperature
                
                # 根据官方示例，使用max_completion_tokens而非max_tokens
                if hasattr(self, "max_tokens") and self.max_tokens:
                    api_params["max_completion_tokens"] = self.max_tokens
                
                # 打印实际API调用参数
                print(f"Azure API调用参数: {api_params}")
                
                # 调用API
                response = self.client.chat.completions.create(**api_params)
                
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                
                # 获取内容
                output = response.choices[0].message.content or ""
                
                # 获取token统计
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    total_tokens = response.usage.completion_tokens
                
                # 打印token使用情况
                if hasattr(response, 'usage'):
                    print(f"\nToken使用情况: 输入={response.usage.prompt_tokens}, "
                          f"输出={response.usage.completion_tokens}, "
                          f"总计={response.usage.total_tokens}")

            except Exception as e:
                print(f"错误: {e}")
                retry_count += 1
                if error_message in str(e):
                    output = error_message
                # 短暂休眠后重试
                time.sleep(min(2 ** retry_count, 10))  # 指数退避策略，最大10秒

            if not output:
                print(f"Azure OpenAI API请求失败，正在进行第{retry_count}次重试...")

        return {
            "content": output,
            "total_tokens": total_tokens
        }


class Azure_o4_miniModel(AzureGPTModel):
    """
    使用Azure GPT-4o模型的简便类
    """
    model_path: str = "azure_o4_mini.json"




if __name__ == "__main__":
    Fire()
