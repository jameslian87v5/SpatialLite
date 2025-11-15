import json
from PIL import Image
from fire import Fire
from openai import OpenAI
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
# from openai import OpenAI

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
            
            # 添加明确的headers，提高兼容性
            default_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            self.client = OpenAI(
                api_key=info["api_key"],
                organization=info.get("organization", None),
                base_url=info.get("base_url", None),
                timeout=self.timeout,  # 设置API调用超时时间
                default_headers=default_headers,
                http_client=httpx.Client(
                    http2=False,
                    http1=True,
                    timeout=self.timeout
                ),
            )
            # 从配置文件中读取参数（如果存在）
            self.temperature = info.get("temperature", self.temperature)
            self.max_tokens = info.get("max_tokens", self.max_tokens)

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        # 根据新错误：KeyError: 'type'，完全调整为OpenAI格式
        if image:
            if isinstance(image, str):
                image = load_image(image)
            img_base64 = convert_image_to_text(self.resize_image(image))
            
            # LMDeploy要求type字段必须存在，并且图片type应为'image_url'或'image_data'
            # 添加data:image/jpeg;base64,前缀因为OpenAI格式需要
            return [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},  # 添加type字段
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_base64}}  # 使用标准OpenAI格式
                ]
            }]
        # 纯文本格式 - 已测试可以工作
        return [{"role": "user", "content": prompt}]

    def run(self, prompt: str, image: Image = None) -> dict:
        import requests
        import time
        
        # 仍然加载配置以获取引擎、温度等参数
        self.load()
        output = ""
        error_message = "The response was filtered"
        total_tokens = 0
        
        # 从OpenAI客户端中提取base_url，先转换为字符串
        base_url = str(self.client.base_url).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        
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
                # 使用make_messages来构造一致的消息格式
                messages = self.make_messages(prompt, image)
                
                # 构造请求payload
                payload = {
                    "model": self.engine,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                
                # 设置请求头
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                # 使用requests发送请求
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # 检查HTTP状态码
                response.raise_for_status()
                
                # 解析响应
                data = response.json()
                
                # 检查内容过滤
                if data["choices"][0].get("finish_reason") == "content_filter":
                    raise ValueError(error_message)
                
                # 获取内容
                output = data["choices"][0]["message"]["content"] or ""
                
                # 获取total_tokens
                if "usage" in data and "total_tokens" in data["usage"]:
                    total_tokens = data["usage"].get("completion_tokens", data["usage"]["total_tokens"])
                
                # 打印token使用情况
                if "usage" in data:
                    print(f"\nToken使用情况: 输入={data['usage'].get('prompt_tokens', 0)}, "
                          f"输出={data['usage'].get('completion_tokens', 0)}, "
                          f"总计={data['usage'].get('total_tokens', 0)}")

            except requests.exceptions.Timeout:
                print("错误: 请求超时")
                retry_count += 1
                time.sleep(min(2 ** retry_count, 10))
            except requests.exceptions.ConnectionError:
                print("错误: 连接错误")
                retry_count += 1
                time.sleep(min(2 ** retry_count, 10))
            except requests.exceptions.HTTPError as e:
                print(f"错误: HTTP错误 - {e}")
                retry_count += 1
                time.sleep(min(2 ** retry_count, 10))
            except Exception as e:
                print(f"错误: {e}")
                retry_count += 1
                if error_message in str(e):
                    output = error_message
                time.sleep(min(2 ** retry_count, 10))  # 指数退避策略，最大10秒

            if not output:
                print(f"API请求失败，正在进行第{retry_count}次重试...")

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
    max_retries: int = 1  # 最大重试次数
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
            "total_tokens": candidates_token_count
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
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(model_name, prompt=None, image_path=None, timeout=300, max_retries=3, **kwargs):
    print("函数参数:", locals())
    
    # 设置默认提示
    if prompt is None:
        prompt = "How to solve the wood_slide puzzle? give me some guide "

    # 加载指定模型
    model = select_model(model_name, timeout=timeout, max_retries=max_retries, **kwargs)
    
    print(f"模型参数加载后: temperature={model.temperature}, max_tokens={model.max_tokens}, engine={model.engine}")
    
    # 恢复图片处理功能，使用修改后的多模态格式
    # 默认图片路径
    if image_path is None and prompt.strip().lower().find("wood_slide") >= 0:
        image_path = "D:\\Projects\\P3_CLER_Framework\\Data_generation\\evaluation\\data\\wood_slide\\solution_198_easy.png"
        print(f"使用默认图片路径: {image_path}")
        
    result = model.run(prompt, image_path)  # 恢复传递图片路径
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
"""


if __name__ == "__main__":
    Fire()
