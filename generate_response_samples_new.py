from pathlib import Path
import string
import re
import os
import random

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_loading import Data, convert_text_to_image
from modeling import select_model
from prompting import BasePrompter
from typing import List, Dict, Any, Optional, ClassVar
import json

class Executor:
    def __init__(self, execution_plan, sample):
        self.actions = execution_plan
        self.sample = sample
        self.env = None

    def _create_environment(self):
        """创建交互式环境。"""
        # 检查样本是否包含环境类型和问题类型
        if not hasattr(self.sample, 'environment_type') or not self.sample.environment_type:
            raise ValueError("样本缺少环境类型")
        
        if not hasattr(self.sample, 'question_type') or self.sample.question_type != "executive":
            return "这是一个描述性问题，不需要创建环境。", False
            
        # 获取环境类型和初始状态
        environment_type = self.sample.environment_type
        
        if not hasattr(self.sample, 'initial_state') or not self.sample.initial_state:
            raise ValueError("样本缺少初始状态")
        
        initial_state = self.sample.initial_state
        
        # 根据环境类型创建相应的环境
        if environment_type == "moving_box":
            # 导入并实例化推箱子环境
            from environment import BoxEnv
            self.env = BoxEnv(initial_state)
        elif environment_type == "wood_slide":
            # 导入并实例化华容道环境
            from environment import WoodSlideEnv
            # 检查样本是否包含最终状态
            if not hasattr(self.sample, 'final_state') or not self.sample.final_state:
                raise ValueError("样本缺少最终状态")
            
            final_state = self.sample.final_state
            self.env = WoodSlideEnv(initial_state, final_state)
        elif environment_type == "number_slide":
            # 导入并实例化数字华容道环境
            from environment import NumberSlideEnv
            self.env = NumberSlideEnv(initial_state)
        else:
            raise ValueError(f"不支持的环境类型: {environment_type}")
        
        return "环境创建成功", True

    def run(self) -> dict:
        """执行解决方案中的所有动作"""
        try:
            # 创建环境
            message, success = self._create_environment()            
            # 如果是描述性问题或创建环境失败，返回相应信息
            if not success:
                # 返回特定的错误类型和消息，以便在评估时区分
                return {
                    "success": False,
                    "error_type": "environment_creation_failed",
                    "message": message
                }
            
            # 逐步执行动作
            executed_actions = []
            failed_action_idx = -1
            failure_reason = ""
            
            for i, action in enumerate(self.actions):
                # 跳过空行和注释
                if not action or action.startswith("#"):
                    continue
                
                try:
                    # 执行前验证动作格式
                    if not hasattr(self.env, 'validate_action'):
                        # 如果环境没有提供验证方法，直接尝试执行
                        result, message = self.env.step(action)
                    else:
                        # 先验证动作格式
                        valid, validate_msg = self.env.validate_action(action)
                        if not valid:
                            failed_action_idx = i
                            failure_reason = f"动作格式无效: {validate_msg}"
                            # 记录执行历史并返回失败信息
                            return {
                                "success": False,
                                "error_type": "invalid_action",
                                "message": failure_reason,
                                "action_index": failed_action_idx,
                                "action": action,
                                "executed_actions": executed_actions,
                                "steps": self.env.steps if hasattr(self.env, 'steps') else executed_actions,
                                "final_state": self.env.get_state()
                            }
                        
                        # 执行有效动作
                        result, message = self.env.step(action)
                    
                    # 记录已执行的动作
                    executed_actions.append(action)
                    
                    # 如果动作执行失败，记录失败信息
                    if not result:
                        failed_action_idx = i
                        failure_reason = message
                        return {
                            "success": False,
                            "error_type": "execution_failed",
                            "message": f"动作执行失败: {message}",
                            "action_index": failed_action_idx,
                            "action": action,
                            "executed_actions": executed_actions,
                            "steps": self.env.steps if hasattr(self.env, 'steps') else executed_actions,
                            "final_state": self.env.get_state()
                        }
                    
                    # 检查是否达到终止状态
                    if self.env.is_terminal():
                        return {
                            "success": True,
                            "message": "任务完成！所有箱子都已移动到目标位置。",
                            "steps": self.env.steps if hasattr(self.env, 'steps') else executed_actions,
                            "executed_actions": executed_actions,
                            "final_state": self.env.get_state()
                        }
                
                except Exception as action_error:
                    # 记录具体动作执行错误
                    failed_action_idx = i
                    failure_reason = str(action_error)
                    return {
                        "success": False,
                        "error_type": "action_exception",
                        "message": f"执行动作 '{action}' 时出错: {failure_reason}",
                        "action_index": failed_action_idx,
                        "action": action,
                        "executed_actions": executed_actions,
                        "steps": self.env.steps if hasattr(self.env, 'steps') else executed_actions,
                        "final_state": self.env.get_state() if hasattr(self.env, 'get_state') else None
                    }
            
            # 如果执行完所有动作后仍未达到终止状态
            return {
                "success": False,
                "error_type": "incomplete_solution",
                "message": "执行完所有动作后，仍有箱子未到达目标位置。",
                "steps": self.env.steps if hasattr(self.env, 'steps') else executed_actions,
                "executed_actions": executed_actions,
                "final_state": self.env.get_state()
            }
            
        except Exception as e:
            # 其他运行时错误
            return {
                "success": False,
                "error_type": "runtime_error",
                "message": f"执行过程中出错: {str(e)}"
            }


class Scorer(BaseModel):
    def run(self, sample) -> float:
        raise NotImplementedError

class FailureAnalyzer(Scorer):
    model: ClassVar = select_model("azure_o4_mini")

    def run(self, sample) -> str:

        # 检查prompt是否存在或为空
        if not hasattr(sample, 'prompt') or sample.prompt == '':
            print(f"样本无效：缺少问题内容")
            return -1.0  # 表示样本无效
        
        # 检查raw_output是否存在或为空
        if not hasattr(sample, 'raw_output') or sample.raw_output == '':
            print(f"样本无效：缺少回答内容")
            return -1.0  # 表示样本无效

        # 检查score是否存在，我们只需要分析失败的样本
        if not hasattr(sample, 'score') or sample.score == None:
            print(f"样本尚未评分，无法分析失败原因")
            return "样本尚未评分，无法分析失败原因"
        
        # 如果样本不是失败的样本(score不等于0)，不需要分析
        if sample.score != -2.0:
            return "样本非失败样本，不需要分析"

        input_prompt = f"""
        这是一个问答记录，请帮我分析其失败的原因，并将失败原因分类为以下三种类型之一：

        1. Perception-Level Error: 无法准确检测对象或表示空间布局和拓扑关系。无法准确地理解和误导性地识别对象的距离或方位，主要是识别错误。

        2. Transformation-Level Error: 空间转换建模或预测错误，例如立方体翻滚和魔方中的面映射、心理旋转中的视角变化，或移动框和木滑块中的基本运动约束。比如不能穿墙但是却方案里面穿墙，或者布局预测错误，比如人 箱子 木块的移动之后的布局预测错误，核心是他们对游戏的机制理解错误，或者是由于长期序列跟踪错误，其中推理链在扩展动作序列上崩溃。

        3. Strategic-Level Error: 有缺陷的策略导致过长、零散或不必要的复杂推理链，使模型陷入死胡同或循环状态，或者使用的方案策略无法执行。

        请详细分析以下问答，给出具体的失败原因，并明确指出属于哪一类错误。
        
        问题：{sample.prompt}
        回答：{sample.raw_output}

        请用以下格式回答：
        - 错误类型：（选择以上三种类型之一）
        - 详细分析：（详细解释为什么回答是错误的，错误在哪里）
        """

        # 运行模型获取结果（字典形式）
        result = self.model.run(input_prompt)
        
        # 从结果中获取内容
        output = result["content"] if isinstance(result, dict) else result
        
        # 如果输出为空，返回默认值
        if not output:
            return "无法分析失败原因"
        
        return output
            
    
        

class LLMScorer(Scorer):
    model: ClassVar = select_model("azure_o4_mini")

    def run(self, sample) -> float:
        # 检查pred是否存在或为空
        if not hasattr(sample, 'pred') or sample.pred == '':
            print(f"样本无效：缺少预测结果")
            return -1.0  # 表示样本无效
        
        # 检查answer是否存在或为空
        if not hasattr(sample, 'answer') or sample.answer == '':
            print(f"样本无效：缺少正确答案")
            return -1.0  # 表示样本无效
            
        input_prompt = f"""
        You are a highly skilled evaluator. Your task is to determine whether the answer provided correctly solves the task or question, the correct answer is provided，so just justify the model_answer if consistent with the correct answer or not .
        
        Original_Question: {sample.question}
        Correct_Answer: {sample.answer}
        Model_Answer: {sample.pred}
        
        The answer is either completely right (1) or wrong (0). Provide your evaluation as a single number (1 or 0).
        """
        
        # 运行模型获取结果（字典形式）
        result = self.model.run(input_prompt)
        
        # 从结果中获取内容
        output = result["content"] if isinstance(result, dict) else result
        
        # 如果输出为空，返回默认值0
        if not output:
            return 0
            
        # 尝试从输出中提取得分（0或1）
        try:
            score = float(output)
            if score == 1:
                return 1.0
            elif score == 0:
                return 0.0
            else:
                return 0.0
        except ValueError:
            return 0.0



class ExcutionScorer(Scorer):

    model: ClassVar = select_model("azure_o4_mini")
    def parse_actions_sequence (self, sample) -> list:
        # 如果是wood_slide环境类型且缺少执行格式相关字段，则添加默认值
        if sample.environment_type == "wood_slide":
            # 检查并设置默认的execution_format（如果不存在）
            if not hasattr(sample, 'execution_format') or not sample.execution_format:
                sample.execution_format = "The action sequence is a series of two-part actions for wood blocks, like ['3u', '4r', '6d']. The first part is the wood block number, and the second part is the move direction: 'u' up, 'd' down, 'l' left, 'r' right. Each action moves one wood block one unit distance. So the action sequence should be a list of two-part actions. So the action sequence ['3u', '4r', '6d'] means block 3 moves up, block 4 moves right, block 6 moves down."
            
            # 检查并设置默认的execution_example（如果不存在）
            if not hasattr(sample, 'execution_example') or not sample.execution_example:
                sample.execution_example = "['3u', '4r', '6d']"
        
        input_prompt = f"""
        You will receive the model's spatial reasoning execution solution for a specific question, and you need to pasrse the model's output plan and break it down into specific operations. Because the execution plan required for spatial reasoning tasks such as moving_box or number_slide game is not unique, you need to interprete the model's output plan based on the current solution and execute it in the environment, and then evaluate the answer. All you need to do is to break down the model's output plan into specific executable actions fromat based on the current situation.
    Please strictly follow the execution action format, and output the foramtted excution solution plan, the format is a list, no more other words. 
    quetion_type:{sample.question_type}
    candidate_solution: {sample.raw_output}
    execution_format: {sample.execution_format if hasattr(sample, 'execution_format') else ""}
    execution_plan_example: {sample.execution_example if hasattr(sample, 'execution_example') else ""}
    formatted_execution_plan:    
    """.strip()
        
        # 运行模型获取结果（字典形式）
        result = self.model.run(input_prompt)
        
        # 从结果中获取内容
        execution_output = result["content"] if isinstance(result, dict) else result
        
        # 如果执行输出为空，直接返回空列表
        if not execution_output:
            return []
            
        # 清理模型输出，删除空白和换行
        cleaned_output = execution_output.strip()
        
        # 将清理后的输出保存到sample对象中
        sample.raw_parse_actions = cleaned_output
        
        # 如果sample环境类型为moving_box
        if sample.environment_type == "moving_box":
            # 为推箱子环境定义动作映射表，将不同表示形式映射到标准动作
            action_mapping = {
                'u': 'u', 'up': 'u', 'U': 'u', 'UP': 'u', 'Up': 'u',
                'd': 'd', 'down': 'd', 'D': 'd', 'DOWN': 'd', 'Down': 'd',
                'l': 'l', 'left': 'l', 'L': 'l', 'LEFT': 'l', 'Left': 'l',
                'r': 'r', 'right': 'r', 'R': 'r', 'RIGHT': 'r', 'Right': 'r'
            }
            
            # 从输出中提取动作
            actions = []
            
            # 尝试从文本中提取动作词
            # 匹配单个字符(u,d,l,r及其大写)或完整单词(up,down,left,right及其大小写混合)
            action_pattern = r'\b([udlrUDLR]|up|down|left|right|UP|DOWN|LEFT|RIGHT|Up|Down|Left|Right)\b'
            matches = re.findall(action_pattern, cleaned_output)
            
            # 转换匹配到的动作为标准格式
            for match in matches:
                if match.lower() in action_mapping:
                    actions.append(action_mapping[match.lower()])
                elif match in action_mapping:
                    actions.append(action_mapping[match])
            
            # 如果找到至少一个动作，返回动作序列
            if actions:
                return actions
            
            # 回退方案1：如果上面的方法没找到动作，尝试直接提取u,d,l,r字符
            valid_actions = ['u', 'd', 'l', 'r', 'U', 'D', 'L', 'R']
            fallback_actions = []
            for char in cleaned_output:
                if char in valid_actions:
                    # 转换为小写标准格式
                    fallback_actions.append(char.lower())
            
            if fallback_actions:
                return fallback_actions
                
            # 如果仍然没找到动作，返回空列表
            return []

        if sample.environment_type == "wood_slide":
            # 尝试检测模型输出是否是有效的动作序列
            # 对于wood_slide环境，直接提取序列括号内的内容
            if '[' in cleaned_output and ']' in cleaned_output:
                # 提取括号中的内容
                bracket_content = cleaned_output[cleaned_output.find('[')+1:cleaned_output.rfind(']')]
                
                # 直接按逗号分割提取元素序列
                actions = []
                for item in re.split(r',', bracket_content):
                    # 去除空白字符和可能的引号
                    clean_item = item.strip().strip('\'"')
                    if clean_item:
                        actions.append(clean_item)
                
                if actions:
                    return actions
            
            return []

        else:
            pass
        
            
    def run(self, sample) -> float:
        """运行评估并返回得分"""
        # 检查样本中是否存在预测结果
        if not hasattr(sample, 'pred') or not sample.pred:
            return -1.0  # 无效样本，缺少预测结果
        
        # 检查样本中是否已有格式化的执行计划
        if hasattr(sample, 'formatted_execution_plan') and sample.formatted_execution_plan:
            # 如果已有执行计划，则直接使用
            execution_plan = sample.formatted_execution_plan
        else:
            # 否则解析模型输出
            execution_plan = self.parse_actions_sequence(sample)
            sample.formatted_execution_plan = execution_plan
        
        # 检查解析结果
        if isinstance(execution_plan, str) and (execution_plan.startswith("格式错误") or execution_plan.startswith("解析错误")):
            # 解析失败，返回-2分（表示解析问题）
            return -2.0
            
        # 如果解析结果为空列表，表示无法提取有效动作序列
        if isinstance(execution_plan, list) and len(execution_plan) == 0:
            # 返回-2分（表示解析问题）
            return -2.0

        # 创建执行器并运行
        executor = Executor(execution_plan, sample)
        
        result = executor.run()
        # 根据执行结果评分
        if result.get("success", False):
            # 执行成功，返回1分
            return 1.0
        else:
            # 检查错误类型
            error_type = result.get("error_type", "")
            if error_type == "environment_creation_failed":
                # 环境创建失败，返回-1分（表示样本问题）
                print(f"环境创建失败: {result.get('message', '')}")
                return -1.0
            elif error_type == "invalid_action":
                # 动作格式无效，返回-2分（表示解析问题）
                print(f"动作格式无效: {result.get('message', '')}, 第{result.get('action_index', -1)}个动作")
                return -2.0
            elif error_type == "action_exception":
                # 动作执行异常，返回0分（表示执行错误）
                print(f"动作执行异常: {result.get('message', '')}, 第{result.get('action_index', -1)}个动作")
                return 0.0
            elif error_type == "execution_failed":
                # 动作执行失败，返回0分（表示执行错误）
                print(f"动作执行失败: {result.get('message', '')}, 第{result.get('action_index', -1)}个动作")
                return 0.0
            elif error_type == "incomplete_solution":
                # 解决方案不完整，返回0分（表示执行错误）
                print(f"解决方案不完整: {result.get('message', '')}")
                return 0.0
            elif error_type == "runtime_error":
                # 运行时错误，返回-2分（表示系统问题）
                print(f"运行时错误: {result.get('message', '')}")
                return -2.0
            else:
                # 其他未知错误，返回0分
                print(f"其他执行错误: {result.get('message', '')}")
                return 0.0

def generate_answer(sample, model, question_type, cot=False, use_image=True):
    """
    为样本生成答案
    
    Args:
        sample: 数据样本
        model: 使用的模型
        question_type: 问题类型 ('descrip'或'executive')
        cot: 是否使用Chain of Thought
        use_image: 是否使用图像，默认为True
    """
    # 寻找prompter 初始化
    prompter = BasePrompter(question_type=question_type, cot=cot)
    # 初始提示生成
    sample.prompt = prompter.run(sample)
    print(f"sample.prompt: {sample.prompt}")
    # 根据use_image参数决定是否使用图像
    image = convert_text_to_image(sample.image_string) if use_image else None
    
    # 运行模型获取结果（字典形式）
    result = model.run(sample.prompt, image)
    
    # 将结果分配给sample对象的相应字段
    sample.raw_output = result["content"]
    sample.total_tokens = result.get("total_tokens", 0)
    sample.reasoning_tokens = result.get("reasoning_tokens", 0)
    sample.reasoning_content = result.get("reasoning_content", "")
    
    print(f"sample.raw_output: {sample.raw_output}")

    if question_type == "descrip":
        # 描述性问题：提取答案
        sample.pred = prompter.get_answer(sample.raw_output)
        
        # 如果需要二次提示（根据实际需求判断）
        if need_refinement(sample.pred):
            sample.prompt = prompter.run(sample, is_second_prompt=True)
            
            # 运行模型获取结果（字典形式）
            result = model.run(sample.prompt, image)
            
            # 将结果分配给sample对象的相应字段
            sample.raw_output = result["content"]
            sample.total_tokens = result.get("total_tokens", 0)
            sample.reasoning_tokens = result.get("reasoning_tokens", 0)
            sample.reasoning_content = result.get("reasoning_content", "")
            
            sample.pred = prompter.get_answer(sample.raw_output)
            
    elif question_type == "executive":
        # 执行类问题：提取执行序列
        sample.pred = prompter.get_answer(sample.raw_output)
        
        # 执行序列可能需要二次提示以获得更规范的输出
        if not is_valid_execution_sequence(sample.pred):
            sample.prompt = prompter.run(sample, is_second_prompt=True)
            
            # 运行模型获取结果（字典形式）
            result = model.run(sample.prompt, image)
            
            # 将结果分配给sample对象的相应字段
            sample.raw_output = result["content"]
            sample.total_tokens = result.get("total_tokens", 0)
            sample.reasoning_tokens = result.get("reasoning_tokens", 0)
            sample.reasoning_content = result.get("reasoning_content", "")
            
            sample.pred = prompter.get_answer(sample.raw_output)
            
    else:
        raise ValueError(f"不支持的问题类型: {question_type}。只支持 'descrip' 或 'executive'")
            
    return sample

# def evaluate(
#     dataset: str,
#     task_name: str,
#     question_type: str,
#     output_dir: str = "outputs",
#     **kwargs,
# ):
#     print(locals())
#     image_dir = f"{dataset}/data"
#     data_path = f"{dataset}/data/{task_name}.json"

#     data = Data.load_with_image_dir(data_path, image_dir)
#     model_name = kwargs.get("model_name")
#     path_out = f"{output_dir}/{dataset}/{question_type}/{model_name}/{task_name}.jsonl"
#     print(dict(path_out=path_out))

#     progress = tqdm(data.samples, desc=path_out)

#     is_correct = []  # 添加初始化
#     for sample in progress:
#         # 寻找prompter 初始化
#         if model_name == "o1":
#             cot = False
#         else:
#             cot = True

#         sample = generate_answer(sample, select_model(**kwargs), question_type, cot)
        
#         # Get Scorer
#         if question_type == "descrip":
#             scorer = LLMScorer()

#         elif question_type == "executive":
#             scorer = ExcutionScorer()

#         else:
#             raise f"Unknown question type: {question_type}"

#         score = scorer.run(sample)
#         sample.score = score
#         if score == -1:
#             pass
#         else:
#             is_correct.append(score)

#         current_score = sum(is_correct) / len(is_correct) if is_correct else 0
#         progress.set_postfix(score=current_score)
#         print(sample.model_dump_json(indent=2, exclude={"image_string"}))
#         print(dict(is_correct=is_correct[-1]))
#         data.save(path_out)

def batch_generate_samples(
    dataset: str,
    task_name: str,
    question_type: str,
    output_dir: str = "outputs",
    sample_probability: float = 1.0,
    output_name: str = None,
    start_idx: int = None,
    end_idx: int = None,
    use_image: bool = True,
    **kwargs,
):
    """
    批量生成样本，但不进行评分评估
    
    Args:
        dataset: 数据集名称或路径
        task_name: 任务名称
        question_type: 问题类型
        output_dir: 输出目录
        sample_probability: 样本选择概率，范围0-1，例如0.5表示50%概率选择每个样本
        output_name: 自定义输出文件名，如果为None则使用task_name
        start_idx: 开始样本索引（包含），如果为None则从第一个样本开始
        end_idx: 结束样本索引（包含），如果为None则处理到最后一个样本
        use_image: 是否使用图像，默认为True，设为False时只使用文本
        **kwargs: 其他参数，如模型名称
    """
    print(locals())
    
    # 使用绝对路径访问文件
    import os
    from pathlib import Path
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 检查dataset路径格式，支持direct_path模式
    if dataset.startswith("/") or (len(dataset) > 1 and dataset[1] == ":"):
        # 直接路径模式 (如 "/data/mental_rotation" 或 "D:/data/mental_rotation")
        if dataset.startswith("/"):
            # 将/data/mental_rotation转换为适合当前系统的绝对路径
            # 移除开头的/并与当前目录结合
            dataset_path = os.path.join(current_dir, dataset.lstrip("/"))
        else:
            # 已经是绝对路径，直接使用
            dataset_path = dataset
            
        # 检查路径中是否已包含data子目录
        base_path = Path(dataset_path)
        if any(part == "data" for part in base_path.parts):
            # 路径中已包含data目录，如D:/Projects/P3_CLER_Framework/Data_generation/data/mental_rotation
            # 直接使用该目录作为图像目录
            image_dir = str(base_path)
        else:
            # 路径中不包含data目录，在路径后添加data子目录
            image_dir = os.path.join(dataset_path, "data")
            
        data_path = os.path.join(dataset_path, f"{task_name}.json")
    else:
        # 传统模式 (如 "evaluation/data/mental_rotation")
        dataset_path = os.path.join(current_dir, dataset)
        
        # 检查路径是否已经包含data目录
        base_path = Path(dataset_path)
        if any(part == "data" for part in base_path.parts):
            # 路径中已包含data目录，如evaluation/data/mental_rotation
            # 直接使用该目录作为图像目录
            image_dir = str(base_path)
        else:
            # 路径中不包含data目录，在路径后添加data子目录
            image_dir = os.path.join(dataset_path, "data")
            
        data_path = os.path.join(dataset_path, f"{task_name}.json")

    print(f"数据集路径: {dataset_path}")
    print(f"图像目录: {image_dir}")
    print(f"数据文件路径: {data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        # 尝试其他可能的路径
        alt_path = data_path.replace('\\data\\', '\\')
        if os.path.exists(alt_path):
            data_path = alt_path
            print(f"找到替代路径: {data_path}")
        else:
            raise FileNotFoundError(f"找不到数据文件: {data_path}\n请检查原始路径和替代路径: {alt_path}")
    
    # 处理数据加载前的路径修正
    # 这里添加钩子函数来修复image_dir中可能的路径重复问题
    def fix_image_paths(sample_list, img_dir):
        """修复样本中的图像路径，防止重复的data目录"""
        fixed_paths = []
        for sample in sample_list:
            # 检查样本的image属性是否已经包含data目录
            if sample.image.startswith('data/') and 'data' in Path(img_dir).parts:
                # 如果都包含，则从样本的image属性中移除data/前缀
                sample.image = sample.image.replace('data/', '', 1)
            fixed_paths.append(sample.image)
        print(f"修复后的图像路径示例: {fixed_paths[:3] if fixed_paths else '无样本'}")
        return img_dir
    
    # 加载数据并修复图像路径
    data = Data.load_with_image_dir(data_path, image_dir)
    
    # 在数据加载完成后修复可能的图像路径问题
    fix_image_paths(data.samples, image_dir)
    
    model_name = kwargs.get("model_name")
    
    # 导入datetime模块获取时间戳
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定输出文件名
    if output_name is None:
        output_name = f"{task_name}_samples"
    
    # 构建包含序列范围信息的输出文件名
    range_info = ""
    if start_idx is not None or end_idx is not None:
        start_str = str(start_idx) if start_idx is not None else "start"
        end_str = str(end_idx) if end_idx is not None else "end"
        range_info = f"_{start_str}-{end_str}"
    
    # 使用自定义路径以区分评估结果，加入时间戳、序列范围和自定义名称
    path_out = os.path.join(output_dir, dataset.lstrip("/"), question_type, model_name, f"{output_name}{range_info}_{timestamp}.jsonl")
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    print(f"输出路径: {path_out}")
    
    # 应用序列选择逻辑
    samples_to_process = data.samples
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else len(data.samples) - 1
        # 确保索引在有效范围内
        start = max(0, min(start, len(data.samples) - 1))
        end = max(0, min(end, len(data.samples) - 1))
        # 如果起始索引大于结束索引，则交换它们
        if start > end:
            start, end = end, start
        samples_to_process = data.samples[start:end+1]  # +1是为了包含end_idx
        print(f"选择样本范围: {start} 到 {end}，共 {len(samples_to_process)} 个样本")
    
    progress = tqdm(samples_to_process, desc=path_out)
    model = select_model(**kwargs)
    
    # 创建动作解析器（复用ExcutionScorer中的解析方法）
    action_parser = ExcutionScorer()
    
    # 导入随机模块
    import random
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    
    # 创建即时保存文件路径 (添加in_progress标记)
    interim_path_out = path_out.replace('.jsonl', '_in_progress.jsonl')
    
    # 处理的样本计数
    processed_count = 0
    # 跳过的样本计数
    skipped_count = 0
    # 已保存的样本计数
    saved_count = 0
    # 当前样本在原始数据集中的索引
    current_index = start if start_idx is not None else 0

    for i, sample in enumerate(progress):
        # 打印当前处理的样本索引
        print(f"处理样本索引: {current_index} (范围内第{i+1}个样本)")
        # 更新当前索引
        current_index = start + i if start_idx is not None else i
        # 根据概率决定是否处理该样本
        if random.random() > sample_probability:
            skipped_count += 1
            continue
            
        processed_count += 1
        
        # 设置COT参数
        if model_name == "o1":
            cot = False
        else:
            cot = True

        # 添加执行格式和示例，后面要针对不同环境类型进行定制化增加
        if question_type == "executive" and sample.environment_type == "moving_box":
            sample.execution_format = "The action sequence is a series of single-character direction instructions for the Sokoban character (movable player): 'u' up, 'd' down, 'l' left, 'r' right."
            sample.execution_example = "['u', 'l', 'l', 'd', 'r', 'r', 'u']"

        if question_type == "executive" and sample.environment_type == "wood_slide":
            sample.execution_format = "The action sequence is a series of two-part actions for wood blocks, like ['3u', '4r', '6d']. The first part is the wood block number, and the second part is the move direction: 'u' up, 'd' down, 'l' left, 'r' right. Each action moves one wood block one unit distance. So the action sequence should be a list of two-part actions. So the action sequence ['3u', '4r', '6d'] means block 3 moves up, block 4 moves right, block 6 moves down. Note that if a block needs to move multiple units in the same direction, you should include the action multiple times. For example, if block 4 needs to move right by 2 units, the sequence should include '4r' twice: ['3u', '4r', '4r', '6d']."
            sample.execution_example = "['3u', '4r', '6d']"

        # 生成答案
        sample = generate_answer(sample, model, question_type, cot, use_image=use_image)
        
        # 对于执行类问题，解析动作序列
        if question_type == "executive":
            sample.formatted_execution_plan = action_parser.parse_actions_sequence(sample)
            print(f"解析后的动作序列: {sample.formatted_execution_plan}")
            
        # 如果样本已有raw_output，就即时保存到中间文件
        if hasattr(sample, 'raw_output') and sample.raw_output:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(interim_path_out), exist_ok=True)
            # 将单个样本即时追加到中间文件
            with open(interim_path_out, 'a', encoding='utf-8') as f:
                f.write(sample.model_dump_json(exclude={"image_string"}) + '\n')
            saved_count += 1
            print(f"\n已即时保存样本 #{current_index} 到中间文件: {interim_path_out}")
            print(f"已保存样本数: {saved_count}")

        # 打印样本信息
        print(sample.model_dump_json(indent=2, exclude={"image_string"}))
    
    # 创建一个只包含处理过的样本的新Data对象并保存
    # 中间文件存在的话，将其重命名为最终文件
    if os.path.exists(interim_path_out):
        # 如果已经有最终文件，则先备份
        if os.path.exists(path_out):
            backup_path = path_out + ".bak"
            try:
                os.rename(path_out, backup_path)
                print(f"原文件已备份为: {backup_path}")
            except Exception as e:
                print(f"备份原文件失败: {e}")
        
        # 将中间文件重命名为最终文件
        try:
            os.rename(interim_path_out, path_out)
            print(f"中间文件已重命名为最终文件: {path_out}")
        except Exception as e:
            print(f"重命名中间文件失败: {e}")
            print(f"请手动将中间文件 {interim_path_out} 重命名为 {path_out}")
    else:
        # 如果中间文件不存在，从样本列表生成文件
        processed_samples = [s for s in samples_to_process if hasattr(s, 'raw_output') and s.raw_output]
        if processed_samples:
            processed_data = Data(samples=processed_samples)
            processed_data.save(path_out)
            print(f"已将{len(processed_samples)}个处理过的样本保存到: {path_out}")
            
            # 输出样本处理统计
            with_pred = len([s for s in processed_samples if hasattr(s, 'pred') and s.pred])
            without_pred = len(processed_samples) - with_pred
            print(f"其中：有pred属性的样本: {with_pred}个, 仅有raw_output的样本: {without_pred}个")
    
    # 打印保存情况统计
    print(f"已处理样本总数: {processed_count}, 已保存样本数: {saved_count}, 跳过样本数: {skipped_count}")
    
    # 打印处理统计信息
    total_available = len(samples_to_process)
    print(f"可处理样本数: {total_available}, 实际处理样本数: {processed_count}, 跳过样本数: {skipped_count}")
    if total_available > 0:  # 避免除以零错误
        print(f"实际处理比例: {processed_count/total_available:.2%}, 目标处理比例: {sample_probability:.2%}")
    print(f"相对于全部样本({len(data.samples)}个)的处理比例: {processed_count/len(data.samples):.2%}")

def analyze_failures_from_file(
    input_file: str,
    output_dir: str = "outputs",
    sample_rate: float = 1.0,
    **kwargs,
):
    """
    从已有的评分结果文件中加载失败样本进行错误原因分析
    
    Args:
        input_file: 输入文件路径，应该是已经通过evaluate_from_file评分过的jsonl或json文件
        output_dir: 输出目录
        sample_rate: 采样率，范围0-1，表示分析样本的概率，默认为1.0（分析所有失败样本）
        **kwargs: 其他参数
    """
    # 解析输入文件路径以获取必要信息
    input_path = Path(input_file)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建输出文件路径
    output_path = Path(f"{output_dir}/analysis_{input_path.name}")
    
    # 创建临时结果路径（用于实时保存）
    temp_output_path = Path(f"{output_dir}/analysis_{input_path.name}_in_progress")
    
    # 检测文件类型（json或jsonl）并加载数据
    print(f"从文件加载样本: {input_file}")
    if input_file.endswith('.jsonl'):
        # JSONL格式，逐行加载
        data = Data.load(input_file)
    else:
        # JSON格式，作为一个整体加载
        data = Data.load(input_file)
    
    # 获取样本列表
    samples = data.samples
    
    # 如果样本是单个样本而非列表，转换为列表
    if not isinstance(samples, list):
        samples = [samples]
    
    # 创建分析器
    analyzer = FailureAnalyzer()
    
    # 计数器
    total_samples = len(samples)
    analyzed_samples = 0  # 成功分析的样本数量
    skipped_samples = 0   # 因为不是失败样本或采样率而跳过的样本
    failed_analysis_samples = 0  # 分析失败的样本
    
    # 分析结果
    sample_results = []
    
    # 确保临时文件不存在
    if temp_output_path.exists():
        temp_output_path.unlink()
    
    # 验证采样率参数
    if not (0 <= sample_rate <= 1):
        raise ValueError(f"采样率必须在0到1之间，当前值: {sample_rate}")
    
    # 筛选出失败样本 (score = -2.0)
    failure_samples = [sample for sample in samples if hasattr(sample, 'score') and sample.score == -2.0]
    
    # 显示失败样本数量
    print(f"总样本数: {total_samples}, 失败样本数: {len(failure_samples)}")
    
    # 显示采样率信息
    if sample_rate < 1:
        print(f"当前采样率: {sample_rate*100:.2f}%, 预计分析 {int(len(failure_samples) * sample_rate)} 个失败样本")
    
    # 逐个样本分析
    for i, sample in enumerate(tqdm(failure_samples, desc="分析失败样本中")):
        # 根据采样率决定是否分析当前样本
        if random.random() > sample_rate:
            # 跳过当前样本
            skipped_samples += 1
            continue
        
        try:
            # 执行分析
            failure_reason = analyzer.run(sample)
            
            # 添加分析结果到样本
            sample_json = sample.model_dump_json()
            result = json.loads(sample_json)
            result["failure_reason"] = failure_reason
            result["analysis_status"] = "analyzed"
            sample_results.append(result)
            analyzed_samples += 1
            
            # 实时保存当前样本结果到临时文件
            with open(temp_output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            # 每10个样本或最后一个样本，输出当前进度信息
            if (i + 1) % 10 == 0 or i == len(failure_samples) - 1:
                print(f"\n当前分析进度 ({i+1}/{len(failure_samples)}):")
                print(f"已分析样本: {analyzed_samples}")
                print(f"已跳过样本: {skipped_samples}")
                print(f"分析失败样本: {failed_analysis_samples}")
                
        except Exception as e:
            # 捕获分析过程中的任何异常
            print(f"样本 {i+1}/{len(failure_samples)} 分析失败: {str(e)}")
            failed_analysis_samples += 1
            # 记录错误信息
            error_result = {
                "index": i,
                "error": str(e),
                "sample_id": getattr(sample, "id", f"sample_{i}") if hasattr(sample, "id") else f"sample_{i}"
            }
            with open(temp_output_path.with_suffix('.errors'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
    
    # 打印结果统计
    print(f"\n分析结果统计:")
    print(f"总样本数: {total_samples}")
    print(f"失败样本数: {len(failure_samples)}")
    print(f"采样率: {sample_rate*100:.2f}%")
    print(f"已分析样本: {analyzed_samples}")
    print(f"已跳过样本: {skipped_samples}")
    print(f"分析失败样本: {failed_analysis_samples}")
    
    # 创建包含统计信息的结果对象
    result_data = {
        "samples": sample_results,
        "statistics": {
            "total_samples": total_samples,
            "failure_samples": len(failure_samples),
            "sample_rate": sample_rate,
            "analyzed_samples": analyzed_samples,
            "skipped_samples": skipped_samples,
            "failed_analysis_samples": failed_analysis_samples,
        }
    }
    
    # 保存完整结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    # 如果临时文件存在，删除它（表示分析正常完成）
    if temp_output_path.exists():
        temp_output_path.unlink()
    
    print(f"分析结果已保存至: {output_path}")
    print(f"单行结果也已保存至: {temp_output_path}")
    
    return result_data

def evaluate_from_file(
    input_file: str,
    output_dir: str = "outputs",
    sample_rate: float = 1.0,
    **kwargs,
):
    """
    从已有的jsonl/json文件中加载样本进行评分，而不是重新生成答案
    
    Args:
        input_file: 输入文件路径，可以是jsonl或json文件
        output_dir: 输出目录
        sample_rate: 采样率，范围0-1，表示评估样本的概率，默认为1.0（评估所有样本）
        **kwargs: 其他参数
    """
    # 解析输入文件路径以获取必要信息
    input_path = Path(input_file)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建输出文件路径
    output_path = Path(f"{output_dir}/eval_{input_path.name}")
    
    # 创建临时结果路径（用于实时保存）
    temp_output_path = Path(f"{output_dir}/eval_{input_path.name}_in_progress")
    
    # 检测文件类型（json或jsonl）并加载数据
    print(f"从文件加载样本: {input_file}")
    if input_file.endswith('.jsonl'):
        # JSONL格式，逐行加载
        data = Data.load(input_file)
    else:
        # JSON格式，作为一个整体加载
        data = Data.load(input_file)
    
    # 获取样本列表
    samples = data.samples
    
    # 如果样本是单个样本而非列表，转换为列表
    if not isinstance(samples, list):
        samples = [samples]
    
    # 创建评分器
    scorer_class = LLMScorer if kwargs.get("question_type") == "descrip" else ExcutionScorer
    scorer = scorer_class()
    
    # 计数器
    total_samples = 0
    valid_success_samples = 0  # 得分为1的样本
    valid_failure_samples = 0  # 得分为0的样本
    invalid_samples = 0   # 得分为-1的样本
    parsing_error_samples = 0  # 得分为-2的样本
    skipped_samples = 0   # 因采样率而跳过的样本
    
    # 评分结果
    sample_results = []
    
    # 确保临时文件不存在
    if temp_output_path.exists():
        temp_output_path.unlink()
    
    # 验证采样率参数
    if not (0 <= sample_rate <= 1):
        raise ValueError(f"采样率必须在0到1之间，当前值: {sample_rate}")
        
    # 显示采样率信息
    if sample_rate < 1:
        print(f"当前采样率: {sample_rate*100:.2f}%，预计评估 {int(len(samples) * sample_rate)} 个样本")
    
    # 逐个样本评分
    for i, sample in enumerate(tqdm(samples, desc="评分中")):
        # 记录总样本数
        total_samples += 1
        
        # 根据采样率决定是否评估当前样本
        if random.random() > sample_rate:
            # 跳过当前样本
            skipped_samples += 1
            # 将样本标记为跳过
            sample.score = None
            sample_json = sample.model_dump_json()
            result = json.loads(sample_json)
            result["evaluation_status"] = "skipped"
            sample_results.append(result)
            continue
        
        try:
            # 执行评分
            score = scorer.run(sample)
            
            # 根据不同得分类型记录数量
            if score == 1.0:
                valid_success_samples += 1
            elif score == 0.0:
                valid_failure_samples += 1
            elif score == -1.0:
                invalid_samples += 1
            elif score == -2.0:
                parsing_error_samples += 1
                
            # 添加得分到样本并保存
            sample.score = score
            
            # 使用model_dump_json代替json()方法（Pydantic v2兼容）
            sample_json = sample.model_dump_json()
            result = json.loads(sample_json)
            result["evaluation_status"] = "evaluated"
            sample_results.append(result)
            
            # 实时保存当前样本结果到临时文件
            with open(temp_output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            # 每10个样本或最后一个样本，输出当前进度信息
            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                valid_samples = valid_success_samples + valid_failure_samples
                invalid_total = invalid_samples + parsing_error_samples
                accuracy = valid_success_samples / valid_samples if valid_samples > 0 else 0
                
                print(f"\n当前评估进度 ({i+1}/{len(samples)}):")
                print(f"有效样本数: {valid_samples} ({valid_samples/(total_samples-skipped_samples)*100:.2f}% 已评估样本)")
                print(f"  - 正确样本: {valid_success_samples} ({valid_success_samples/valid_samples*100:.2f}% 如果有效样本>0)")
                print(f"准确率: {accuracy*100:.2f}% (仅考虑有效样本)")
                print(f"已跳过样本数: {skipped_samples} ({skipped_samples/total_samples*100:.2f}%)")
                
        except Exception as e:
            # 捕获评分过程中的任何异常
            print(f"样本 {i+1}/{len(samples)} 评分失败: {str(e)}")
            # 记录错误信息并继续评分下一个样本
            error_result = {
                "index": i,
                "error": str(e),
                "sample_id": getattr(sample, "id", f"sample_{i}") if hasattr(sample, "id") else f"sample_{i}"
            }
            with open(temp_output_path.with_suffix('.errors'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
    
    # 计算有效样本数量与准确率
    valid_samples = valid_success_samples + valid_failure_samples
    invalid_total = invalid_samples + parsing_error_samples
    
    # 计算准确率（仅基于有效样本）
    accuracy = valid_success_samples / valid_samples if valid_samples > 0 else 0
    
    # 获取实际评估的样本数（总数减去跳过的）
    actually_evaluated = total_samples - skipped_samples
    
    # 打印结果统计
    print(f"\n评估结果统计:")
    print(f"总样本数: {total_samples}")
    print(f"采样率: {sample_rate*100:.2f}%")
    print(f"已评估样本: {actually_evaluated} ({actually_evaluated/total_samples*100:.2f}%)")
    print(f"已跳过样本: {skipped_samples} ({skipped_samples/total_samples*100:.2f}%)")
    print(f"有效样本数: {valid_samples} ({valid_samples/actually_evaluated*100:.2f}% 已评估样本)")
    print(f"  - 正确样本: {valid_success_samples} ({valid_success_samples/valid_samples*100:.2f}% 如果有效样本>0)")
    print(f"  - 错误样本: {valid_failure_samples} ({valid_failure_samples/valid_samples*100:.2f}% 如果有效样本>0)")
    print(f"无效样本数: {invalid_total} ({invalid_total/actually_evaluated*100:.2f}% 已评估样本)")
    print(f"  - 样本问题: {invalid_samples} ({invalid_samples/actually_evaluated*100:.2f}% 已评估样本)")
    print(f"  - 解析错误: {parsing_error_samples} ({parsing_error_samples/actually_evaluated*100:.2f}% 已评估样本)")
    print(f"准确率: {accuracy*100:.2f}% (仅考虑有效样本)")
    
    # 创建包含统计信息的结果对象
    result_data = {
        "samples": sample_results,
        "statistics": {
            "total_samples": total_samples,
            "sample_rate": sample_rate,
            "evaluated_samples": actually_evaluated,
            "skipped_samples": skipped_samples,
            "valid_samples": valid_samples,
            "valid_success_samples": valid_success_samples,
            "valid_failure_samples": valid_failure_samples,
            "invalid_samples": invalid_samples,
            "parsing_error_samples": parsing_error_samples,
            "accuracy": accuracy
        }
    }
    
    # 保存完整结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    # 如果临时文件存在，删除它（表示评估正常完成）
    if temp_output_path.exists():
        temp_output_path.unlink()
    
    print(f"评估结果已保存至: {output_path}")
    print(f"单行结果也已保存至: {temp_output_path}")
    
    return result_data

def need_refinement(pred):
    # 这里可以添加判断逻辑，例如：检查答案长度、是否包含特定关键词等
    # 简化实现，默认不需要优化
    return False
        
def is_valid_execution_sequence(pred):
    # 这里可以添加判断逻辑，例如：检查序列格式、步骤完整性等
    # 简化实现，默认有效
    return True


""" # Scoring
        score = scorer.run(sample)

        sample.correct = score
        is_correct.append(score)
        score = sum(is_correct) / len(is_correct)
        progress.set_postfix(score=score)
        print(sample.json(indent=2, exclude={"image_string"}))
        print(dict(is_correct=is_correct[-1]))
        data.save(path_out)  """

"""

"""


"""
python main.py evaluate --dataset PuzzleVQA --puzzle color_hexagon --question_type open --model_name gpt4o --output_dir outputs_test
python main.py evaluate --dataset PuzzleVQA --puzzle color_hexagon --question_type mcq --model_name gpt4o --output_dir outputs_test
python main.py evaluate --dataset AlgoPuzzleVQA --puzzle board_tile --question_type open --model_name gpt4o --output_dir outputs_test
python main.py evaluate --dataset AlgoPuzzleVQA --puzzle board_tile --question_type mcq --model_name gpt4o --output_dir outputs_test
python generate_response_samples.py batch_generate_samples --dataset /data/mental_rotation --task_name mental_rotation_samples --question_type descrip --model_name gpt4o --output_dir outputs
python generate_response_samples.py batch_generate_samples --dataset /data/mental_rotation --task_name mental_rotation_samples --question_type descrip --model_name o1 --output_dir outputs_o4 --sample_probability 0.4
python generate_response_samples_new.py batch_generate_samples --dataset /data/rubiks_cube  --task_name rubiks_cube_colors_2025-05-08   --question_type descrip --start_idx 100  --end_idx 150  --model_name gemini  --output_dir outputs_o4
python .\generate_response_samples_new.py evaluate_from_file --input_file "D:\Projects\P3_CLER_Framework\Data_generation\evaluation\outputs\data\move_box\executive\deepseek\move_box_fine_tune_cot_for_DS_samples_20250424_011849_filtered_20250512_102138.jsonl"  --output_dir "outputs/move_box_DS" --question_type "executive"
analysis = analyze_failures_from_file("output_dir/eval_samples.jsonl", "analysis_output_dir")

"""
if __name__ == "__main__":
    Fire()
