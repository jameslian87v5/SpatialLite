import re
from typing import List
from modeling import select_model, EvalModel
from environment import BoxEnv

def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        o1=O1Model,
        gpt4o=GPT4oModel,
        gpt4t=GPT4tModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


class Prompter:
    def run(self, sample) -> str:
        raise NotImplementedError


class BasePrompter(Prompter):
    def __init__(self, question_type: str, cot: str):
        # 问题类型，只能是以下两种类型之一：
        # - "descrip": 描述性问题，需要语言模型生成描述性答案
        # - "executive": 执行类问题，需要语言模型生成可执行的动作序列
        self.question_type = question_type
        self.cot = cot

    def run(self, sample, is_second_prompt=False) -> str:
        # 验证question_type是否为支持的类型
        if self.question_type not in ["descrip", "executive"]:
            raise ValueError(f"不支持的问题类型: {self.question_type}。只支持 'descrip' 或 'executive'。")
            
        # 检查sample是否有必要的属性
        if not hasattr(sample, 'question'):
            raise ValueError("样本缺少question属性")
        
        # 如果是第二次提示，使用更复杂的提示格式
        if is_second_prompt:
            # 检查样本是否有原始输出
            if not hasattr(sample, 'raw_output') or not sample.raw_output:
                # 如果没有原始输出，回退到基础提示
                return self.run(sample, is_second_prompt=False)
                
            # 先获取基础提示
            base_prompt = self.run(sample, is_second_prompt=False)
            
            # 构建完整提示，包含原始输出和后续引导
            parts = [
                base_prompt,
                sample.raw_output,
                "",
            ]
            
            if self.question_type == "descrip":
                # 描述性问题的完整提示不需要额外引导
                pass
                
            elif self.question_type == "executive":
                # 执行类问题需要引导出最终执行方案
                parts.append("Based on the above analysis, the final executive solution is:")
                
            return "\n".join(parts)
        
        # 基础提示生成（第一次提示）
        base_question = sample.question.rstrip()
        environment_type = sample.environment_type
        
        if self.question_type == "descrip":
            # 描述性问题，添加思考步骤提示
            prompt = f"Question: {base_question}\n\nPlease tell me your answer, be concise in your thinking and step by step. And Please show your thinking process, details and evidence and any keypoint in your final answer, ok？"
        
        elif self.question_type == "executive":

            if environment_type == "moving_box":
                initial_state = sample.initial_state
                # 执行类问题，添加执行解决方案提示
                prompt = f"Question: {base_question}\n\nthe initial state is {initial_state}\n\nPlease be concise in your thinking and figure out the final executive solution. And Please show your thinking process, details and evidence and any keypoint or anything valuable in your final answer. the final answer should be the most deatil and thinking process! you should do your best ! "
            elif environment_type == "wood_slide":
                prompt = f"Question: {base_question}\n\nPlease show your thinking and step by step and figure out the final executive solution, be concise in your thinking. And Please show your thinking process, details and evidence and any keypoint or anything valuable in your final answer. the final answer should be the most deatil and thinking process! you should do your best ! "
       
        return prompt

    def get_answer(self, text: str, options: List[str] = None):
        if self.question_type == "descrip":
            # 描述性问题：返回完整回答
            return text.strip()
            
        elif self.question_type == "executive":
            # 执行类问题：尝试提取执行序列
            # 这里只是基础实现，具体的执行序列提取逻辑可能需要更复杂的处理
            # 通常这部分逻辑会在main.py的ExcutionScorer中进一步处理
            return text.strip()
            
        else:
            raise ValueError(f"不支持的问题类型: {self.question_type}。只支持 'descrip' 或 'executive'。")
