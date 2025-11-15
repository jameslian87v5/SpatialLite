import base64
import io
import json
import random
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

Point = Tuple[float, float]


def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def convert_image_to_bytes(image: Image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return data


def convert_text_to_image(text: str) -> Image:
    data = base64.b64decode(text.encode("utf-8"))
    return Image.open(io.BytesIO(data))


def load_image(path: str) -> Image:
    if Path(path).exists():
        return Image.open(path)

    response = requests.get(path)
    return Image.open(io.BytesIO(response.content))


def sample_options(answer: str, options: List[str], k: int):
    # Ensure random order and no duplicates
    options = [o for o in options if o != answer]
    assert len(options) + 1 >= k
    options = random.sample(options, k=k - 1)
    options.append(answer)
    assert len(set(options)) == k
    return random.sample(options, k=k)


class Sample(BaseModel):
    question: str
    answer: str
    options: List[str] = []
    image: str
    image_string: str = ""
    caption: str = ""
    explanation: str = ""
    deduction: str = ""
    prompt: str = ""
    raw_output: str = ""
    pred: str = ""
    # 任务类型， 包含Mental_rotation、Cube_rolling、Rubiks_cube、Wood_slide、Move_box、Number_puzzle 六大类    
    task_type: str = ""
    score: float = -1
    # 环境类型，支持以下三种类型之一：
    # - "moving_box": 推箱子类游戏，动作格式为 move_<direction>
    # - "wood_slide": 华容道类游戏，动作格式为 move_<piece_id>_<direction>
    # - "number_slide": 数字华容道类游戏，动作格式为 move_<number>_<direction>
    environment_type: str = ""
    # 问题类型，只能是以下两种类型之一：
    # - "descrip": 描述性问题，需要语言模型生成描述性答案
    # - "executive": 执行类问题，需要语言模型生成可执行的动作序列
    question_type: str = ""
    execution_format: str = ""
    execution_example: str = ""
    formatted_execution_plan: List[str] = []
    initial_state: dict = {}
    final_state: dict = {}
    reasoning_content: str = ""
    total_tokens: int = 0
    reasoning_tokens: int = 0
    raw_parse_actions: str = ""


class Data(BaseModel):
    samples: List

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in self.samples:
                # 先使用model_dump获取字典，再使用json.dumps处理ensure_ascii
                data_dict = s.model_dump(exclude={"image_string"})
                json_str = json.dumps(data_dict, ensure_ascii=False)
                print(json_str, file=f)

    @classmethod
    def load(cls, path: str):
        samples = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 确保 answer 是字符串类型
                if 'answer' in data and not isinstance(data['answer'], str):
                    data['answer'] = str(data['answer'])
                samples.append(Sample(**data))
        print(dict(path=path, samples=len(samples)))
        return cls(samples=samples)

    @classmethod
    def load_with_image_dir(cls, path: str, image_dir: str):
        data = cls.load(path)
        for s in tqdm(data.samples, desc=path):
            # 尝试多种可能的路径组合
            possible_paths = [
                Path(image_dir, s.image),                       # 原始路径
                Path(image_dir, "images", s.image),             # 添加images子目录
                Path(str(Path(image_dir).parent), s.image),     # 去掉最后一级目录
                Path(str(Path(image_dir).parent), "images", s.image),  # 去掉最后一级目录并添加images
                Path("D:/Projects/P3_CLER_Framework/Data_generation/data/images", s.image.split("/")[-1] if "/" in s.image else s.image)  # 直接使用固定路径
            ]
            
            # 尝试每个可能的路径
            found = False
            for path_image in possible_paths:
                try:
                    if path_image.exists():
                        print(f"找到图像: {path_image}")
                        image = Image.open(path_image)
                        s.image_string = convert_image_to_text(image)
                        found = True
                        break
                except:
                    continue
            
            if not found:
                paths_str = "\n".join([str(p) for p in possible_paths])
                raise FileNotFoundError(f"无法找到图像文件。尝试了以下路径:\n{paths_str}")
                
        return data

    def analyze(self):
        for s in random.sample(self.samples, k=4):
            s = s.copy(deep=True)
            s.image_string = s.image_string[:80] + "..."
            print(s.json(indent=2))
        for s in self.samples:
            assert "..." not in s.image_string and len(s.image_string) > 100
        info = dict(
            samples=len(self.samples),
            unique_samples=len(set(s.json() for s in self.samples)),
        )
        print(json.dumps(info, indent=2))


def test_data(**kwargs):
    data = Data.load_with_image_dir(**kwargs)
    data.analyze()


if __name__ == "__main__":
    Fire()
