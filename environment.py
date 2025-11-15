import copy
from typing import Dict, List, Tuple

class BaseEnv:
    """环境基类，定义所有环境的通用接口和基础行为。"""

    def __init__(self, initial_state: Dict, final_state: Dict = None):
        """初始化环境基类。

        Args:
            initial_state (Dict): 初始状态，不同环境有不同的状态表示
            final_state (Dict, optional): 最终目标状态，用于判断任务是否完成。默认为None
        """
        self.initial_state = copy.deepcopy(initial_state)  # 保存初始状态的副本
        self.final_state = copy.deepcopy(final_state) if final_state else None  # 保存目标状态的副本
        self.state = copy.deepcopy(initial_state)  # 当前状态
        self.history = []  # 动作历史
        self.success = False  # 是否成功完成任务
        self.terminal = False  # 是否达到终止状态
    
    def reset(self) -> None:
        """重置环境到初始状态。"""
        self.state = copy.deepcopy(self.initial_state)
        self.history = []
        self.success = False
        self.terminal = False
    
    def step(self, action: str) -> Tuple[bool, str]:
        """执行一个动作，更新环境状态。

        Args:
            action (str): 动作字符串，格式取决于具体环境

        Returns:
            Tuple[bool, str]: (成功标志, 信息)
                成功标志表示动作是否成功执行
                信息提供有关执行结果的详细说明
        """
        raise NotImplementedError
    
    def get_valid_actions(self) -> List[str]:
        """获取当前状态下的有效动作列表。

        Returns:
            List[str]: 有效动作列表
        """
        raise NotImplementedError
    
    def validate_action(self, action: str) -> Tuple[bool, str]:
        """验证动作的格式和合法性。

        Args:
            action (str): 动作字符串

        Returns:
            Tuple[bool, str]: (是否有效, 验证消息)
                是否有效表示动作是否有效
                验证消息提供有关验证结果的详细说明
        """
        raise NotImplementedError
    
    def is_terminal(self) -> bool:
        """检查当前状态是否为终止状态。

        Returns:
            bool: 是否为终止状态
        """
        return self.terminal
    
    def is_success(self) -> bool:
        """检查当前任务是否已成功完成。

        Returns:
            bool: 是否成功完成任务
        """
        return self.success
        
    def get_visualization(self) -> str:
        """返回可视化状态，便于调试。

        Returns:
            str: 状态的文本可视化表示
        """
        raise NotImplementedError

class BoxEnv(BaseEnv):
    """推箱子环境类，定义推箱子接口行为和行为。"""

    execution_format = {
        "action_sequence": ["u", "d", "l", "r"],  # 示例动作序列
        "description": "动作序列是针对推箱子小人（可移动玩家）的一系列单字符方向指令，：'u'上, 'd'下, 'l'左, 'r'右。例如['l','l','d']表示向左移动两次，然后向下移动一次。"
    }

    def __init__(self, initial_state: Dict, final_state: Dict = None):
        """初始化推箱子环境
        
        Args:
            initial_state: 初始状态，包含地图信息和符号描述
            final_state: 最终状态，可选，如果不提供则只使用初始状态
        """
        super().__init__(initial_state, final_state if final_state else {})
        
        # 解析地图
        self.map = initial_state.get("map", [])
        self.descrip = initial_state.get("descrip", {})
        
        # 查找玩家位置、箱子位置和目标位置
        self.player_pos = None
        self.boxes = []
        self.targets = []
        
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                cell = self.map[i][j]
                if cell == self.descrip.get("person", "S"):
                    self.player_pos = [i, j]
                elif cell == self.descrip.get("box", "B"):
                    self.boxes.append([i, j])
                elif cell == self.descrip.get("target", "T"):
                    self.targets.append([i, j])
        
        # 如果没有找到玩家或箱子或目标，抛出异常
        if not self.player_pos:
            raise ValueError("没有找到玩家位置")
        if not self.boxes:
            raise ValueError("没有找到箱子位置")
        if not self.targets:
            raise ValueError("没有找到目标位置")
        
        # 当前状态
        self.current_map = [row[:] for row in self.map]  # 深拷贝地图
        self.current_player_pos = self.player_pos.copy()
        self.current_boxes = [box.copy() for box in self.boxes]
        
        # 动作映射
        self.actions = {
            "u": [-1, 0],  # 上
            "d": [1, 0],   # 下
            "l": [0, -1],  # 左
            "r": [0, 1]    # 右
        }
        
        # 移动历史
        self.move_history = []
    
    @property
    def steps(self):
        """返回移动历史记录，兼容Executor类中的调用"""
        return self.move_history

    def reset(self):
        """重置环境到初始状态"""
        self.current_map = [row[:] for row in self.map]
        self.current_player_pos = self.player_pos.copy()
        self.current_boxes = [box.copy() for box in self.boxes]
        self.move_history = []
        return True

    def get_current_state(self) -> Dict:
        """获取当前状态
        
        Returns:
            Dict: 当前状态
        """
        return {
            "map": self.current_map,
            "player_pos": self.current_player_pos,
            "boxes": self.current_boxes
        }

    def validate_action(self, action: str) -> Tuple[bool, str]:
        """验证动作是否有效
        
        Args:
            action: 动作字符，'u'上, 'd'下, 'l'左, 'r'右
            
        Returns:
            Tuple[bool, str]: (是否有效, 信息)
        """
        if action not in self.actions:
            return False, f"无效动作: {action}, 有效动作为: {list(self.actions.keys())}"
        
        # 计算玩家的下一个位置
        direction = self.actions[action]
        next_row = self.current_player_pos[0] + direction[0]
        next_col = self.current_player_pos[1] + direction[1]
        
        # 检查是否超出边界
        if next_row < 0 or next_row >= len(self.current_map) or next_col < 0 or next_col >= len(self.current_map[0]):
            return False, "玩家移动出界"
        
        # 检查是否撞墙
        if self.current_map[next_row][next_col] == self.descrip.get("brick", "#"):
            return False, "不能穿墙"
        
        # 检查是否推箱子
        box_idx = -1
        for i, box in enumerate(self.current_boxes):
            if box[0] == next_row and box[1] == next_col:
                box_idx = i
                break
        
        if box_idx != -1:
            # 如果前方是箱子，需要检查箱子的下一个位置
            box_next_row = next_row + direction[0]
            box_next_col = next_col + direction[1]
            
            # 检查箱子是否超出边界
            if box_next_row < 0 or box_next_row >= len(self.current_map) or box_next_col < 0 or box_next_col >= len(self.current_map[0]):
                return False, "箱子将移动出界"
            
            # 检查箱子的下一个位置是否是墙或另一个箱子
            if self.current_map[box_next_row][box_next_col] == self.descrip.get("brick", "#"):
                return False, "箱子不能穿墙"
            
            for other_box in self.current_boxes:
                if other_box[0] == box_next_row and other_box[1] == box_next_col:
                    return False, "箱子不能推动另一个箱子"
        
        return True, "动作有效"

    def step(self, action: str) -> Tuple[bool, str]:
        """执行动作
        
        Args:
            action: 动作字符，'u'上, 'd'下, 'l'左, 'r'右
            
        Returns:
            Tuple[bool, str]: (是否成功, 信息)
        """
        valid, message = self.validate_action(action)
        if not valid:
            # 检查是否是碰墙情况
            direction = self.actions[action]
            next_row = self.current_player_pos[0] + direction[0]
            next_col = self.current_player_pos[1] + direction[1]
            
            # 检查是否超出边界或撞墙
            if (next_row < 0 or next_row >= len(self.current_map) or 
                next_col < 0 or next_col >= len(self.current_map[0]) or
                self.current_map[next_row][next_col] == self.descrip.get("brick", "#")):
                # 碰墙情况，记录动作但不移动
                self.move_history.append(action)
                return True, "玩家碰到墙壁，停在原地"
            
            # 其他错误情况，如推两个箱子等
            return False, message
        
        # 计算玩家的下一个位置
        direction = self.actions[action]
        next_row = self.current_player_pos[0] + direction[0]
        next_col = self.current_player_pos[1] + direction[1]
        
        # 检查是否推箱子
        box_idx = -1
        for i, box in enumerate(self.current_boxes):
            if box[0] == next_row and box[1] == next_col:
                box_idx = i
                break
        
        # 更新地图
        # 首先清除玩家当前位置
        self.current_map[self.current_player_pos[0]][self.current_player_pos[1]] = self.descrip.get("empty", ".")
        
        if box_idx != -1:
            # 推箱子
            box = self.current_boxes[box_idx]
            box_next_row = box[0] + direction[0]
            box_next_col = box[1] + direction[1]
            
            # 更新箱子位置
            self.current_map[box[0]][box[1]] = self.descrip.get("empty", ".")
            self.current_map[box_next_row][box_next_col] = self.descrip.get("box", "B")
            self.current_boxes[box_idx] = [box_next_row, box_next_col]
        
        # 更新玩家位置
        self.current_map[next_row][next_col] = self.descrip.get("person", "S")
        self.current_player_pos = [next_row, next_col]
        
        # 记录移动历史
        self.move_history.append(action)
        
        return True, "成功执行动作"

    def is_terminal(self) -> bool:
        """检查当前状态是否为终止状态
        
        Returns:
            bool: 是否为终止状态
        """
        # 检查所有箱子是否都在目标位置上
        for box in self.current_boxes:
            is_on_target = False
            for target in self.targets:
                if box[0] == target[0] and box[1] == target[1]:
                    is_on_target = True
                    break
            if not is_on_target:
                return False
        
        return True
        
    def get_state(self) -> Dict:
        """获取当前状态，兼容Executor的调用
        
        Returns:
            Dict: 当前状态
        """
        return self.get_current_state()


class WoodSlideEnv(BaseEnv):
    """华容道环境类，定义华容道接口行为和行为。"""

    execution_format = {
        "action_sequence": ["3u", "4r", "6d"],  # 示例动作序列
        "description": "动作序列是一系列两部分动作，第一部分是木块编号，第二部分是移动方向：'u'上, 'd'下, 'l'左, 'r'右。例如['3u', '4r', '6d']表示3号木块向上移动，4号木块向右移动，6号木块向下移动。每个动作移动一个木块一个单位距离。注意，必须严格是数字+一个字母形式，严格遵守！！！"
    }

    def __init__(self, initial_state: Dict, final_state: Dict = None):
        """初始化华容道环境
        
        Args:
            initial_state: 初始状态，包含地图信息和符号描述
            final_state: 最终状态，如果提供则用于判断是否达到目标
        """
        super().__init__(initial_state, final_state if final_state else {})
        
        # 解析地图
        self.map = initial_state.get("map", [])
        self.descrip = initial_state.get("descrip", {})
        
        # 当前状态
        self.current_map = [row[:] for row in self.map]  # 深拷贝地图
        
        # 空白位置
        self.empty_symbol = self.descrip.get("empty", "0")
        self.empty_positions = []
        
        # 方块位置和大小
        self.blocks = {}
        
        # 水平检测地图数据和空白符号
        print(f"\n初始化WoodSlideEnv - 空白符号: '{self.empty_symbol}'")
        print(f"地图大小: {len(self.current_map)}x{len(self.current_map[0]) if self.current_map else 0}")
        
        # 找出空白位置和所有方块
        for i in range(len(self.current_map)):
            for j in range(len(self.current_map[i])):
                cell = self.current_map[i][j]
                # 进行更严格的比较，确保浮点数或其他类型也能正确比较
                str_cell = str(cell)
                if str_cell == str(self.empty_symbol):
                    self.empty_positions.append([i, j])
                    print(f"  找到空白位置: [{i},{j}]")
                else:
                    # 方块编号，如 "1"、"2" 等
                    block_id = str(cell)  # 统一转为字符串类型
                    if block_id not in self.blocks:
                        self.blocks[block_id] = [[i, j]]
                    else:
                        self.blocks[block_id].append([i, j])
        
        # 检查是否有两个空白位置
        print(f"\n找到 {len(self.empty_positions)} 个空白位置")
        if len(self.empty_positions) != 2:
            print("\n警告：华容道相关信息检查:")
            print(f"  - initial_state类型: {type(initial_state)}")
            
            # 输出地图的更多详细信息
            print("\n地图内容:")
            for i, row in enumerate(self.current_map):
                print(f"  第{i}行: {row}")
                
            # 输出符号描述信息
            print("\n符号描述:")
            for key, value in self.descrip.items():
                print(f"  {key}: {value}")
                
            # 尝试处理无空白格子的情况，这里选择不进行修复，只是提供更详细的错误信息
            raise ValueError(f"华容道需要有两个空白位置，但找到了 {len(self.empty_positions)} 个。请检查初始状态图和空白符号设置是否一致。使用的空白符号是: '{self.empty_symbol}'")
        
        # 动作映射
        self.actions = {
            "u": [-1, 0],  # 上
            "d": [1, 0],   # 下
            "l": [0, -1],  # 左
            "r": [0, 1]    # 右
        }
        
        # 移动历史
        self.move_history = []
    
    @property
    def steps(self):
        """返回移动历史记录，兼容Executor类中的调用"""
        return self.move_history
    
    def reset(self):
        """重置环境到初始状态"""
        self.current_map = [row[:] for row in self.map]
        self.empty_positions = []
        self.blocks = {}
        self.move_history = []
        self.terminal = False
        self.success = False

        # 扫描地图，识别空白位置和方块
        for i in range(len(self.current_map)):
            for j in range(len(self.current_map[i])):
                cell = self.current_map[i][j]
                # 使用与__init__相同的字符串比较逻辑
                str_cell = str(cell)
                if str_cell == str(self.empty_symbol):
                    self.empty_positions.append([i, j])
                else:
                    # 方块编号，如 "1"、"2" 等
                    block_id = str(cell)  # 统一转为字符串类型
                    if block_id not in self.blocks:
                        self.blocks[block_id] = [[i, j]]
                    else:
                        self.blocks[block_id].append([i, j])

        # 检查是否有两个空白位置
        if len(self.empty_positions) != 2:
            raise ValueError(f"华容道需要两个空白位置，但找到了 {len(self.empty_positions)} 个")

        return True

    
    def get_current_state(self) -> Dict:
        """获取当前状态
        
        Returns:
            Dict: 当前状态
        """
        return {
            "map": self.current_map,
            "empty_positions": self.empty_positions,
            "blocks": self.blocks
        }
    
    def get_block_shape(self, block_id):
        """判断方块的形状类型（2x2, 1x2, 2x1, 1x1）
        
        Args:
            block_id: 方块ID
            
        Returns:
            tuple: (高度, 宽度)
        """
        positions = self.blocks[block_id]
        rows = [pos[0] for pos in positions]
        cols = [pos[1] for pos in positions]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        return (height, width)
    
    def is_block_movable(self, block_id, direction):
        """检查一个方块是否可以在指定方向上移动
        
        Args:
            block_id: 方块ID
            direction: 移动方向 [行增量, 列增量]
            
        Returns:
            tuple: (是否可移动, 错误原因)
        """
        # 获取方块占据的位置
        positions = self.blocks[block_id]
        
        # 方块移动后的新位置
        new_positions = [[pos[0] + direction[0], pos[1] + direction[1]] for pos in positions]
        
        # 找出方块移动后会离开的位置
        leaving_positions = [pos for pos in positions if [pos[0] + direction[0], pos[1] + direction[1]] not in positions]
        
        # 找出方块移动后会进入的新位置
        entering_positions = [new_pos for new_pos in new_positions if new_pos not in positions]
        
        # 1. 检查离开和进入的位置数量是否相等
        if len(leaving_positions) != len(entering_positions):
            return False, f"方块移动位置错误: 离开位置数 {len(leaving_positions)} 与进入位置数 {len(entering_positions)} 不匹配"
        
        # 2. 检查新位置是否有效
        for new_pos in new_positions:
            # 2.1 超出边界检查
            if new_pos[0] < 0 or new_pos[0] >= len(self.current_map) or \
               new_pos[1] < 0 or new_pos[1] >= len(self.current_map[0]):
                return False, f"方块移动超出边界: 新位置 {new_pos} 不合法"
        
        # 3. 检查进入的新位置是否都是空白或当前方块的一部分
        for new_pos in entering_positions:
            # 如果新位置不是空白，则不能移动
            if new_pos not in self.empty_positions:
                # 查找该位置存在的方块
                blocking_block = None
                for bid, block_positions in self.blocks.items():
                    if new_pos in block_positions and bid != block_id:
                        blocking_block = bid
                        break
                
                if blocking_block:
                    return False, f"方块被阻挡: 位置 {new_pos} 被方块 {blocking_block} 占据"
                else:
                    return False, f"方块无法移动: 位置 {new_pos} 不是空白位置"
        
        return True, "可以移动"
    
    def validate_action(self, action: str) -> Tuple[bool, str]:
        """验证动作是否有效
        
        Args:
            action: 动作字符串，如 "1u"
            
        Returns:
            Tuple[bool, str]: (是否有效, 消息)
        """
        if len(action) != 2:
            return False, f"动作格式错误，应为两个字符，如 '1u'，但得到 '{action}'"
        
        block_id, direction_key = action[0], action[1]
        
        # 检查方块ID是否存在
        if block_id not in self.blocks:
            return False, f"方块ID '{block_id}' 不存在"
        
        # 检查方向是否有效
        if direction_key not in self.actions:
            return False, f"方向 '{direction_key}' 无效，应为 'u', 'd', 'l' 或 'r'"
        
        # 检查方块在该方向上是否可移动
        direction = self.actions[direction_key]
        movable, reason = self.is_block_movable(block_id, direction)
        if not movable:
            return False, f"方块 {block_id} 不能向 {direction_key} 方向移动: {reason}"
        
        return True, "动作有效"
    def step(self, action: str) -> Tuple[bool, str]:
        """执行动作
        
        Args:
            action: 动作字符串，形如 "1u"、"3r" 等
            
        Returns:
            Tuple[bool, str]: (是否成功, 信息)
        """
        valid, message = self.validate_action(action)
        if not valid:
            return False, message
        
        block_id = action[0]
        direction_key = action[1]
        direction = self.actions[direction_key]
        
        # 获取方块当前位置
        positions = self.blocks[block_id]
        
        # 方块移动后的新位置
        new_positions = [[pos[0] + direction[0], pos[1] + direction[1]] for pos in positions]
        
        # 空白位置更新逻辑修正
        # 1. 先记录当前方块所有位置
        old_positions = [pos.copy() for pos in positions]
        
        # 2. 更新地图 - 将原位置全部设为空白
        for pos in old_positions:
            self.current_map[pos[0]][pos[1]] = self.empty_symbol
        
        # 3. 再将新位置全部设为方块ID
        for new_pos in new_positions:
            self.current_map[new_pos[0]][new_pos[1]] = block_id
        
        # 4. 更新空白位置列表 - 重新计算
        # 先清除所有空白位置
        self.empty_positions = []
        
        # 重新扫描地图找出空白位置
        for i in range(len(self.current_map)):
            for j in range(len(self.current_map[i])):
                if str(self.current_map[i][j]) == str(self.empty_symbol):
                    self.empty_positions.append([i, j])
        
        print(f"DEBUG: 执行动作 {action} 后，空白位置: {self.empty_positions}")
        
        # 更新方块位置信息
        self.blocks[block_id] = new_positions
        
        # 记录移动历史
        self.move_history.append(action)
        
        # 检查是否达到终止状态
        if self.final_state and self.is_terminal():
            self.terminal = True
            self.success = True
        
        return True, "成功执行动作"
    
    def is_terminal(self) -> bool:
        """检查当前状态是否为终止状态
        
        Returns:
            bool: 是否为终止状态
        """
        if not self.final_state:
            return False
        
        final_map = self.final_state.get("map", [])
        if not final_map:
            return False
        
        # 比较当前地图和目标地图，统一转换为字符串进行比较
        for i in range(len(self.current_map)):
            for j in range(len(self.current_map[i])):
                # 将两边都转换为字符串再比较，解决类型不一致问题
                if str(self.current_map[i][j]) != str(final_map[i][j]):
                    return False
        
        return True
    
    def get_valid_actions(self) -> List[str]:
        """获取当前状态下的有效动作列表
        
        Returns:
            List[str]: 有效动作列表
        """
        valid_actions = []
        
        for block_id in self.blocks:
            for direction_key in self.actions:
                action = f"{block_id}{direction_key}"
                is_valid, _ = self.validate_action(action)
                if is_valid:
                    valid_actions.append(action)
        
        return valid_actions
    
    def get_state(self) -> Dict:
        """获取当前状态，兼容Executor的调用
        
        Returns:
            Dict: 当前状态
        """
        return self.get_current_state()
    
    def get_visualization(self) -> str:
        """返回可视化状态，便于调试
        
        Returns:
            str: 状态的文本可视化表示
        """
        visualization = ""
        for row in self.current_map:
            visualization += " ".join(row) + "\n"
        return visualization