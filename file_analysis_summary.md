# Interactive_Search_Class.py 文件分析总结

## 文件概述
`interactive_search_class.py` 是一个智能旅行规划系统的核心模块，实现了从用户需求到完整旅行计划的自动化生成。

## 主要组件

### 1. 导入模块
- **API工具类**: 酒店(Accommodations)、餐厅(Restaurants)、景点(Attractions)、城际交通(IntercityTransport)
- **LLM模型**: DeepSeek和GPT-4等大语言模型
- **评估模块**: 约束检查、逻辑验证、成本计算
- **机器学习库**: scikit-learn用于文本相似度计算

### 2. 工具函数

#### 时间处理函数
- `time_compare_if_earlier_equal()`: 比较两个时间字符串
- `next_time_delta()`: 调整时间到下一个间隔
- `add_time_delta()`: 在时间上添加间隔

#### 相似度计算函数
- `jaccard_similarity()`: 计算Jaccard相似度
- `mmr_algorithm()`: 最大边际相关性算法，平衡重要性和多样性

### 3. 核心类

#### Logger类
- 同时输出到终端和文件的日志记录器
- 用于程序运行过程的调试和分析

#### Interactive_Search类
这是整个系统的核心类，主要功能包括：

##### 初始化 (`__init__`)
- 配置LLM模型（DeepSeek或GPT-4）
- 初始化各种API工具（酒店、餐厅、景点、交通）
- 设置超时和详细输出选项

##### 符号化搜索 (`symbolic_search`)
**功能**: 解析用户约束条件并转换为程序可理解的格式

**支持的约束类型**:
- `rooms==N`: 房间数量
- `cost<=N`: 总预算限制
- `room_type==N`: 房间类型
- `train_type=='XXX'`: 火车类型
- `{'川菜', '粤菜'} <= food_type`: 餐饮类型偏好
- `transport_type <= {'taxi'}`: 交通方式偏好
- `hotel_price<=N`: 酒店价格限制
- `intercity_transport=='train'`: 城际交通类型
- `{'文化遗址'} <= spot_type`: 景点类型偏好
- `{'天安门'} <= attraction_names`: 特定景点名称
- `{'全聚德'} <= restaurant_names`: 特定餐厅名称
- `{'北京饭店'} <= hotel_names`: 特定酒店名称
- `{'温泉'} <= hotel_feature`: 酒店特性需求

##### 计划搜索 (`search_plan`)
**功能**: 搜索和生成完整的旅行计划

**主要步骤**:
1. 搜索城际交通选项（飞机、火车）
2. 为交通选项评分和排序
3. 选择最优的往返交通组合
4. 计算交通成本并验证预算约束
5. 调用POI搜索生成详细行程

##### POI搜索 (`search_poi`)
**功能**: 生成具体的景点、餐厅、酒店行程安排

**处理的活动类型**:
- 城际交通（去程/回程）
- 早餐（在酒店）
- 午餐/晚餐（根据时间和位置选择餐厅）
- 景点游览（根据用户偏好和LLM评分）
- 酒店住宿（考虑特性和价格要求）

##### LLM评分系统
**相关方法**:
- `score_poi_think_overall_act_page()`: 使用LLM为POI打分
- `reason_prompt()`: 生成LLM推理提示
- `extract_score_from_plan()`: 从LLM输出中提取分数

##### 约束验证 (`constraints_validation`)
**功能**: 验证生成的计划是否满足所有约束条件
- 常识约束检查
- 逻辑约束验证
- 预算约束验证

## 主要特点

### 1. 智能推荐
- 使用大语言模型进行POI评分和推荐
- 结合用户偏好和历史行程进行个性化推荐

### 2. 约束满足
- 支持多种复杂约束条件
- 实时验证约束满足情况
- 预算控制和时间规划

### 3. 时间规划
- 精确的时间计算和调度
- 考虑营业时间、交通时间等因素
- 合理安排用餐和住宿时间

### 4. 多样性保证
- 使用MMR算法避免推荐相似POI
- 平衡重要性和多样性

### 5. 可扩展性
- 模块化设计，易于添加新的约束类型
- 支持多种LLM模型
- 灵活的API接口设计

## 使用流程

1. **输入解析**: 将用户的自然语言需求转换为结构化约束
2. **交通规划**: 搜索和选择最优的城际交通方案
3. **行程生成**: 基于时间、位置、偏好生成详细的每日行程
4. **智能推荐**: 使用LLM对景点、餐厅进行评分和筛选
5. **约束验证**: 确保生成的计划满足所有用户要求
6. **结果输出**: 返回完整的旅行计划

这个系统展示了现代AI在复杂规划任务中的应用，结合了符号推理、机器学习和大语言模型的优势，能够生成高质量的个性化旅行计划。
