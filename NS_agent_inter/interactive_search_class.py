import sys
import re
import random
import numpy as np
import argparse

sys.path.append("../")
from tools.hotels.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.attractions.apis import Attractions
from tools.intercity_transport.apis import IntercityTransport

from envs import goto
import json
import os

from evaluation.utils import load_json_file

from evaluation.commonsense_constraint import func_commonsense_constraints
from evaluation.hard_constraint import (
    get_symbolic_concepts,
    calc_cost_from_itinerary_wo_intercity,
)
from evaluation.hard_constraint import (
    evaluate_constraints as evaluate_logical_constraints,
)
import time
import datetime
import llms
from llms import deepseek, deepseek_json, deepseek_poi, model_GPT_poi
import pandas as pd
import numpy as np
from NS_agent_inter.retrieval import Retriever
import random

random.seed(0)
from evaluation.utils import (
    score_go_intercity_transport,
    score_back_intercity_transport,
    combine_transport_dataframe,
)


def time_compare_if_earlier_equal(time_1, time_2):

    time1 = float(time_1.split(":")[0]) * 60 + float(time_1.split(":")[1])
    time2 = float(time_2.split(":")[0]) * 60 + float(time_2.split(":")[1])

    return time1 <= time2


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_similarity(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def mmr_algorithm(df, name_key, lambda_value=0.3):
    selected_indices = []
    remaining_indices = list(df.index)

    tfidf_vectorizer = TfidfVectorizer()

    while len(selected_indices) < len(df):
        if len(selected_indices) == 0:
            mmr_scores = df["importance"].values
        else:
            selected_data = df.iloc[selected_indices]
            selected_names = [
                name.split()[0] for name in selected_data[name_key].values
            ]
            remaining_data = df.iloc[remaining_indices]
            remaining_names = [
                name.split()[0] for name in remaining_data[name_key].values
            ]

            tfidf_matrix = tfidf_vectorizer.fit_transform(
                np.concatenate((selected_names, remaining_names))
            )
            similarity_matrix = cosine_similarity(tfidf_matrix)

            selected_similarities = similarity_matrix[
                : len(selected_names), len(selected_names) :
            ]
            remaining_similarities = similarity_matrix[
                len(selected_names) :, len(selected_names) :
            ]

            mmr_scores = lambda_value * remaining_data["importance"].values - (
                1 - lambda_value
            ) * np.max(selected_similarities, axis=0)

        max_index = np.argmax(mmr_scores)
        selected_indices.append(remaining_indices[max_index])
        del remaining_indices[max_index]

    return df.iloc[selected_indices]


def next_time_delta(time1, time_delta):

    hour, minu = int(time1.split(":")[0]), int(time1.split(":")[1])

    min_new = int(minu / time_delta + 1) * time_delta
    if min_new >= 60:
        hour_new = hour + int(min_new / 60)
        min_new = min_new % 60
    else:
        hour_new = hour

    if hour_new < 10:
        time_new = "0" + str(hour_new) + ":"
    else:
        time_new = str(hour_new) + ":"
    if min_new < 10:

        time_new = time_new + "0" + str(min_new)
    else:
        time_new = time_new + str(min_new)

    return time_new


def add_time_delta(time1, time_delta):

    hour, minu = int(time1.split(":")[0]), int(time1.split(":")[1])

    min_new = minu + time_delta % 60
    hour_new = hour + int(min_new / 60) + int(time_delta / 60)
    min_new = min_new % 60

    time_new = f"{hour_new:02d}:{min_new:02d}"

    return time_new


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def __del__(self):
        self.log.close()


class Interactive_Search:
    def __init__(self, timeout=600, mode="gpt4", verbose=True):
        self.verbose = verbose
        self.ret = Retriever()
        if mode == "deepseek":
            self.llm_model = deepseek_poi()
        elif mode == "deepseek_json":
            self.llm_model = deepseek_json()
        elif mode == "gpt4":
            self.llm_model = model_GPT_poi()

        self.TIME_CUT = timeout

        self.accommodation = Accommodations()
        self.restaurants = Restaurants()
        self.attractions = Attractions()
        self.intercity_transport = IntercityTransport()

    def symbolic_search(self, query, query_idx=0):
        if self.verbose:
            print(f"[SYMBOLIC_SEARCH] 开始符号搜索，查询索引: {query_idx}")
            print(f"[SYMBOLIC_SEARCH] 查询内容: {query}")
        self.avialable_plan = {}
        query["transport_preference"] = "metro"
        query["hard_logic_unseen"] = []
        if self.verbose:
            print(f"[SYMBOLIC_SEARCH] 硬约束条件: {query['hard_logic']}")
        #         **支持的约束类型**:
        # - `rooms==N`: 房间数量
        # - `cost<=N`: 总预算限制
        # - `room_type==N`: 房间类型
        # - `train_type=='XXX'`: 火车类型
        # - `{'川菜', '粤菜'} <= food_type`: 餐饮类型偏好
        # - `transport_type <= {'taxi'}`: 交通方式偏好
        # - `hotel_price<=N`: 酒店价格限制
        # - `intercity_transport=='train'`: 城际交通类型
        # - `{'文化遗址'} <= spot_type`: 景点类型偏好
        # - `{'天安门'} <= attraction_names`: 特定景点名称
        # - `{'全聚德'} <= restaurant_names`: 特定餐厅名称
        # - `{'北京饭店'} <= hotel_names`: 特定酒店名称
        # - `{'温泉'} <= hotel_feature`: 酒店特性需求
        for idx, item in enumerate(query["soft_logic"]):  # 这里开始解析每个约束
            if item.startswith("rooms=="):

                # print(item)
                query["rooms"] = int(item.split("==")[1])

            elif item.startswith("cost<="):
                query["cost"] = int(item.split("<=")[1])

            elif item.startswith("room_type=="):
                query["room_type"] = int(item.split("==")[1])

            elif item.startswith("train_type == "):
                query["train_type"] = item.split("'")[1]

            elif item.endswith(" <= food_type") or item.endswith("food_type"):
                ftlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in ftlist.split(","):
                    str_i = s_i.split("'")[1]  # 提取想吃的菜系
                    seen_list.append(str_i)

                if len(seen_list) > 0:
                    query["food_type"] = set(seen_list)
                    # query["hard_logic"][idx] = str(query["food_type"]) + "<=food_type"
                else:
                    # query["hard_logic"][idx] = " 3 < 33"
                    pass
            # 解析交通方式约束: transport_type <= {'taxi'} 或 transport_type == {'metro'}
            elif item.startswith("transport_type"):
                if ("<=") in item:
                    str_set = item.split("transport_type<=")[1]
                elif ("==") in item:
                    str_set = item.split("transport_type==")[1]

                if "taxi" in str_set:
                    query["transport_preference"] = "taxi"
                elif "metro" in str_set:
                    query["transport_preference"] = "metro"
                else:
                    query["transport_preference"] = "walk"

            # 解析酒店价格约束: hotel_price<=N
            elif item.startswith("hotel_price<="):
                query["hotel_price"] = int(item.split("<=")[1])

            # 解析城际交通类型约束: intercity_transport == 'train'
            elif item.endswith("intercity_transport") or item.startswith(
                "intercity_transport"
            ):
                query["intercity_transport_type"] = item.split("'")[1]

            # 解析景点类型约束: {'文化遗址', '自然景观'} <= spot_type
            elif item.endswith("spot_type"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)

                if len(seen_list) > 0:
                    query["spot_type"] = set(seen_list)
                    # query["hard_logic"][idx] = str(query["spot_type"]) + "<=spot_type"
                else:
                    # query["hard_logic"][idx] = " 3 < 33"
                    pass

            # 解析特定景点名称约束: {'天安门', '故宫'} <= attraction_names
            elif item.endswith("attraction_names"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["attraction_names"] = set(seen_list)
                    # query["hard_logic"][idx] = (
                    #     str(query["attraction_names"]) + "<=attraction_names"
                    # )
                else:
                    # query["hard_logic"][idx] = " 3 < 33"
                    pass

            # 解析特定餐厅名称约束: {'全聚德', '东来顺'} <= restaurant_names
            elif item.endswith("restaurant_names"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["restaurant_names"] = set(seen_list)
                    # query["hard_logic"][idx] = (
                    #     str(query["restaurant_names"]) + "<=restaurant_names"
                    # )
                else:
                    # query["hard_logic"][idx] = " 3 < 33"
                    pass

            # 解析特定酒店名称约束: {'北京饭店', '王府井大饭店'} <= hotel_names
            elif item.endswith("hotel_names"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["hotel_names"] = set(seen_list)
                    # query["hard_logic"][idx] = (
                    #     str(query["hotel_names"]) + "<=hotel_names"
                    # )
                else:
                    # query["hard_logic"][idx] = " 3 < 33"
                    pass

            # 解析酒店特性约束: {'温泉', '游泳池'} <= hotel_feature
            elif item.endswith("hotel_feature"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["hotel_feature"] = set(seen_list)
                    # query["hard_logic"][idx] = (
                    #     str(query["hotel_feature"]) + "<=hotel_feature"
                    # )
                else:
                    # query["hard_logic"][idx] = " 3 < 33"
                    pass

            # 其他无法识别的约束
            else:
                if (
                    "days==" not in item
                    and "people_number==" not in item
                    and "tickets==" not in item
                ):
                    query["hard_logic_unseen"].append(item)

        # 过滤掉个性化约束
        query["hard_logic"] = [
            item
            for item in query["hard_logic"]
            if item not in query["hard_logic_unseen"]
        ]

        # 开始搜索计划
        success, plan = self.search_plan(query)
        if self.verbose:
            print(f"[SYMBOLIC_SEARCH]line 348 搜索完成 - 成功: {success}, 计划: {plan}")
        return success, plan

    def search_plan(self, query):
        """
        搜索和生成完整的旅行计划

        这个函数负责：
        1. 搜索城际交通（飞机、火车）
        2. 为交通选项评分和排序
        3. 选择最优的往返交通组合
        4. 调用POI搜索生成详细行程

        Args:
            query (dict): 解析后的用户查询字典

        Returns:
            tuple: (success, plan) - 成功标志和生成的计划
        """
        if self.verbose:
            print(f"[SEARCH_PLAN] 开始规划搜索")
            print(f"[SEARCH_PLAN] 查询参数: {query}")

        poi_plan = {}  # POI计划存储
        self.poi_info = {}  # POI信息存储

        # 初始化访问记录列表（避免重复访问）
        self.restaurants_visiting = []  # 已访问餐厅索引
        self.attractions_visiting = []  # 已访问景点索引
        self.food_type_visiting = []  # 已尝试菜系类型
        self.spot_type_visiting = []  # 已访问景点类型
        self.attraction_names_visiting = []  # 已访问景点名称
        self.restaurant_names_visiting = []  # 已访问餐厅名称

        source_city = query["start_city"]  # 出发城市
        target_city = query["target_city"]  # 目标城市

        # 搜索去程火车
        train_go = self.intercity_transport.select(
            start_city=source_city, end_city=target_city, intercity_type="train"
        )
        # 搜索回程火车
        train_back = self.intercity_transport.select(
            start_city=target_city, end_city=source_city, intercity_type="train"
        )

        # 搜索去程航班
        flight_go = self.intercity_transport.select(
            start_city=source_city, end_city=target_city, intercity_type="airplane"
        )
        # 搜索回程航班
        flight_back = self.intercity_transport.select(
            start_city=target_city, end_city=source_city, intercity_type="airplane"
        )

        if flight_go is not None and not flight_go.empty:
            flight_go["Score"] = flight_go.apply(score_go_intercity_transport, axis=1)
        else:
            flight_go = pd.DataFrame(
                columns=[
                    "FlightID",
                    "From",
                    "To",
                    "BeginTime",
                    "EndTime",
                    "Duration",
                    "Cost",
                    "Score",
                ]
            )

        if flight_back is not None and not flight_back.empty:
            flight_back["Score"] = flight_back.apply(
                score_back_intercity_transport, axis=1
            )
        else:
            flight_back = pd.DataFrame(
                columns=[
                    "FlightID",
                    "From",
                    "To",
                    "BeginTime",
                    "EndTime",
                    "Duration",
                    "Cost",
                    "Score",
                ]
            )

        if train_go is not None and not train_go.empty:
            train_go["Score"] = train_go.apply(score_go_intercity_transport, axis=1)
        else:
            train_go = pd.DataFrame(
                columns=[
                    "TrainID",
                    "TrainType",
                    "From",
                    "To",
                    "BeginTime",
                    "EndTime",
                    "Duration",
                    "Cost",
                    "Score",
                ]
            )

        if train_back is not None and not train_back.empty:
            train_back["Score"] = train_back.apply(
                score_back_intercity_transport, axis=1
            )
        else:
            train_back = pd.DataFrame(
                columns=[
                    "TrainID",
                    "TrainType",
                    "From",
                    "To",
                    "BeginTime",
                    "EndTime",
                    "Duration",
                    "Cost",
                    "Score",
                ]
            )

        # 合并飞机和火车的选项
        go_transport = combine_transport_dataframe(flight_go, train_go)
        back_transport = combine_transport_dataframe(flight_back, train_back)

        # 按评分升序排列（分数越低越好）
        go_transport = go_transport.sort_values(by="Score", ascending=True)
        back_transport = back_transport.sort_values(by="Score", ascending=True)

        # flight_go = flight_go.sort_values(by="Score", ascending=True)
        # flight_back = flight_back.sort_values(by="Score", ascending=True)
        # train_go = train_go.sort_values(by="Score", ascending=True)
        # train_back = train_back.sort_values(by="Score", ascending=True)

        # print(flight_go)
        # print(flight_back)
        # print(train_go)
        # print(train_back)
        # print(flight_go.iloc[0])
        # exit()

        flight_go_num = 0 if flight_go is None else flight_go.shape[0]
        train_go_num = 0 if train_go is None else train_go.shape[0]
        flight_back_num = 0 if flight_back is None else flight_back.shape[0]
        train_back_num = 0 if train_back is None else train_back.shape[0]

        if self.verbose:
            print(f"[TRANSPORT_SEARCH] 交通搜索结果:")
            print(
                f"  从 {source_city} 到 {target_city}: {flight_go_num} 个航班, {train_go_num} 个火车班次"
            )
            print(
                f"  从 {target_city} 到 {source_city}: {flight_back_num} 个航班, {train_back_num} 个火车班次"
            )

        poi_plan["transport_preference"] = query["transport_preference"]

        found_intercity_transport = False

        for go_i in range(go_transport.shape[0]):
            if found_intercity_transport:
                break

            if go_transport.iloc[go_i]["Type"] == "飞机":
                poi_plan["go_transport"] = go_transport.iloc[go_i]

                if (
                    "intercity_transport_type" in query
                    and query["intercity_transport_type"] != "airplane"
                ):
                    continue
                if "train_type" in query:
                    continue
            else:
                poi_plan["go_transport"] = go_transport.iloc[go_i]

                if (
                    "intercity_transport_type" in query
                    and query["intercity_transport_type"] != "train"
                ):
                    continue

                # print(poi_plan["go_transport"])
                # train_id[0]
                # exit(0)
                if (
                    "train_type" in query
                    and query["train_type"] != poi_plan["go_transport"]["Type"]
                ):
                    continue

            for back_i in range(back_transport.shape[0]):
                if found_intercity_transport:
                    break

                # print("go idx: ", go_i, "back id: ", back_i)

                if back_transport.iloc[back_i]["Type"] == "飞机":
                    poi_plan["back_transport"] = back_transport.iloc[back_i]

                    if (
                        "intercity_transport_type" in query
                        and query["intercity_transport_type"] != "train"
                    ):
                        continue

                    if (
                        "train_type" in query
                        and query["train_type"] != poi_plan["back_transport"]["Type"]
                    ):
                        continue
                else:
                    poi_plan["back_transport"] = back_transport.iloc[back_i]

                    if (
                        "intercity_transport_type" in query
                        and query["intercity_transport_type"] != "airplane"
                    ):
                        continue

                    if "train_type" in query:
                        continue

                # print(poi_plan)

                if "cost" in query:

                    intercity_cost = (
                        poi_plan["go_transport"]["Cost"]
                        + poi_plan["back_transport"]["Cost"]
                    ) * query["people_number"]

                    if intercity_cost >= query["cost"]:
                        continue

                    else:
                        cost_wo_inter_trans = query["cost"] - intercity_cost
                        found_intercity_transport = True
                        if self.verbose:
                            print(
                                f"[COST_CHECK] 城际交通费用: {intercity_cost}, 剩余预算: {cost_wo_inter_trans}"
                            )
                else:
                    found_intercity_transport = True

        success, plan = self.search_poi(
            query, poi_plan, plan=[], current_time="", current_position=""
        )

        if self.verbose:
            print(f"[SEARCH_PLAN] POI搜索完成 - 成功: {success}")
        if success:
            return True, plan
        return False, {"info": "No Solution"}

    def extract_score_from_plan(self, chart, name_list):
        """从chart中提取分数"""

        score_list = [0 for _ in range(len(name_list))]

        # 将表格按行分割
        lines = chart.split("\n")
        table_lines = [
            line.strip() for line in lines if "|" in line and not line.startswith("|--")
        ]

        for i, name in enumerate(name_list):
            # 尝试多种匹配方式
            found_score = False

            # 方式1：精确匹配
            for line in table_lines:
                if re.search(rf"\|\s*{re.escape(name)}\s*\|\s*(\d+)\s*\|", line):
                    match = re.search(
                        rf"\|\s*{re.escape(name)}\s*\|\s*(\d+)\s*\|", line
                    )
                    if match:
                        score_list[i] = int(match.group(1))
                        found_score = True
                        if self.verbose:
                            print(
                                f"[SCORE_EXTRACTION] 精确匹配 '{name}' -> {score_list[i]}"
                            )
                        break

            # 方式2：部分匹配（如果精确匹配失败）
            if not found_score:
                for line in table_lines:
                    # 尝试匹配包含目标名字的行
                    if name in line:
                        match = re.search(r"\|\s*([^|]+)\s*\|\s*(\d+)\s*\|", line)
                        if match:
                            table_name = match.group(1).strip()
                            score = int(match.group(2))
                            # 检查是否是部分匹配
                            if name in table_name or table_name in name:
                                score_list[i] = score
                                found_score = True
                                if self.verbose:
                                    print(
                                        f"[SCORE_EXTRACTION] 部分匹配 '{name}' <- '{table_name}' -> {score}"
                                    )
                                break

            # 方式3：模糊匹配（基于相似度）
            if not found_score:
                best_match_score = 0
                best_similarity = 0

                for line in table_lines:
                    match = re.search(r"\|\s*([^|]+)\s*\|\s*(\d+)\s*\|", line)
                    if match:
                        table_name = match.group(1).strip()
                        score = int(match.group(2))

                        # 计算相似度（简单的字符串包含检查）
                        similarity = 0
                        name_chars = set(name)
                        table_chars = set(table_name)
                        if name_chars & table_chars:  # 有共同字符
                            similarity = len(name_chars & table_chars) / max(
                                len(name_chars), len(table_chars)
                            )

                        if (
                            similarity > best_similarity and similarity > 0.3
                        ):  # 相似度阈值
                            best_similarity = similarity
                            best_match_score = score

                if best_similarity > 0.85:
                    score_list[i] = best_match_score
                    found_score = True
                    if self.verbose:
                        print(
                            f"[SCORE_EXTRACTION] 模糊匹配 '{name}' -> {best_match_score} (相似度: {best_similarity:.2f})"
                        )

            if not found_score and self.verbose:
                print(f"[SCORE_EXTRACTION] 未找到匹配 '{name}' -> 默认分数 0")

        if self.verbose:
            print(f"[SCORE_EXTRACTION] 最终评分列表: {score_list}")

        return score_list

    def extract_name_list(self, poi_info_list):
        """从poi_info_list中提取名字列表"""

        name_list = ["" for _ in range(len(poi_info_list))]
        for i, rest in enumerate(poi_info_list):
            match = re.search(r"^.*name: (.*) , .*$", rest)
            if match:
                name = match.group(1)
                # print("name: ", name)
                name_list[i] = name
        return name_list

    def score_poi_think_overall_act_page(
        self,
        planning_info,  # 提示词
        poi_info_list,  # 景点信息
        need_db=False,
        react=False,
        history_message=[],
    ):
        # 添加输入验证
        if not poi_info_list or len(poi_info_list) == 0:
            if self.verbose:
                print("[POI_SCORING] 警告: poi_info_list 为空，返回空评分列表")
            return []

        info_list = []
        # new_poi_info=poi_info_list[0:30]
        new_poi_info = []

        for p_i in planning_info:  # 用户信息偏好或规划
            info_list.append(p_i)

        info_list.append(
            f"以下是可以选择的景点/餐厅, 请为其中的每个景点/餐厅排序打分, 并输出为表格. 表格格式为: 景点/餐厅名字 | 分数. 请简洁的回复, 不要输出多余内容. 评分满分100分, 越满足用户要求分越高。为了满足多样性,之前的行程中出现的景点/餐厅打0,与之前行程相似的景点/餐厅打低分: "
        )

        # if need_db:
        #     new_poi_info = random.choices(poi_info_list, k=min(25, len(poi_info_list)))
        for item in poi_info_list:
            info_list.append(item)

        if self.verbose:
            print(
                "-------------------------------------------------------------------------"
            )
            print(f"[POI_SCORING] poi_info_list 长度: {len(poi_info_list)}")
            print("poi_info_list前三条: ", poi_info_list[0:3])

        overall_plan, history_message_think_overall = self.reason_prompt(
            info_list, return_history_message=True, history_message=[]
        )

        name_list = self.extract_name_list(poi_info_list)
        score_list = self.extract_score_from_plan(overall_plan, name_list)

        if self.verbose:
            print(f"[POI_SCORING] 提取的名字列表: {name_list}")
            print(f"[POI_SCORING] 评分列表: {score_list}")

        # print("score_list: ", score_list)
        # print("poi_info_list: ", poi_info_list)
        # exit(0)

        # # rewrite the plan
        # rewrite_ans = self.rewrite_plan(overall_plan)

        # querys = rewrite_ans.split(", ")
        # scores = []
        # for q in querys:
        #     score = self.ret.get_score(poi_info_list, q)
        #     scores.append(score)

        # scores = np.array(scores)
        # score_list = np.max(scores, axis=0)

        if self.verbose:
            print("[POI_SCORING] POI评分完成")

        return score_list

    def reason_prompt(self, info_list, return_history_message=True, history_message=[]):

        scratchpad = ""

        for info_i in info_list:
            scratchpad = scratchpad + info_i + "\n"

        # scratchpad += f"请将以上餐厅或景点进行排序打分, 并输出为表格. 表格格式为: 景点/餐厅名字 | 分数. 请简洁的回复, 不要输出多余内容."

        json_scratchpad = history_message
        json_scratchpad.append({"role": "user", "content": scratchpad})

        if self.verbose:
            print(f"[LLM_QUERY] 发送查询: {json_scratchpad}")
        thought = self.llm_model(json_scratchpad)
        scratchpad = scratchpad + " " + thought
        json_scratchpad.append({"role": "assistant", "content": thought})

        if self.verbose:
            print(f"[LLM_RESPONSE] 回答: {thought}")

        if return_history_message:
            return thought, json_scratchpad
        return thought

    def rewrite_plan(self, query):

        scratchpad = (
            query + "\n请将上述内容中涉及到的名字（景点、餐厅）提取出来，并用逗号隔开。"
        )

        scratchpad += f"Answer:"

        json_scratchpad = []
        json_scratchpad.append({"role": "user", "content": scratchpad})

        if self.verbose:
            print(f"[REWRITE_PLAN] 发送查询: {json_scratchpad}")
        thought = self.llm_model(json_scratchpad)

        if self.verbose:
            print(f"[REWRITE_PLAN] 回答: {thought}")
        return thought

    def search_poi(
        self, query, poi_plan, plan, current_time, current_position, current_day=0
    ):
        if self.verbose:
            print(f"[SEARCH_POI] 开始POI搜索")
            # print(f"  查询: {query}")
            print(
                f"  当前时间: {current_time}, 位置: {current_position}, 天数: {current_day}"
            )
            # print(f"  当前计划: {plan}")
            # print(f"  POI计划: {poi_plan}")

        target_city = query["target_city"]
        if "cost_wo_intercity" in query:
            inner_city_cost = calc_cost_from_itinerary_wo_intercity(
                plan, query["people_number"]
            )

            if inner_city_cost >= query["cost_wo_intercity"]:

                if self.verbose:
                    print(
                        f"[BUDGET_CHECK] 预算不足 - 内城预算: {query['cost_wo_intercity']}, 当前花费: {inner_city_cost}"
                    )

                return False, plan

        if current_time != "" and time_compare_if_earlier_equal("23:00", current_time):
            if self.verbose:
                print("[TIME_CHECK] 时间过晚，超过23:00")
            return False, plan

        if current_time != "" and current_day == query["days"] - 1:
            # We should go back in time ...
            transports_sel = goto(
                city=query["target_city"],
                start=current_position,
                end=poi_plan["back_transport"]["From"],
                start_time=current_time,
                method=poi_plan["transport_preference"],
                verbose=True,
            )
            arrived_time = transports_sel[-1]["end_time"]
            """
            print("arrived_time",arrived_time,poi_plan["back_transport"]["BeginTime"])
            if time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], arrived_time):
                if self.verbose:
                    print("Can not go back source-city in time, current POI {}, station arrived time: {}".format(current_position, arrived_time))
                return False, plan
            """
        elif current_time != "":
            # 搜索酒店的逻辑
            keywords = "酒店"
            search_query = ""
            hotel_info = None  # 初始化 hotel_info 变量

            if self.verbose:
                print(f"[HOTEL_SEARCH] 开始搜索酒店")
                print(f"  目标城市: {target_city}")
                print(f"  查询条件: {query}")

            # 1. 处理酒店特性
            if "hotel_feature" in query:
                # 把特性添加到关键字前面
                features = query["hotel_feature"]
                if isinstance(features, (list, tuple, set)) and features:
                    for feature in features:
                        keywords = f"{feature}{keywords}"
                    if self.verbose:
                        print(f"  添加酒店特性: {features}")

            # 2. 处理酒店名称
            if "hotel_names" in query and query["hotel_names"]:
                hotel_name = (
                    list(query["hotel_names"])[0]
                    if isinstance(query["hotel_names"], (list, tuple, set))
                    else query["hotel_names"]
                )
                search_query += f" 酒店名字为{hotel_name}"
                if self.verbose:
                    print(f"  指定酒店名称: {hotel_name}")

            # 3. 处理价格筛选
            if "hotel_price" in query:
                price = query["hotel_price"]
                search_query += f" 价格为{price}"
                if self.verbose:
                    print(f"  价格限制: {price}")

            # 组合最终搜索关键词
            final_keywords = f"{keywords} {search_query}".strip()
            if self.verbose:
                print(f"  最终搜索关键词: '{final_keywords}'")

            # 执行酒店搜索
            try:
                hotel_info = self.accommodation.select(
                    target_city, keywords=final_keywords
                )
                if self.verbose:
                    print(f"  酒店搜索结果(只输出前5条):", hotel_info[0:5])
                    if hotel_info is not None:
                        print(
                            f"  搜索结果数量: {len(hotel_info) if hasattr(hotel_info, '__len__') else 'N/A'}"
                        )
                    else:
                        print("  搜索结果为None")
            except Exception as e:
                if self.verbose:
                    print(f"  酒店搜索异常: {str(e)}")
                hotel_info = None

            # 检查酒店搜索结果是否为空
            if (
                hotel_info is None
                or (hasattr(hotel_info, "empty") and hotel_info.empty)
                or (hasattr(hotel_info, "__len__") and len(hotel_info) == 0)
            ):
                if self.verbose:
                    print(
                        f"[HOTEL_SEARCH] 警告: 在{target_city}没有找到符合条件的酒店，搜索关键词: {final_keywords}"
                    )

                # 尝试使用更宽泛的搜索条件
                backup_keywords = "酒店"
                if self.verbose:
                    print(f"[HOTEL_SEARCH] 尝试使用备用搜索关键词: {backup_keywords}")

                try:
                    hotel_info = self.accommodation.select(
                        target_city, keywords=backup_keywords
                    )
                    if self.verbose:
                        print(f"[HOTEL_SEARCH] 备用搜索结果: {hotel_info}")
                except Exception as e:
                    if self.verbose:
                        print(f"[HOTEL_SEARCH] 备用酒店搜索异常: {str(e)}")
                    hotel_info = None

                # 如果仍然没有结果，返回失败
                if (
                    hotel_info is None
                    or (hasattr(hotel_info, "empty") and hotel_info.empty)
                    or (hasattr(hotel_info, "__len__") and len(hotel_info) == 0)
                ):
                    if self.verbose:
                        print(f"[HOTEL_SEARCH] 错误: 在{target_city}没有找到任何酒店")
                    return False, plan

            hotel_sel = hotel_info.iloc[0]
            poi_plan["accommodation"] = hotel_info.iloc[0]

            if self.verbose:
                print(f"[HOTEL_SEARCH] 选择的酒店: {hotel_sel.get('name', 'Unknown')}")

            transports_sel = goto(
                city=query["target_city"],
                start=current_position,
                end=hotel_sel["name"],
                start_time=current_time,
                method=poi_plan["transport_preference"],
                verbose=False,
            )
            arrived_time = transports_sel[-1]["end_time"]
            arrived_time = "23:00"
            if time_compare_if_earlier_equal("24:00", arrived_time):
                if self.verbose:
                    print(
                        f"[HOTEL_CHECK] 无法及时到达酒店 - 当前位置: {current_position}, 酒店到达时间: {arrived_time}"
                    )
                return False, plan

        # intercity_transport - go
        if current_day == 0 and current_time == "":
            plan = [{"day": current_day + 1, "activities": []}]

            plan[current_day]["activities"].append(
                {
                    "start_time": poi_plan["go_transport"]["BeginTime"],
                    "end_time": poi_plan["go_transport"]["EndTime"],
                    "start": poi_plan["go_transport"]["From"],
                    "end": poi_plan["go_transport"]["To"],
                    "ID": poi_plan["go_transport"]["ID"],
                    "type": poi_plan["go_transport"]["Type"],
                    "transports": [],
                    "cost": poi_plan["go_transport"]["Cost"],
                    "tickets": query["people_number"],
                }
            )

            new_time = poi_plan["go_transport"]["EndTime"]
            new_position = poi_plan["go_transport"]["To"]
            success, plan = self.search_poi(
                query, poi_plan, plan, new_time, new_position, current_day
            )
            if success:
                return True, plan

            else:
                if self.verbose:
                    print("[GO_TRANSPORT] 给定去程交通无解决方案")
                return False, plan

        # breakfast

        if current_time == "00:00":  # 新的一天开始了
            if self.verbose:
                print("[NEW_DAY] 新的一天开始")
            if len(plan) < current_day + 1:
                plan.append({"day": current_day + 1, "activities": []})

            # breakat at hotel
            plan[current_day]["activities"].append(
                {
                    "position": poi_plan["accommodation"]["name"],
                    "type": "breakfast",
                    "transports": [],
                    "cost": 0,
                    "start_time": "08:00",
                    "end_time": "08:30",
                }
            )

            new_time = plan[current_day]["activities"][-1]["end_time"]
            if self.verbose:
                print(f"[NEW_DAY] 早餐结束时间: {new_time}")
            new_position = current_position
            success, plan = self.search_poi(
                query, poi_plan, plan, new_time, new_position, current_day
            )
            if success:
                return True, plan

            plan[current_day]["activities"].pop()
        haved_lunch_today, haved_dinner_today = False, False

        for act_i in plan[current_day]["activities"]:
            if act_i["type"] == "lunch":
                haved_lunch_today = True
            if act_i["type"] == "dinner":
                haved_dinner_today = True
        candidates_type = ["attraction"]
        if (
            not haved_lunch_today
        ):  # and time_compare_if_earlier_equal(current_time, "12:30"):
            candidates_type.append("lunch")
        if (
            not haved_dinner_today
        ):  #  and time_compare_if_earlier_equal(current_time, "18:30"):
            candidates_type.append("dinner")
        if ("accommodation" in poi_plan) and (current_day < query["days"] - 1):
            candidates_type.append("hotel")

        if (
            current_day == query["days"] - 1 and current_time != ""
        ):  # and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], add_time_delta(current_time, 180)):
            candidates_type.append("back-intercity-transport")

        while len(candidates_type) > 0:

            poi_type = self.get_poi_type_from_time_sym(
                current_time, candidates_type, poi_plan["back_transport"]["BeginTime"]
            )

            if self.verbose:
                print(
                    f"[POI_PLANNING] 第{current_day + 1}天 {current_time}, 位置: {current_position}, 下一个POI类型: {poi_type}"
                )

            if poi_type == "back-intercity-transport":  # 回家家咯
                if len(plan) < current_day + 1:
                    plan.append({"day": current_day + 1, "activities": []})

                transports_sel = goto(
                    city=query["target_city"],
                    start=current_position,
                    end=poi_plan["back_transport"]["From"],
                    start_time=current_time,
                    method=poi_plan["transport_preference"],
                    verbose=False,
                )
                if len(transports_sel) == 3:
                    transports_sel[1]["tickets"] = query["people_number"]
                elif transports_sel[0]["mode"] == "taxi":
                    transports_sel[0]["car"] = int((query["people_number"] - 1) / 4) + 1

                if poi_plan["back_transport"]["Type"] == "train":
                    plan[current_day]["activities"].append(
                        {
                            "start_time": poi_plan["back_transport"]["BeginTime"],
                            "end_time": poi_plan["back_transport"]["EndTime"],
                            "start": poi_plan["back_transport"]["From"],
                            "end": poi_plan["back_transport"]["To"],
                            "ID": poi_plan["back_transport"]["ID"],
                            "type": "train",
                            "transports": transports_sel,
                            "cost": poi_plan["back_transport"]["Cost"],
                            "tickets": query["people_number"],
                        }
                    )
                else:
                    plan[current_day]["activities"].append(
                        {
                            "start_time": poi_plan["back_transport"]["BeginTime"],
                            "end_time": poi_plan["back_transport"]["EndTime"],
                            "start": poi_plan["back_transport"]["From"],
                            "end": poi_plan["back_transport"]["To"],
                            "ID": poi_plan["back_transport"]["ID"],
                            "type": "airplane",
                            "transports": transports_sel,
                            "cost": poi_plan["back_transport"]["Cost"],
                            "tickets": query["people_number"],
                        }
                    )
                res_bool, res_plan = self.constraints_validation(query, plan, poi_plan)
                if self.verbose:
                    print(f"[BACK_TRANSPORT] 约束验证结果: {res_plan}")
                # return True, res_plan  # TODO 先不检查约束了，成功回家要紧
                if res_bool:
                    return True, res_plan
                else:

                    plan[current_day]["activities"].pop()

                    if self.verbose:
                        print(
                            "[BACK_TRANSPORT] 选择回程交通并完成行程，但约束验证失败..."
                        )

            elif poi_type in ["lunch", "dinner"]:
                keywords = f"{current_position}" + "附近美食"  # 吃附近的餐厅
                if self.verbose:
                    print(
                        f"[RESTAURANT_SEARCH] 开始搜索餐厅 - 关键词: {keywords}, 城市: {target_city}"
                    )

                self.poi_info["restaurants"] = self.restaurants.select(
                    target_city, keywords
                )
                search_query = ""
                info_list = [query["nature_language"]]
                info_list.append(
                    f"以下是计划中已有的行程: {self.extract_location_from_plan(plan)}"
                )
                info_list.append(
                    "在这次在{}的旅行中,请帮我选择一些餐厅去吃，评估每个餐厅的相关性，看看是否需要安排到行程里".format(
                        query["target_city"]
                    )
                )
                rest_info = self.poi_info["restaurants"]

                # 添加调试信息
                if self.verbose:
                    print(f"[RESTAURANT_SEARCH] 餐厅搜索原始结果: {type(rest_info)}")
                    if rest_info is not None:
                        print(
                            f"[RESTAURANT_SEARCH] 餐厅数量: {len(rest_info) if hasattr(rest_info, '__len__') else 'N/A'}"
                        )
                        if hasattr(rest_info, "empty"):
                            print(f"[RESTAURANT_SEARCH] 是否为空: {rest_info.empty}")
                    else:
                        print("[RESTAURANT_SEARCH] 餐厅搜索结果为 None")

                # 检查搜索结果是否为空
                if (
                    rest_info is None
                    or (hasattr(rest_info, "empty") and rest_info.empty)
                    or (hasattr(rest_info, "__len__") and len(rest_info) == 0)
                ):
                    if self.verbose:
                        print(
                            f"[RESTAURANT_SEARCH] 警告: 在{target_city}没有找到餐厅，尝试更宽泛的搜索"
                        )

                    # 尝试使用更宽泛的搜索条件
                    backup_keywords = "餐厅"
                    self.poi_info["restaurants"] = self.restaurants.select(
                        target_city, backup_keywords
                    )
                    rest_info = self.poi_info["restaurants"]

                    if self.verbose:
                        print(f"[RESTAURANT_SEARCH] 备用搜索结果: {type(rest_info)}")
                        if rest_info is not None and hasattr(rest_info, "__len__"):
                            print(
                                f"[RESTAURANT_SEARCH] 备用搜索餐厅数量: {len(rest_info)}"
                            )

                    # 如果仍然没有结果，跳过这个POI类型
                    if (
                        rest_info is None
                        or (hasattr(rest_info, "empty") and rest_info.empty)
                        or (hasattr(rest_info, "__len__") and len(rest_info) == 0)
                    ):
                        if self.verbose:
                            print(
                                f"[RESTAURANT_SEARCH] 错误: 在{target_city}没有找到任何餐厅，跳过此POI类型"
                            )
                        continue

                if "food_type" in query:
                    req_food_type = set()
                    req_food_type = set.union(req_food_type, set(query["food_type"]))
                    info_list.append(
                        "本次旅行有对餐饮类型的需求，这些类型至少需要吃一次 {}, 在选择餐厅的时候请根据餐厅的餐饮类型加以考虑".format(
                            req_food_type
                        )
                    )
                if "restaurant_names" in query:
                    req_res_name = set()
                    req_res_name = set.union(
                        req_res_name, set(query["restaurant_names"])
                    )
                    info_list.append(
                        "本次旅行有对特定餐饮商家的需求，这几家至少需要去一次 {}, 在选择餐厅的时候请根据餐厅的名字加以考虑".format(
                            req_res_name
                        )
                    )

                # info_list.append("请根据以上信息, 从以下餐厅信息中选出最合适的餐厅. ")

                poi_info_list = []
                score_list = []

                # 确保 rest_info 不为空且有数据
                if rest_info is not None and len(rest_info) > 0:
                    for idx in range(len(rest_info)):
                        try:
                            res_i = self.poi_info["restaurants"].iloc[idx]

                            poi_info_list.append(
                                "**{}** name: {} , cusine: {},price_per_preson: {},recommended food: {}".format(
                                    idx,
                                    res_i["name"],
                                    res_i["cuisine"],
                                    res_i["price"],
                                    res_i["recommendedfood"],
                                )
                            )
                        except Exception as e:
                            if self.verbose:
                                print(
                                    f"[RESTAURANT_SEARCH] 处理餐厅数据时出错 (索引 {idx}): {e}"
                                )
                            continue
                else:
                    if self.verbose:
                        print(
                            "[RESTAURANT_SEARCH] 餐厅数据为空，无法构建 poi_info_list"
                        )
                    continue

                if self.verbose:
                    print(
                        f"[RESTAURANT_SEARCH] 构建的 poi_info_list 长度: {len(poi_info_list)}"
                    )
                print("poi_info_list前三条: ", poi_info_list[0:3])

                # 只有当 poi_info_list 不为空时才进行评分
                if len(poi_info_list) > 0:
                    score_list = self.score_poi_think_overall_act_page(
                        info_list, poi_info_list, need_db=True, react=True
                    )
                    rest_info["importance"] = score_list
                    rest_info = rest_info.sort_values(
                        by="importance", ascending=False, ignore_index=True
                    )
                else:
                    if self.verbose:
                        print("[RESTAURANT_SEARCH] poi_info_list 为空，跳过此POI类型")
                    continue

                # rest_info = mmr_algorithm(name_key="name", df=rest_info)
                if self.verbose:
                    print(
                        f"[RESTAURANT_SEARCH] 评分后餐厅(只输出前3名): {rest_info[0:3]}"
                    )
                self.poi_info["restaurants"] = rest_info

                score_list = rest_info["importance"].values

                ranking_idx = list(range(len(score_list)))

                if poi_type == "lunch" and time_compare_if_earlier_equal(
                    "12:30", current_time
                ):
                    pass
                elif poi_type == "dinner" and time_compare_if_earlier_equal(
                    "19:30", current_time
                ):
                    pass
                else:
                    for r_i in ranking_idx:

                        res_idx = r_i
                        # print(ranking_idx, r_i)
                        # print("visiting: ", self.restaurants_visiting, res_idx)

                        if not (res_idx in self.restaurants_visiting):
                            poi_sel = rest_info.iloc[res_idx]

                            transports_sel = goto(
                                city=query["target_city"],
                                start=current_position,
                                end=poi_sel["name"],
                                start_time=current_time,
                                method=poi_plan["transport_preference"],
                                verbose=False,
                            )

                            if transports_sel[0]["mode"] == "metro":
                                transports_sel[0]["tickets"] = query["people_number"]
                            elif transports_sel[0]["mode"] == "taxi":
                                transports_sel[0]["car"] = (
                                    int((query["people_number"] - 1) / 4) + 1
                                )

                            arrived_time = transports_sel[-1]["end_time"]
                            if self.verbose:
                                print(f"[RESTAURANT_TIME] 到达时间: {arrived_time}")
                            # 开放时间
                            # TODO:从高德api返还的字符串提取出该餐厅开放时间和关闭时间
                            # opentime, endtime = poi_sel["weekdayopentime"],  poi_sel["weekdayclosetime"]
                            opentime = "08:00"
                            endtime = "22:00"
                            # it is closed ...
                            if time_compare_if_earlier_equal(endtime, arrived_time):
                                continue
                            if time_compare_if_earlier_equal(arrived_time, opentime):
                                act_start_time = opentime
                            else:
                                act_start_time = arrived_time

                            if poi_type == "lunch" and time_compare_if_earlier_equal(
                                act_start_time, "11:00"
                            ):
                                act_start_time = "11:00"

                            if poi_type == "lunch" and time_compare_if_earlier_equal(
                                endtime, "11:00"
                            ):
                                continue

                            if poi_type == "dinner" and time_compare_if_earlier_equal(
                                act_start_time, "17:00"
                            ):
                                act_start_time = "17:00"

                            if poi_type == "dinner" and time_compare_if_earlier_equal(
                                endtime, "17:00"
                            ):
                                continue

                            act_end_time = add_time_delta(act_start_time, 50)
                            if time_compare_if_earlier_equal(endtime, act_end_time):
                                act_end_time = endtime

                            if poi_type == "lunch" and time_compare_if_earlier_equal(
                                "13:00", act_start_time
                            ):
                                continue
                            if poi_type == "dinner" and time_compare_if_earlier_equal(
                                "20:00", act_start_time
                            ):
                                continue

                            if time_compare_if_earlier_equal(
                                act_end_time, act_start_time
                            ):
                                continue
                            if self.verbose:
                                print(
                                    f"[RESTAURANT_COST] 餐厅价格: {poi_sel['price']}, 人数: {query['people_number']}"
                                )

                            activity_i = {
                                "position": poi_sel["name"],
                                "type": poi_type,
                                "transports": transports_sel,
                                "cost": 100,
                                "start_time": act_start_time,
                                "end_time": act_end_time,
                                "photos": (
                                    poi_sel["photos"][0]
                                    if len(poi_sel["photos"]) > 0
                                    else ""
                                ),
                            }

                            plan[current_day]["activities"].append(activity_i)

                            new_time = act_end_time
                            new_position = poi_sel["name"]
                            if self.verbose:
                                print(
                                    f"[RESTAURANT_PLAN] 餐厅规划完成 - 时间: {new_time}, 位置: {new_position}"
                                )
                            self.restaurant_names_visiting.append(poi_sel["name"])
                            self.restaurants_visiting.append(res_idx)
                            self.food_type_visiting.append(poi_sel["cuisine"])

                            success, plan = self.search_poi(
                                query,
                                poi_plan,
                                plan,
                                new_time,
                                new_position,
                                current_day,
                            )

                            if success:
                                return True, plan

                            plan[current_day]["activities"].pop()
                            self.restaurants_visiting.pop()
                            self.food_type_visiting.pop()
                            self.restaurant_names_visiting.pop()

                            if self.verbose:
                                print(f"[RESTAURANT_FAIL] 餐厅 {poi_sel['name']} 失败")

            elif poi_type == "hotel":
                hotel_sel = poi_plan["accommodation"]

                transports_sel = goto(
                    city=query["target_city"],
                    start=current_position,
                    end=hotel_sel["name"],
                    start_time=current_time,
                    method=poi_plan["transport_preference"],
                    verbose=False,
                )
                if transports_sel[0]["mode"] == "metro":
                    transports_sel[0]["tickets"] = query["people_number"]
                elif transports_sel[0]["mode"] == "taxi":
                    transports_sel[0]["car"] = int((query["people_number"] - 1) / 4) + 1

                arrived_time = transports_sel[-1]["end_time"]
                if (
                    query["target_city"] == "北京"
                    or query["target_city"] == "上海"
                    or query["target_city"] == "广州"
                    or query["target_city"] == "深圳"
                ):
                    cost_per_room = random.randint(300, 400)
                else:
                    cost_per_room = random.randint(250, 350)

                activity_i = {
                    "position": hotel_sel["name"],
                    "type": "accommodation",
                    "room_type": 2,
                    "transports": transports_sel,
                    "cost": 100,  # //TODO 酒店价格
                    "start_time": arrived_time,
                    "end_time": "24:00",
                    "rooms": 1,  # TODO:用户需求房型
                    "photos": (
                        hotel_sel["photos"][0] if len(hotel_sel["photos"]) > 0 else ""
                    ),
                }

                plan[current_day]["activities"].append(activity_i)

                new_time = "00:00"
                new_position = hotel_sel["name"]

                success, plan = self.search_poi(
                    query, poi_plan, plan, new_time, new_position, current_day + 1
                )

                if success:
                    return True, plan

                plan[current_day]["activities"].pop()

            elif poi_type == "attraction":
                self.poi_info["attractions"] = self.attractions.select(
                    query["target_city"],
                    "旅游景点",
                    # + (str(query["spot_type"]) if "spot_type" in query else ""),
                )
                if "spot_type" in query:
                    spot_type_attractions = self.attractions.select(
                        query["target_city"],
                        "旅游景点",
                        str(query["spot_type"]),
                    )
                    if (
                        spot_type_attractions is not None
                        and not spot_type_attractions.empty
                    ):
                        # 只添加不重复条目
                        # 假设景点唯一标识为'name'字段
                        existing_names = (
                            set(self.poi_info["attractions"]["name"])
                            if not self.poi_info["attractions"].empty
                            else set()
                        )
                        new_attractions = spot_type_attractions[
                            ~spot_type_attractions["name"].isin(existing_names)
                        ]
                        if not new_attractions.empty:
                            self.poi_info["attractions"] = pd.concat(
                                [self.poi_info["attractions"], new_attractions],
                                ignore_index=True,
                            )  # TODO: 会导致大量的重复条目
                            if self.verbose:
                                print(
                                    f"[ATTRACTION_SEARCH] 添加了特定类型的景点（去重后）"
                                )

                info_list = [query["nature_language"]]
                info_list.append(
                    f"以下是计划中已有的行程: {self.extract_location_from_plan(plan)}"
                )
                info_list.append(
                    "在这次在{}的旅行中，请帮我评估每个景点的游玩重要性，看看是否需要安排到行程里".format(
                        query["target_city"]
                    )
                )
                if "spot_type" in query:
                    req_spot_type = set()
                    req_spot_type = set.union(req_spot_type, set(query["spot_type"]))
                    info_list.append(
                        "本次旅行有对景点类型的需求，这些类型至少需要去一次 {}, 在选择景点的时候请根据景点的景点类型加以考虑".format(
                            req_spot_type
                        )
                    )
                if "attraction_names" in query:
                    req_attr_name = set()
                    req_attr_name = set.union(
                        req_attr_name, set(query["attraction_names"])
                    )
                    info_list.append(
                        "本次旅行有对特定景点的需求，这几个地方至少需要去一次 {}, 在选择景点的时候请根据景点的名字加以考虑".format(
                            req_attr_name
                        )
                    )
                # info_list.append("请根据以上信息, 从以下景点信息中选出最合适的景点. ")
                attr_info = self.poi_info["attractions"]

                # 添加调试信息
                if self.verbose:
                    print(f"[ATTRACTION_SEARCH] 景点搜索原始结果: {type(attr_info)}")
                    if attr_info is not None:
                        print(
                            f"[ATTRACTION_SEARCH] 景点数量: {len(attr_info) if hasattr(attr_info, '__len__') else 'N/A'}"
                        )
                        if hasattr(attr_info, "empty"):
                            print(f"[ATTRACTION_SEARCH] 是否为空: {attr_info.empty}")
                    else:
                        print("[ATTRACTION_SEARCH] 景点搜索结果为 None")

                # 检查搜索结果是否为空
                if (
                    attr_info is None
                    or (hasattr(attr_info, "empty") and attr_info.empty)
                    or (hasattr(attr_info, "__len__") and len(attr_info) == 0)
                ):
                    if self.verbose:
                        print(
                            f"[ATTRACTION_SEARCH] 错误: 在{query['target_city']}没有找到任何景点，跳过此POI类型"
                        )
                    continue

                poi_info_list = []

                # 确保 attr_info 不为空且有数据
                if attr_info is not None and len(attr_info) > 0:
                    for idx in range(len(attr_info)):
                        try:
                            res_i = attr_info.iloc[idx]
                            poi_info_list.append(
                                "**{}** name: {} , price_per_preson: {},type: {}".format(
                                    idx, res_i["name"], res_i["price"], res_i["type"]
                                )
                            )
                        except Exception as e:
                            if self.verbose:
                                print(
                                    f"[ATTRACTION_SEARCH] 处理景点数据时出错 (索引 {idx}): {e}"
                                )
                            continue
                else:
                    if self.verbose:
                        print(
                            "[ATTRACTION_SEARCH] 景点数据为空，无法构建 poi_info_list"
                        )
                    continue

                if self.verbose:
                    print(
                        f"[ATTRACTION_SEARCH] 构建的 poi_info_list 长度: {len(poi_info_list)}"
                    )

                # 只有当 poi_info_list 不为空时才进行评分
                if len(poi_info_list) == 0:
                    if self.verbose:
                        print("[ATTRACTION_SEARCH] poi_info_list 为空，跳过此POI类型")
                    continue

                score_list = self.score_poi_think_overall_act_page(
                    info_list, poi_info_list, react=True
                )
                attr_info["importance"] = score_list

                # attr_info = mmr_algorithm(name_key="name", df=attr_info)
                attr_info = attr_info.sort_values(
                    by="importance", ascending=False, ignore_index=True
                )
                # attr_info = attr_info.sort_values(by = ["importance"], ascending=False)
                if self.verbose:
                    print(
                        f"[ATTRACTION_SEARCH] 评分后景点结果(只输出前3名): {attr_info[0:3]}"
                    )
                # attr_info.to_csv(attr_path, index=False)
                # print("save  >>> ", attr_path)

                self.poi_info["attractions"] = attr_info
                score_list = attr_info["importance"].values
                for r_i in range(len(score_list)):

                    attr_idx = r_i
                    # print(ranking_idx, r_i)

                    # print("visiting: ", self.restaurants_visiting, res_idx)
                    if not (attr_idx in self.attractions_visiting):

                        poi_sel = self.poi_info["attractions"].iloc[attr_idx]

                        transports_sel = goto(
                            city=query["target_city"],
                            start=current_position,
                            end=poi_sel["name"],
                            start_time=current_time,
                            method=poi_plan["transport_preference"],
                            verbose=False,
                        )

                        if transports_sel[0]["mode"] == "metro":
                            transports_sel[0]["tickets"] = query["people_number"]
                        elif transports_sel[0]["mode"] == "taxi":
                            transports_sel[0]["car"] = (
                                int((query["people_number"] - 1) / 4) + 1
                            )
                        arrived_time = transports_sel[-1]["end_time"]
                        # exit(0)
                        # 开放时间
                        # //TODO 景点开放时间
                        # opentime, endtime = poi_sel["opentime"],  poi_sel["endtime"]
                        opentime = "08:00"
                        endtime = "23:00"
                        if self.verbose:
                            print(f"[ATTRACTION_TIME] 到达时间: {arrived_time}")
                        # too late
                        if time_compare_if_earlier_equal("23:00", arrived_time):
                            continue
                        # it is closed ...
                        if time_compare_if_earlier_equal(endtime, arrived_time):
                            continue

                        if time_compare_if_earlier_equal(arrived_time, opentime):
                            act_start_time = opentime
                        else:
                            act_start_time = arrived_time

                        if ("accommodation" in poi_plan) and (
                            current_day < query["days"] - 1
                        ):
                            candidates_type.append("hotel")

                        if (
                            current_day == query["days"] - 1 and current_time != ""
                        ):  # and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], add_time_delta(current_time, 180)):
                            candidates_type.append("back-intercity-transport")

                        poi_time = 90

                        planning_info = []

                        # print(poi_sel)
                        # //TODO 景区游玩时间
                        # recommendmintime = int(poi_sel["recommendmintime"] * 60)
                        # recommendmaxtime = int(poi_sel["recommendmaxtime"] * 60)
                        recommendmintime = 120
                        recommendmaxtime = 180
                        planning_info.append(
                            "请作为一个旅行规划助手帮助我想构建行程，我的需求是{}".format(
                                query["nature_language"]
                            )
                        )
                        planning_info.append(
                            "现在是第{}天{},我到达了{},这个景点的开放时间是[{}--{}]，建议的游玩时间是{}-{}分钟，请帮助我思考我在这个景点游玩多久".format(
                                current_day + 1,
                                current_time,
                                poi_sel["name"],
                                opentime,
                                endtime,
                                recommendmintime,
                                recommendmaxtime,
                            )
                        )

                        planning_info.append(
                            "在邻近中午的时间找家餐厅吃饭，注意午餐时间要求在[11:00 -- 13:00]，请在游玩后预留时间用于交通到达相应地点"
                        )
                        planning_info.append(
                            "在邻近傍晚的时间找家餐厅吃饭，注意午餐时间要求在[17:00 -- 20:00]，请在游玩后预留时间用于交通到达相应地点"
                        )

                        if ("accommodation" in poi_plan) and (
                            current_day < query["days"] - 1
                        ):
                            planning_info.append(
                                "在一天中游览不同的景点，注意景点的游览和中途的交通要花费一定时间，请在游玩后预留时间以便在合适时间用餐、在夜间回到酒店"
                            )

                        if (
                            current_day == query["days"] - 1 and current_time != ""
                        ):  # and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], add_time_delta(current_time, 180)):
                            planning_info.append(
                                "今天是旅程的最后一天，可以选择回家,返程车票时间是{}，请预留时间用于交通到达相应地点".format(
                                    poi_plan["back_transport"]["BeginTime"]
                                )
                            )

                        # poi_time = recommend_poi_time(planning_info, poi_sel["name"])
                        poi_time = recommendmintime

                        act_end_time = add_time_delta(act_start_time, poi_time)
                        if time_compare_if_earlier_equal(endtime, act_end_time):
                            act_end_time = endtime

                        if time_compare_if_earlier_equal(act_end_time, act_start_time):
                            continue

                        activity_i = {
                            "position": poi_sel["name"],
                            "type": poi_type,
                            "transports": transports_sel,
                            "cost": 100,
                            "start_time": act_start_time,
                            "end_time": act_end_time,
                            "photos": (
                                poi_sel["photos"][0]
                                if len(poi_sel["photos"]) > 0
                                else ""
                            ),
                        }

                        plan[current_day]["activities"].append(activity_i)

                        new_time = act_end_time
                        new_position = poi_sel["name"]

                        self.attractions_visiting.append(attr_idx)
                        self.spot_type_visiting.append(poi_sel["type"])
                        self.attraction_names_visiting.append(poi_sel["name"])
                        if self.verbose:
                            print(
                                f"[ATTRACTION_PLAN] 景点规划 - 已访问景点: {self.attraction_names_visiting}, 时间: {new_time}, 位置: {new_position}, 天数: {current_day}"
                            )
                        success, plan = self.search_poi(
                            query, poi_plan, plan, new_time, new_position, current_day
                        )

                        if success:
                            return True, plan
                        plan[current_day]["activities"].pop()
                        self.attractions_visiting.pop()
                        self.spot_type_visiting.pop()
                        self.attraction_names_visiting.pop()
            else:
                if self.verbose:
                    print(f"[POI_TYPE_ERROR] 错误的POI类型: {poi_type}")
                continue

            # list.remove(x): x not in list
            if poi_type in candidates_type:
                candidates_type.remove(poi_type)
            if self.verbose:
                print("[POI_SEARCH] 尝试另一种POI类型")

        return False, plan

    def get_poi_type_from_time_sym(
        self, current_time, candidates_type, back_transport_time
    ):
        if "back-intercity-transport" in candidates_type:
            if not time_compare_if_earlier_equal(
                add_time_delta(current_time, 240), back_transport_time
            ):
                return "back-intercity-transport"

        # too late
        if self.verbose:
            print(
                f"[POI_TYPE_SELECTION] 当前时间: {current_time}, 候选类型: {candidates_type}"
            )
        if (
            time_compare_if_earlier_equal("20:40", current_time)
            and "hotel" in candidates_type
        ):
            return "hotel"

        # lunch time
        if (
            ("lunch" in candidates_type)
            and time_compare_if_earlier_equal("11:00", current_time)
            and time_compare_if_earlier_equal(current_time, "13:00")
        ):
            return "lunch"

        # dinner time
        if (
            ("dinner" in candidates_type)
            and time_compare_if_earlier_equal("17:00", current_time)
            and time_compare_if_earlier_equal(current_time, "20:00")
        ):
            return "dinner"

        return "attraction"

    def constraints_validation(self, query, plan, poi_plan):
        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": plan,
        }
        print("[PLAN_RESULTS]现有的计划res_plan：", res_plan)
        bool_result = func_commonsense_constraints(query, res_plan)
        print("[CONSTRAINTS]常识检验结果:", bool)

        # 提取所有参数
        try:
            # 默认不启用详细搜索以避免在约束验证时触发大量API调用
            extracted_vars = get_symbolic_concepts(
                query, res_plan, enable_detailed_search=False
            )
        except Exception as e:
            if self.verbose:
                print(f"[CONSTRAINTS] 提取符号概念时出错: {str(e)}")
                print(f"[CONSTRAINTS] 错误类型: {type(e).__name__}")
            extracted_vars = None
        if self.verbose:
            print("[CONSTRAINTS] ========================================")
            print(f"[CONSTRAINTS] 硬性约束(extracted_var): {extracted_vars}")

        logical_result = evaluate_logical_constraints(
            extracted_vars, query["hard_logic"]
        )
        if self.verbose:
            print(f"[CONSTRAINTS] 逻辑约束结果(logical_result): {logical_result}")

        logical_pass = True
        for idx, item in enumerate(logical_result):
            logical_pass = logical_pass and item

            if item:
                if self.verbose:
                    print(f"[CONSTRAINTS] {query['hard_logic'][idx]} 通过!")
            else:
                if self.verbose:
                    print(f"[CONSTRAINTS] {query['hard_logic'][idx]} 失败...")

        # if logical_result:
        #     print("Logical passed!")

        bool_result = bool_result and logical_pass

        # exit(0)

        if bool_result:
            self.avialable_plan = res_plan
            if self.verbose:
                print("[CONSTRAINTS] ========== 通过! ==========")
            return True, res_plan
        else:
            if self.verbose:
                print("[CONSTRAINTS] ========== 失败 ==========")
            return False, plan

    def select_feature(self, planning_info):

        info_list = []

        # poi_info = poi_info.drop(["latitude", "longitude"])

        for p_i in planning_info:
            info_list.append(p_i)

        # info_list.append("请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的列表，列表长度与给出的信息条数一致一般是10条")

        action_prompt = (
            "如果你认为描述:XX是最合适的，请按##XX##的格式输出，不要输出多余信息"
        )

        result = self.react_prompt(info_list, action_prompt)
        if self.verbose:
            print(f"[SELECT_FEATURE] 特性选择结果: {result}")

        sel_feature = result.split("##")[1]

        if self.verbose:
            print(f"[SELECT_FEATURE] 选择的特性: {sel_feature}")

        return sel_feature

    def react_prompt(
        self, info_list, action_prompt, return_history_message=False, history_message=[]
    ):

        scratchpad = ""
        for info_i in info_list:
            scratchpad = scratchpad + info_i + "\n"

        scratchpad += f"Thought:"

        json_scratchpad = history_message
        json_scratchpad.append({"role": "user", "content": scratchpad})

        # thought = self.llm(self.prompt+self.scratchpad)
        if self.verbose:
            print(f"[REACT_PROMPT] LLM查询: {json_scratchpad}")
        thought = llm_model(json_scratchpad)
        scratchpad = scratchpad + " " + thought

        json_scratchpad.append({"role": "assistant", "content": thought})
        # if self.need_print:
        if self.verbose:
            print(f"[REACT_PROMPT] 思考: {thought}")
        # Act
        scratchpad = action_prompt + f"\nAction: "

        json_scratchpad.append({"role": "user", "content": scratchpad})

        if self.verbose:
            print(f"[REACT_PROMPT] LLM查询: {json_scratchpad}")
        # action = self.llm(self.prompt+self.scratchpad)
        action = llm_model(json_scratchpad)

        json_scratchpad.append({"role": "assistant", "content": action})
        # print(action)

        scratchpad += " " + str(action)
        # # if self.need_print:
        if self.verbose:
            print(f"[REACT_PROMPT] 行动: {str(action)}")

        if return_history_message:
            return action, json_scratchpad

        return action

    def extract_location_from_plan(self, plan):
        """从plan中按照天数提取去过的地点(餐厅, 景点)"""
        location_list = {}
        for i, day in enumerate(plan):
            location_list[i] = []
            for activity in day["activities"]:
                if "position" in activity:
                    location_list[i].append(activity["position"])
        return location_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument(
        "--level",
        "-l",
        type=str,
        default="easy",
        choices=["easy", "medium", "medium_plus", "human", "human_small", "example"],
        help="query subset",
    )
    parser.add_argument("--index", "-i", type=int, default=None, help="query index")
    parser.add_argument(
        "--start", "-s", type=int, default=None, help="start query index"
    )
    parser.add_argument("--mode", "-m", type=str, default="human", help="backend model")
    parser.add_argument("--fast", action="store_true", default=False)

    args = parser.parse_args()

    # coordinate_A = poi_search.search("杭州", "杭州西站")
    # print(coordinate_A)

    # test_example=load_json_file("../evaluation/example/query_1.json")
    # symbolic_search(test_example)

    query_level = args.level

    if args.level == "easy":
        query_level = "easy_0923"
    elif args.level == "medium":
        query_level = "medium_0925"
    elif query_level == "example":
        query_level = "example"

    if args.mode == "deepseek":
        interactive_mode = "llm"
        llm_model = deepseek_poi()
    elif args.mode == "deepseek_json":
        interactive_mode = "llm"
        llm_model = deepseek_json()
    elif args.mode == "gpt4":
        interactive_mode = "llm"
        llm_model = model_GPT_poi()

    test_input = load_json_file(
        "/lamda/shaojj/codes/TravelPlanner-main/ChinaTravel/data/{}.json".format(
            query_level
        )
    )

    success_count, total = 0, 0

    result_path = "results_tmp/"
    if not os.path.exists(result_path + "{}".format(query_level)):
        os.makedirs(result_path + "{}".format(query_level))

    global fast_mode

    model_name = args.mode
    if args.fast:
        model_name = model_name + "_fast"
        fast_mode = True
    else:
        fast_mode = False

    if not os.path.exists(result_path + "{}/{}".format(query_level, model_name)):
        os.makedirs(result_path + "{}/{}".format(query_level, model_name))
    result_dir = result_path + "{}/{}".format(query_level, model_name)

    fail_list = []

    # for idx in range(len(test_input)):
    # test_idx = [13]
    # test_idx = range(len(test_input))
    test_idx = range(150)

    # test_idx = [9]
    if not (args.index is None):
        test_idx = [args.index]

    if not (args.start is None):
        test_idx = range(args.start, 150)

    with open(result_dir + "/fail_list.txt".format(query_level), "a+") as dump_f:
        dump_f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        dump_f.write("testing range: {}".format(str(test_idx)) + "\n")

    for idx in test_idx:

        query_i = test_input[idx]
        print(f"[QUERY] 查询 {idx}/{len(test_input)}")

        sys.stdout = Logger(result_dir + "/plan_{}.log".format(idx), sys.stdout)
        sys.stderr = Logger(result_dir + "/plan_{}.error".format(idx), sys.stderr)

        avialable_plan = {}

        # try:
        searcher = Interactive_Search()
        query_i["nature_language"] = query_i["nature_language"] + "想去黄鹤楼"
        success, plan = searcher.symbolic_search(query_i)
        print(f"[RESULT] 计划: {plan}")
        exit(0)
    print(f"[EVALUATION] 成功率 [{query_level}]: {success_count}/{total}")

    res_stat = {
        "success": success_count,
        "total": total,
        "fail_list": fail_list,
    }

    with open(
        result_dir + "/result_stat_{}.json".format(str(time.time())),
        "w",
        encoding="utf8",
    ) as dump_f:
        json.dump(res_stat, dump_f, ensure_ascii=False, indent=4)

    with open(result_dir + "/fail_list.txt", "a+") as dump_f:
        dump_f.write(
            f"[EVALUATION] 成功率 [{query_level}]: {success_count}/{total}" + "\n"
        )
