import sys

sys.path.append("../")
from tools.hotels.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.attractions.apis import Attractions
from tools.intercity_transport.apis import IntercityTransport

from envs import goto
import json
import os
import sys


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
import sys
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
            print("query:", query)
        self.avialable_plan = {}
        query["transport_preference"] = "metro"
        query["hard_logic_unseen"] = []
        if self.verbose:
            print("line 208,hard_logic:", query["hard_logic"])
        for idx, item in enumerate(query["hard_logic"]):  # 这里开始解析每个约束
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
                    query["hard_logic"][idx] = str(query["food_type"]) + "<=food_type"
                else:
                    query["hard_logic"][idx] = " 3 < 33"
            # print(item)
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

            elif item.startswith("hotel_price<="):
                query["hotel_price"] = int(item.split("<=")[1])

            elif item.endswith("intercity_transport") or item.startswith(
                "intercity_transport"
            ):
                query["intercity_transport_type"] = item.split("'")[1]

            elif item.endswith("spot_type"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []

                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)

                if len(seen_list) > 0:
                    query["spot_type"] = set(seen_list)
                    query["hard_logic"][idx] = str(query["spot_type"]) + "<=spot_type"
                else:
                    query["hard_logic"][idx] = " 3 < 33"

            elif item.endswith("attraction_names"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["attraction_names"] = set(seen_list)
                    query["hard_logic"][idx] = (
                        str(query["attraction_names"]) + "<=attraction_names"
                    )
                else:
                    query["hard_logic"][idx] = " 3 < 33"

            elif item.endswith("restaurant_names"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["restaurant_names"] = set(seen_list)
                    query["hard_logic"][idx] = (
                        str(query["restaurant_names"]) + "<=restaurant_names"
                    )
                else:
                    query["hard_logic"][idx] = " 3 < 33"

            elif item.endswith("hotel_names"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["hotel_names"] = set(seen_list)
                    query["hard_logic"][idx] = (
                        str(query["hotel_names"]) + "<=hotel_names"
                    )
                else:
                    query["hard_logic"][idx] = " 3 < 33"

            elif item.endswith("hotel_feature"):
                stlist = item.split("<=")[0].split("}")[0].split("{")[1]

                seen_list = []
                for s_i in stlist.split(","):
                    str_i = s_i.split("'")[1]
                    seen_list.append(str_i)
                if len(seen_list) > 0:
                    query["hotel_feature"] = set(seen_list)
                    query["hard_logic"][idx] = (
                        str(query["hotel_feature"]) + "<=hotel_feature"
                    )
                else:
                    query["hard_logic"][idx] = " 3 < 33"

            else:
                if (
                    "days==" not in item
                    and "people_number==" not in item
                    and "tickets==" not in item
                ):
                    query["hard_logic_unseen"].append(item)

        query["hard_logic"] = [
            item
            for item in query["hard_logic"]
            if item not in query["hard_logic_unseen"]
        ]
        success, plan = self.search_plan(query)
        print("line425:success", success, "plan", plan)
        return success, plan

    def search_plan(self, query):
        if self.verbose:
            print("query:", query)
        poi_plan = {}
        self.poi_info = {}
        self.restaurants_visiting = []
        self.attractions_visiting = []
        self.food_type_visiting = []
        self.spot_type_visiting = []
        self.attraction_names_visiting = []
        self.restaurant_names_visiting = []
        source_city = query["start_city"]
        target_city = query["target_city"]
        # todo:交通也改成调用高德api
        train_go = self.intercity_transport.select(
            start_city=source_city, end_city=target_city, intercity_type="train"
        )
        train_back = self.intercity_transport.select(
            start_city=target_city, end_city=source_city, intercity_type="train"
        )

        flight_go = self.intercity_transport.select(
            start_city=source_city, end_city=target_city, intercity_type="airplane"
        )
        flight_back = self.intercity_transport.select(
            start_city=target_city, end_city=source_city, intercity_type="airplane"
        )

        flight_go["Score"] = flight_go.apply(score_go_intercity_transport, axis=1)
        flight_back["Score"] = flight_back.apply(score_back_intercity_transport, axis=1)
        train_go["Score"] = train_go.apply(score_go_intercity_transport, axis=1)
        train_back["Score"] = train_back.apply(score_back_intercity_transport, axis=1)

        go_transport = combine_transport_dataframe(flight_go, train_go)
        back_transport = combine_transport_dataframe(flight_back, train_back)

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
            print(
                "from {} to {}: {} flights, {} trains".format(
                    source_city, target_city, flight_go_num, train_go_num
                )
            )
            print(
                "from {} to {}: {} flights, {} trains".format(
                    target_city, source_city, flight_back_num, train_back_num
                )
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

                    print("intercity_cost: ", intercity_cost)
                    if intercity_cost >= query["cost"]:
                        continue

                    else:
                        cost_wo_inter_trans = query["cost"] - intercity_cost
                        found_intercity_transport = True
                else:
                    found_intercity_transport = True

        success, plan = self.search_poi(
            query, poi_plan, plan=[], current_time="", current_position=""
        )

        print(success, plan)
        if success:
            return True, plan
        return False, {"info": "No Solution"}

    def score_poi_think_overall_act_page(
        self,
        planning_info,
        poi_info_list,
        need_db=False,
        react=False,
        history_message=[],
    ):

        info_list = []
        # new_poi_info=poi_info_list[0:30]
        new_poi_info = []
        if need_db:
            new_poi_info = random.choices(poi_info_list, k=min(25, len(poi_info_list)))
        for item in new_poi_info:
            info_list.append(item)

        for p_i in planning_info:  # 用户信息偏好或规划
            info_list.append(p_i)

        overall_plan, history_message_think_overall = self.reason_prompt(
            info_list, return_history_message=True, history_message=[]
        )
        score_list = []

        # rewrite the plan
        rewrite_ans = self.rewrite_plan(overall_plan)

        querys = rewrite_ans.split(", ")
        scores = []
        for q in querys:
            score = self.ret.get_score(poi_info_list, q)
            scores.append(score)

        scores = np.array(scores)
        score_list = np.max(scores, axis=0)

        if self.verbose:
            print("Score Over!")

        return score_list

    def reason_prompt(self, info_list, return_history_message=True, history_message=[]):

        scratchpad = ""

        for info_i in info_list:
            scratchpad = scratchpad + info_i + "\n"

        scratchpad += f"请简洁的回复，要求景点和餐厅在满足用户要求的同时尽可能多样:"

        json_scratchpad = history_message
        json_scratchpad.append({"role": "user", "content": scratchpad})

        if self.verbose:
            print("LLM query: ", json_scratchpad)
        thought = self.llm_model(json_scratchpad)
        scratchpad = scratchpad + " " + thought
        json_scratchpad.append({"role": "assistant", "content": thought})

        if self.verbose:
            print(f"Answer:", thought)

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
            print("LLM query: ", json_scratchpad)
        thought = self.llm_model(json_scratchpad)

        if self.verbose:
            print(f"Answer:", thought)
        return thought

    def search_poi(
        self, query, poi_plan, plan, current_time, current_position, current_day=0
    ):
        print("line874,query", query)
        print("line874,poi_plan", poi_plan)
        print("line874,plan", plan)
        print("line874,current_time", current_time)
        print("line874,current_position", current_position)
        print("line874,current_day", current_day)

        target_city = query["target_city"]
        if "cost_wo_intercity" in query:
            inner_city_cost = calc_cost_from_itinerary_wo_intercity(
                plan, query["people_number"]
            )

            if inner_city_cost >= query["cost_wo_intercity"]:

                if self.verbose:
                    print(
                        "budget run out: inner-city budget {}, cost {}".format(
                            query["cost_wo_intercity"], inner_city_cost
                        )
                    )

                return False, plan

        if current_time != "" and time_compare_if_earlier_equal("23:00", current_time):
            if self.verbose:
                print("too late, after 23:00")
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
            keywords = "酒店"
            search_query = ""

            # 1. 处理酒店特性
            if "hotel_feature" in query:
                # 把特性添加到关键字前面
                features = query["hotel_feature"]
                if isinstance(features, list) and features:
                    for feature in features:
                        keywords = f"{feature}{keywords}"
            # 2. 处理酒店名称
            if "hotel_names" in query and query["hotel_names"]:
                hotel_name = (
                    list(query["hotel_names"])[0]
                    if isinstance(query["hotel_names"], (list, tuple, set))
                    else query["hotel_names"]
                )
                search_query += f" 酒店名字为{hotel_name}"
            # 3. 处理价格筛选
            if "hotel_price" in query:
                price = query["hotel_price"]
                search_query += f" 价格为{price}"

            # 组合最终搜索关键词
            final_keywords = f"{keywords} {search_query}".strip()
            hotel_info = self.accommodation.select(target_city, keywords=final_keywords)
            hotel_sel = hotel_info.iloc[0]
            poi_plan["accommodation"] = hotel_info.iloc[0]

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
                        "Can not go back to hotel, current POI {}, hotel arrived time: {}".format(
                            current_position, arrived_time
                        )
                    )
                return False, plan

        # intercity_transport - go
        if current_day == 0 and current_time == "":
            plan = [{"day": current_day + 1, "activities": []}]

            if poi_plan["go_transport"]["Type"] == "train":
                plan[current_day]["activities"].append(
                    {
                        "start_time": poi_plan["go_transport"]["BeginTime"],
                        "end_time": poi_plan["go_transport"]["EndTime"],
                        "start": poi_plan["go_transport"]["From"],
                        "end": poi_plan["go_transport"]["To"],
                        "ID": poi_plan["go_transport"]["ID"],
                        "type": "train",
                        "transports": [],
                        "cost": poi_plan["go_transport"]["Cost"],
                        "tickets": query["people_number"],
                    }
                )
            else:
                plan[current_day]["activities"].append(
                    {
                        "start_time": poi_plan["go_transport"]["BeginTime"],
                        "end_time": poi_plan["go_transport"]["EndTime"],
                        "start": poi_plan["go_transport"]["From"],
                        "end": poi_plan["go_transport"]["To"],
                        "ID": poi_plan["go_transport"]["ID"],
                        "type": "airplane",
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
                    print("No solution for the given Go Transport")
                return False, plan

        # breakfast

        if current_time == "00:00":  # 新的一天开始了
            print("new day coming!")
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
            print("new_time", new_time)
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
                    "POI planning, day {} {}, {}, next-poi type: {}".format(
                        current_day, current_time, current_position, poi_type
                    )
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
                print("line676", res_plan)
                # return True, res_plan  # todo 先不检查约束了，成功回家要紧
                if res_bool:
                    return True, res_plan
                else:

                    plan[current_day]["activities"].pop()

                    print(
                        "[We choose to go back transport and finish this trip], but constraints_validation failed..."
                    )

            elif poi_type in ["lunch", "dinner"]:
                keywords = f"{current_position}" + "餐厅"  # 吃附近的餐厅
                self.poi_info["restaurants"] = self.restaurants.select(
                    target_city, keywords
                )
                search_query = ""
                info_list = [query["nature_language"]]
                info_list.append(
                    "在这次在{}的旅行中,请帮我选择一些餐厅去吃，评估每个餐厅的相关性，看看是否需要安排到行程里".format(
                        query["target_city"]
                    )
                )
                rest_info = self.poi_info["restaurants"]

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

                info_list.append("请根据以上信息考虑什么样的餐厅满足我们的需求")

                poi_info_list = []
                score_list = []
                for idx in range(len(rest_info)):

                    res_i = self.poi_info["restaurants"].iloc[idx]

                    poi_info_list.append(
                        "**{}** name: {} , cusine: {}, price_per_preson: {}, recommended food: {}".format(
                            idx,
                            res_i["name"],
                            res_i["cuisine"],
                            res_i["price"],
                            res_i["recommendedfood"],
                        )
                    )
                    # poi_info_list.append("**{}** name: {} , arrived_time: {}, cusine: {}, price_per_preson: {}".format(idx, res_i["name"], arrived_time, res_i["cuisine"], res_i["price"]))

                score_list = self.score_poi_think_overall_act_page(
                    info_list, poi_info_list, need_db=True, react=True
                )
                rest_info["importance"] = score_list

                rest_info = mmr_algorithm(name_key="name", df=rest_info)
                if self.verbose:
                    print(rest_info)
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

                            if len(transports_sel) == 3:
                                transports_sel[1]["tickets"] = query["people_number"]
                            elif transports_sel[0]["mode"] == "taxi":
                                transports_sel[0]["car"] = (
                                    int((query["people_number"] - 1) / 4) + 1
                                )

                            arrived_time = add_time_delta(
                                current_time, 30
                            )  # todo 接入小交通，精确计算路程时间
                            print("line761", arrived_time)
                            # 开放时间
                            # todo:从高德api返还的字符串提取出该餐厅开放时间和关闭时间
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

                            act_end_time = add_time_delta(act_start_time, 90)
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

                            activity_i = {
                                "position": poi_sel["name"],
                                "type": poi_type,
                                "transports": transports_sel,
                                "cost": 100,  # todo 餐厅价格
                                "start_time": act_start_time,
                                "end_time": act_end_time,
                            }

                            plan[current_day]["activities"].append(activity_i)

                            new_time = act_end_time
                            new_position = poi_sel["name"]
                            print("line814", new_time, new_position)
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

                            print("res {} fail...".format(poi_sel["name"]))

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
                if len(transports_sel) == 3:
                    transports_sel[1]["tickets"] = query["people_number"]
                elif transports_sel[0]["mode"] == "taxi":
                    transports_sel[0]["car"] = int((query["people_number"] - 1) / 4) + 1

                arrived_time = transports_sel[-1]["end_time"]

                activity_i = {
                    "position": hotel_sel["name"],
                    "type": "accommodation",
                    "room_type": 2,  # //todo 酒店房型
                    "transports": transports_sel,
                    "cost": 350,  # //todo 酒店价格
                    "start_time": arrived_time,
                    "end_time": "24:00",
                    "rooms": 1,  # todo:用户需求房型
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
                    query["target_city"], "旅游景点"
                )

                info_list = [query["nature_language"]]
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
                info_list.append("请根据以上信息考虑什么样的景点满足我们的需求")
                attr_info = self.poi_info["attractions"]
                poi_info_list = []
                for idx in range(len(attr_info)):
                    res_i = attr_info.iloc[idx]
                    poi_info_list.append(
                        "**{}** name: {} , price_per_preson: {}, type: {}".format(
                            idx, res_i["name"], res_i["price"], res_i["type"]
                        )
                    )

                score_list = self.score_poi_think_overall_act_page(
                    info_list, poi_info_list, react=True
                )
                attr_info["importance"] = score_list

                attr_info = mmr_algorithm(name_key="name", df=attr_info)
                # attr_info = attr_info.sort_values(by = ["importance"], ascending=False)
                if self.verbose:
                    print(attr_info)
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

                        if len(transports_sel) == 3:
                            transports_sel[1]["tickets"] = query["people_number"]
                        elif transports_sel[0]["mode"] == "taxi":
                            transports_sel[0]["car"] = (
                                int((query["people_number"] - 1) / 4) + 1
                            )
                        arrived_time = transports_sel[-1]["end_time"]
                        # exit(0)
                        # 开放时间
                        # //todo 景点开放时间
                        # opentime, endtime = poi_sel["opentime"],  poi_sel["endtime"]
                        opentime = "08:00"
                        endtime = "23:00"
                        print("line973,arrived_time:", arrived_time)
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
                        # //todo 景区游玩时间
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
                            "cost": 100,  # //todo 景点门票价格xin'xi'que'shi
                            "start_time": act_start_time,
                            "end_time": act_end_time,
                        }

                        plan[current_day]["activities"].append(activity_i)

                        new_time = act_end_time
                        new_position = poi_sel["name"]

                        self.attractions_visiting.append(attr_idx)
                        self.spot_type_visiting.append(poi_sel["type"])
                        self.attraction_names_visiting.append(poi_sel["name"])
                        print(
                            "line1046,attraction_names_visiting:",
                            self.attraction_names_visiting,
                            new_time,
                            new_position,
                            current_day,
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
                    print("incorrect poi type: {}".format(poi_type))
                continue

            # list.remove(x): x not in list
            if poi_type in candidates_type:
                candidates_type.remove(poi_type)
            if self.verbose:
                print("try another poi type")

        return False, plan

    def get_poi_type_from_time_sym(
        self, current_time, candidates_type, back_transport_time
    ):
        if "back-intercity-transport" in candidates_type:
            if time_compare_if_earlier_equal(
                back_transport_time, add_time_delta(current_time, 180)
            ):
                return "back-intercity-transport"

        # too late
        print("line1036", current_time, candidates_type)
        if (
            time_compare_if_earlier_equal("22:00", current_time)
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

        bool_result = func_commonsense_constraints(query, res_plan)

        if bool_result:
            self.avialable_plan = res_plan

        try:
            extracted_vars = get_symbolic_concepts(query, res_plan)

        except:
            extracted_vars = None
        if self.verbose:
            print(extracted_vars)

        logical_result = evaluate_logical_constraints(
            extracted_vars, query["hard_logic"]
        )
        if self.verbose:
            print(logical_result)

        logical_pass = True
        for idx, item in enumerate(logical_result):
            logical_pass = logical_pass and item

            if item:
                print(query["hard_logic"][idx], "passed!")
            else:

                print(query["hard_logic"][idx], "failed...")

        # if logical_result:
        #     print("Logical passed!")

        bool_result = bool_result and logical_pass

        # exit(0)

        if bool_result:
            print("\n Pass! \n")
            return True, res_plan
        else:
            print("\n Failed \n")
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
            print(result)

        sel_feature = result.split("##")[1]

        if self.verbose:
            print("selected_feature: ", sel_feature)

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
            print("LLM query: ", json_scratchpad)
        thought = llm_model(json_scratchpad)
        scratchpad = scratchpad + " " + thought

        json_scratchpad.append({"role": "assistant", "content": thought})
        # if self.need_print:
        if self.verbose:
            print(f"Thought:", thought)
        # Act
        scratchpad = action_prompt + f"\nAction: "

        json_scratchpad.append({"role": "user", "content": scratchpad})

        if self.verbose:
            print("LLM query: ", json_scratchpad)
        # action = self.llm(self.prompt+self.scratchpad)
        action = llm_model(json_scratchpad)

        json_scratchpad.append({"role": "assistant", "content": action})
        # print(action)

        scratchpad += " " + str(action)
        # # if self.need_print:
        if self.verbose:
            print(f"Action:", str(action))

        if return_history_message:
            return action, json_scratchpad

        return action


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
        print("query {}/{}".format(idx, len(test_input)))

        sys.stdout = Logger(result_dir + "/plan_{}.log".format(idx), sys.stdout)
        sys.stderr = Logger(result_dir + "/plan_{}.error".format(idx), sys.stderr)

        avialable_plan = {}

        # try:
        searcher = Interactive_Search()
        query_i["nature_language"] = query_i["nature_language"] + "想去黄鹤楼"
        success, plan = searcher.symbolic_search(query_i)
        print(plan)
        exit(0)
    print("success rate [{}]: {}/{}".format(query_level, success_count, total))

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
            "success rate [{}]: {}/{}".format(query_level, success_count, total) + "\n"
        )
