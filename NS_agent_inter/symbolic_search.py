import argparse
import numpy as np
import datetime
import time
from evaluation.hard_constraint import evaluate_constraints as evaluate_logical_constraints
from evaluation.hard_constraint import get_symbolic_concepts, calc_cost_from_itinerary_wo_intercity
from evaluation.commonsense_constraint import func_commonsense_constraints
from envs import ReactEnv
from evaluation.utils import load_json_file
from tqdm import tqdm
import os
import json
from envs import goto
from tools.transportation.apis import GoTo
from tools.intercity_transport.apis import IntercityTransport
from tools.attractions.apis import Attractions
from tools.restaurants.apis import Restaurants
from tools.hotels.apis import Accommodations
import sys
sys.path.append("../")


TIME_CUT = 60 * 5
time_before_search = 0

accommodation = Accommodations()
restaurants = Restaurants()
attractions = Attractions()
intercity_transport = IntercityTransport()

# 日志记录器类，将日志写入文件和终端


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        # 初始化日志文件和输出流
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        # 写入日志
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 刷新缓冲区（此处为空实现）
        pass

# 验证旅行计划是否满足所有常识性和硬性约束
# 返回是否通过及最终计划
# query: 查询条件，plan: 当前行程，poi_plan: 规划细节


def constraints_validation(query, plan, poi_plan):

    global avialable_plan
    res_plan = {"people_number": query["people_number"],
                "start_city": query["start_city"],
                "target_city": query["target_city"],
                "itinerary": plan,
                }
    print("validate the plan [for query {}]: ".format(poi_plan["query_idx"]))
    print(res_plan)
    # 检验常识约束
    bool_result = func_commonsense_constraints(query, res_plan)

    if bool_result:
        avialable_plan = res_plan

    try:
        extracted_vars = get_symbolic_concepts(query, res_plan)

    except:
        extracted_vars = None

    print(extracted_vars)
    # 验证逻辑检验
    logical_result = evaluate_logical_constraints(
        extracted_vars, query["hard_logic"])

    print(logical_result)

    logical_pass = True
    for idx, item in enumerate(logical_result):
        logical_pass = logical_pass and item

        if item:
            print(query["hard_logic"][idx], "passed!")
        else:

            print(query["hard_logic"][idx], "failed...")

    bool_result = bool_result and logical_pass

    if bool_result:
        print("\n Pass! \n")

    else:
        print("\n Failed \n")

    # exit(0)

    if bool_result:
        return True, res_plan
    else:
        return False, plan

# 比较两个时间字符串，判断time_1是否早于等于time_2


def time_compare_if_earlier_equal(time_1, time_2):

    time1 = float(time_1.split(":")[0])*60 + float(time_1.split(":")[1])
    time2 = float(time_2.split(":")[0])*60 + float(time_2.split(":")[1])

    return time1 <= time2


# 计算下一个时间区间点（如对齐到下一个时间段）
# time1: 当前时间字符串，time_delta: 间隔分钟数
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

# 给时间字符串增加指定分钟数，返回新的时间字符串


def add_time_delta(time1, time_delta):

    hour, minu = int(time1.split(":")[0]), int(time1.split(":")[1])

    min_new = minu + time_delta

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


# 根据当前时间和候选类型，判断下一个POI类型（如午餐/晚餐/景点/酒店）
def get_poi_type_from_time(current_time, candidates_type):

    hour, minuate = int(current_time.split(":")[0]), int(
        current_time.split(":")[1])

    # too late
    if time_compare_if_earlier_equal("22:30", add_time_delta(current_time, 120)) and "hotel" in candidates_type:
        return "hotel"

    # lunch time
    if ("lunch" in candidates_type) and \
        (time_compare_if_earlier_equal("11:00", add_time_delta(current_time, 40))
         or time_compare_if_earlier_equal("12:40", add_time_delta(current_time, 120))):
        return "lunch"

    # dinner time
    if ("dinner" in candidates_type) and \
        (time_compare_if_earlier_equal("17:00", add_time_delta(current_time, 40)) or
            time_compare_if_earlier_equal("19:00", add_time_delta(current_time, 120))):
        return "dinner"

    return "attraction"


# 递归搜索并安排下一个POI，构建完整行程
# query: 查询条件，poi_plan: 规划细节，plan: 当前行程，current_time: 当前时间，current_position: 当前地点
# current_day: 当前天数，verbose: 是否打印详细信息

def search_poi(query, poi_plan, plan, current_time, current_position, current_day=0, verbose=True):

    # 记录搜索起始时间，超时则返回失败
    global time_before_search
    if time.time() > time_before_search + TIME_CUT:
        print("Searching TIME OUT !!!")
        return False, plan
        # return False, {"info": "TIME OUT"}

    # 检查城市内预算是否超支，超出返回false
    if "cost_wo_intercity" in query:
        inner_city_cost = calc_cost_from_itinerary_wo_intercity(
            plan, query["people_number"])
        if inner_city_cost >= query["cost_wo_intercity"]:
            print("budget run out: inner-city budget {}, cost {}".format(
                query["cost_wo_intercity"], inner_city_cost))
            return False, plan

    # 初始化全局变量，记录已访问的POI等
    global poi_info
    global restaurants_visiting
    global attractions_visiting
    global food_type_visiting
    global spot_type_visiting
    global attraction_names_visiting
    global restaurant_names_visiting

    # 如果时间已晚，直接返回失败
    if current_time != "" and time_compare_if_earlier_equal("23:00", current_time):
        print("too late, after 23:00")
        return False, plan
    # 最后一天，判断能否及时返程
    if current_time != "" and current_day == query["days"] - 1:
        # 返程交通安排
        transports_sel = goto(city=query["target_city"],
                              start=current_position, end=poi_plan["back_transport"]["From"],
                              start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
        arrived_time = transports_sel[-1]["end_time"]
        if time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], arrived_time):
            print("Can not go back source-city in time, current POI {}, station arrived time: {}".format(
                current_position, arrived_time))
            return False, plan
    # 其他情况，判断能否及时回酒店
    elif current_time != "":
        hotel_sel = poi_plan["accommodation"]
        transports_sel = goto(city=query["target_city"],
                              start=current_position, end=hotel_sel["name"],
                              start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
        arrived_time = transports_sel[-1]["end_time"]
        if time_compare_if_earlier_equal("24:00", arrived_time):
            print("Can not go back to hotel, current POI {}, hotel arrived time: {}".format(
                current_position, arrived_time))
            return False, plan

    # 最后一天且距离返程不足3小时，直接安排返程交通
    if current_day == query["days"] - 1 \
            and current_time != "" \
            and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], add_time_delta(current_time, 180)):
        if len(plan) < current_day + 1:
            plan.append({"day": current_day + 1, "activities": []})
        transports_sel = goto(city=query["target_city"],
                              start=current_position, end=poi_plan["back_transport"]["From"],
                              start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
        if len(transports_sel) == 3:
            transports_sel[1]["tickets"] = query["people_number"]
        elif transports_sel[0]["mode"] == "taxi":
            transports_sel[0]["car"] = int(
                (query["people_number"] - 1) / 4) + 1
        # 添加返程交通活动
        if "TrainID" in poi_plan["back_transport"]:
            plan[current_day]["activities"].append(
                {
                    "start_time": poi_plan["back_transport"]["BeginTime"],
                    "end_time": poi_plan["back_transport"]["EndTime"],
                    "start": poi_plan["back_transport"]["From"],
                    "end": poi_plan["back_transport"]["To"],
                    "TrainID": poi_plan["back_transport"]["TrainID"],
                    "type": "train",
                    "transports": transports_sel,
                    "cost": poi_plan["back_transport"]["Cost"],
                    "tickets": query["people_number"]
                })
        else:
            plan[current_day]["activities"].append(
                {
                    "start_time": poi_plan["back_transport"]["BeginTime"],
                    "end_time": poi_plan["back_transport"]["EndTime"],
                    "start": poi_plan["back_transport"]["From"],
                    "end": poi_plan["back_transport"]["To"],
                    "FlightID": poi_plan["back_transport"]["FlightID"],
                    "type": "airplane",
                    "transports": transports_sel,
                    "cost": poi_plan["back_transport"]["Cost"],
                    "tickets": query["people_number"]
                })
        # 检查所有约束
        res_bool, res_plan = constraints_validation(query, plan, poi_plan)
        if res_bool:
            return True, res_plan
        else:
            plan[current_day]["activities"].pop()
            print(
                "[We have to go back transport], but constraints_validation failed...")
            return False, plan

    # 第一天且未开始，先安排去程交通
    if current_day == 0 and current_time == "":
        plan = [{"day": current_day + 1, "activities": []}]
        if "TrainID" in poi_plan["go_transport"]:
            plan[current_day]["activities"].append(
                {
                    "start_time": poi_plan["go_transport"]["BeginTime"],
                    "end_time": poi_plan["go_transport"]["EndTime"],
                    "start": poi_plan["go_transport"]["From"],
                    "end": poi_plan["go_transport"]["To"],
                    "TrainID": poi_plan["go_transport"]["TrainID"],
                    "type": "train",
                    "transports": [],
                    "cost": poi_plan["go_transport"]["Cost"],
                    "tickets": query["people_number"]
                })
        else:
            plan[current_day]["activities"].append(
                {
                    "start_time": poi_plan["go_transport"]["BeginTime"],
                    "end_time": poi_plan["go_transport"]["EndTime"],
                    "start": poi_plan["go_transport"]["From"],
                    "end": poi_plan["go_transport"]["To"],
                    "FlightID": poi_plan["go_transport"]["FlightID"],
                    "type": "airplane",
                    "transports": [],
                    "cost": poi_plan["go_transport"]["Cost"],
                    "tickets": query["people_number"]
                })
        # 递归进入下一步
        new_time = poi_plan["go_transport"]["EndTime"]
        new_position = poi_plan["go_transport"]["To"]
        success, plan = search_poi(
            query, poi_plan, plan, new_time, new_position, current_day, verbose)
        if success:
            return True, plan
        else:
            print("No solution for the given Go Transport")
            return False, plan

    # 早餐安排
    if current_time == "00:00":
        if len(plan) < current_day + 1:
            plan.append({"day": current_day + 1, "activities": []})
        # 酒店早餐
        plan[current_day]["activities"].append({
            "position": poi_plan["accommodation"]["name"],
            "type": "breakfast",
            "transports": [],
            "cost": 0,
            "start_time": "08:00",
            "end_time": "08:30"
        })
        new_time = plan[current_day]["activities"][-1]["end_time"]
        new_position = current_position
        success, plan = search_poi(
            query, poi_plan, plan, new_time, new_position, current_day, verbose)
        if success:
            return True, plan
        plan[current_day]["activities"].pop()

    # 判断今天是否已安排午餐/晚餐
    haved_lunch_today, haved_dinner_today = False, False
    for act_i in plan[current_day]["activities"]:
        if act_i["type"] == "lunch":
            haved_lunch_today = True
        if act_i["type"] == "dinner":
            haved_dinner_today = True

    # 动态生成候选POI类型
    candidates_type = ["attraction"]  # 默认总是可以是景点
    # 如果今天还没吃午餐，且当前时间早于等于12:30，则午餐可作为候选
    if (not haved_lunch_today) and time_compare_if_earlier_equal(current_time, "12:30"):
        candidates_type.append("lunch")
    # 如果今天还没吃晚餐，且当前时间早于等于18:30，则晚餐可作为候选
    if (not haved_dinner_today) and time_compare_if_earlier_equal(current_time, "18:30"):
        candidates_type.append("dinner")
    # 如果还没到最后一天，且有住宿信息，则酒店可作为候选
    if "accommodation" in poi_plan and current_day < query["days"]-1:
        candidates_type.append("hotel")

    # 根据当前时间和候选类型，判断下一个POI类型
    poi_type = get_poi_type_from_time(current_time, candidates_type)
    if verbose:
        print("POI planning, day {} {}, {}, next-poi type: {}".format(current_day,
              current_time, current_position, poi_type))

    # 以下分支分别处理午餐/晚餐、酒店、景点三类POI的递归搜索与安排
    # 1. 午餐/晚餐
    if poi_type in ["lunch", "dinner"]:

        # print(poi_info["restaurants"])

        res_weight = np.ones(poi_info["restaurants_num"])

        if "cost_wo_intercity" in query:

            # priority low cost

            res_price = poi_info["restaurants"]["price"].values

            max_price = res_price.max()
            min_price = res_price.min()

            regularized_price = (res_price - min_price) / \
                (max_price - min_price)

            res_weight = 1. - regularized_price
            # print(regularized_price)

        # print(query["food_type"])
        # print(food_type_visiting)
        # print(not set(query["food_type"]) <= set(food_type_visiting))

        if "food_type" in query and (not set(query["food_type"]) <= set(food_type_visiting)):

            # weight_list = []
            res_info = poi_info["restaurants"]
            for res_i, cuisine_type in enumerate(res_info["cuisine"]):
                # print(cuisine_type)

                res_weight[res_i] *= max(int(cuisine_type in query["food_type"]
                                         and (not cuisine_type in food_type_visiting)), 1e-5)

                # if cuisine_type in query["food_type"] and (not cuisine_type in food_type_visiting):

                #     raw_weight[res_i] *= int(cuisine_type in query["food_type"] and (not cuisine_type in food_type_visiting))
                # else:
                #     raw_weight[res_i] *= 0

        # print(res_weight)
        if "restaurant_names" in query and (not set(query["restaurant_names"]) <= set(restaurant_names_visiting)):
            res_info = poi_info["restaurants"]
            for res_i, attr_name in enumerate(res_info["name"]):
                # print(cuisine_type)
                res_weight[res_i] *= max(int(attr_name in query["restaurant_names"] and (
                    not attr_name in restaurant_names_visiting)), 1e-5)

        ranking_idx = np.argsort(-np.array(res_weight))

        # print(ranking_idx[0])
        # print(res_info.iloc[ranking_idx[0]]["cuisine"], res_info.iloc[ranking_idx[0]]["name"])
        # print(res_info.iloc[ranking_idx[63]])

        # print(ranking_idx[1])
        # print(res_info.iloc[ranking_idx[1]]["cuisine"], res_info.iloc[ranking_idx[1]]["name"])
        # print(ranking_idx[2])
        # print(res_info.iloc[ranking_idx[2]]["cuisine"])
        # print(ranking_idx[3])
        # print(res_info.iloc[ranking_idx[3]]["cuisine"])
        # exit(0)

        for r_i in ranking_idx:

            res_idx = r_i
            # print(ranking_idx, r_i)

            # print("visiting: ", restaurants_visiting, res_idx)

            if not (res_idx in restaurants_visiting):

                # print("in-loop: ", r_i, weight_list[r_i], poi_info["restaurants"].iloc[res_idx]["name"], poi_info["restaurants"].iloc[res_idx]["cuisine"])

                # exit(0)

                poi_sel = poi_info["restaurants"].iloc[res_idx]

                transports_sel = goto(city=query["target_city"], start=current_position,
                                      end=poi_sel["name"], start_time=current_time, method=poi_plan["transport_preference"], verbose=False)

                if len(transports_sel) == 3:
                    transports_sel[1]["tickets"] = query["people_number"]
                elif transports_sel[0]["mode"] == "taxi":
                    transports_sel[0]["car"] = int(
                        (query["people_number"] - 1) / 4) + 1

                arrived_time = transports_sel[-1]["end_time"]

                # 开放时间
                opentime, endtime = poi_sel["weekdayopentime"],  poi_sel["weekdayclosetime"]

                # it is closed ...
                if time_compare_if_earlier_equal(endtime, arrived_time):
                    continue
                if time_compare_if_earlier_equal(arrived_time, opentime):
                    act_start_time = opentime
                else:
                    act_start_time = arrived_time

                if poi_type == "lunch" and time_compare_if_earlier_equal(act_start_time, "11:00"):
                    act_start_time = "11:00"

                if poi_type == "lunch" and time_compare_if_earlier_equal(endtime, "11:00"):
                    continue

                if poi_type == "dinner" and time_compare_if_earlier_equal(act_start_time, "17:00"):
                    act_start_time = "17:00"

                if poi_type == "dinner" and time_compare_if_earlier_equal(endtime, "17:00"):
                    continue

                act_end_time = add_time_delta(act_start_time, 90)
                if time_compare_if_earlier_equal(endtime, act_end_time):
                    act_end_time = endtime

                if poi_type == "lunch" and time_compare_if_earlier_equal("13:00", act_start_time):
                    continue
                if poi_type == "dinner" and time_compare_if_earlier_equal("20:00", act_start_time):
                    continue

                activity_i = {
                    "position": poi_sel["name"],
                    "type": poi_type,
                    "transports": transports_sel,
                    "cost": int(poi_sel["price"]),
                    "start_time": act_start_time,
                    "end_time": act_end_time
                }

                plan[current_day]["activities"].append(activity_i)

                new_time = act_end_time
                new_position = poi_sel["name"]

                restaurants_visiting.append(res_idx)
                food_type_visiting.append(poi_sel["cuisine"])

                # print(res_weight)
                # print(ranking_idx)
                # print(plan)
                # exit(0)

                # print("try to eat res {} ...".format(poi_sel["name"]))

                success, plan = search_poi(
                    query, poi_plan, plan, new_time, new_position, current_day, verbose)

                if success:
                    return True, plan

                plan[current_day]["activities"].pop()
                restaurants_visiting.pop()
                food_type_visiting.pop()

                # print("res {} fail...".format(poi_sel["name"]))

    # 2. 酒店
    elif poi_type == "hotel":

        hotel_sel = poi_plan["accommodation"]

        transports_sel = goto(city=query["target_city"],
                              start=current_position, end=hotel_sel["name"],
                              start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
        if len(transports_sel) == 3:
            transports_sel[1]["tickets"] = query["people_number"]
        elif transports_sel[0]["mode"] == "taxi":
            transports_sel[0]["car"] = int(
                (query["people_number"] - 1) / 4) + 1

        arrived_time = transports_sel[-1]["end_time"]

        activity_i = {
            "position": hotel_sel["name"],
            "type": "accommodation",
                    "room_type": hotel_sel["numbed"],
                    "transports": transports_sel,
                    "cost": hotel_sel["price"],
                    "start_time": arrived_time,
                    "end_time": "24:00",
                    # "rooms": int((query["people_number"] - 1) / hotel_sel["numbed"]) + 1
                    "rooms": query["required_rooms"]
        }

        plan[current_day]["activities"].append(activity_i)

        new_time = "00:00"
        new_position = hotel_sel["name"]

        success, plan = search_poi(
            query, poi_plan, plan, new_time, new_position, current_day + 1, verbose)

        if success:
            return True, plan

        plan[current_day]["activities"].pop()

    # 3. 景点
    elif poi_type == "attraction":
        # 初始化每个景点的权重为1
        attr_weight = np.ones(poi_info["attractions_num"])
        # 如果用户有“景点类型”要求，且还有没去过的类型，则优先安排这些类型的景点。
        if "spot_type" in query and (not set(query["spot_type"]) <= set(spot_type_visiting)):

            # weight_list = []
            attr_info = poi_info["attractions"]
            for res_i, attr_type in enumerate(attr_info["type"]):
                # print(cuisine_type)

                attr_weight[res_i] *= max(int(attr_type in query["spot_type"]
                                          and (not attr_type in spot_type_visiting)), 1e-5)

                # if cuisine_type in query["food_type"] and (not cuisine_type in food_type_visiting):

                #     raw_weight[res_i] *= int(cuisine_type in query["food_type"] and (not cuisine_type in food_type_visiting))
                # else:
                #     raw_weight[res_i] *= 0
        # 如果用户指定了必须要去的景点，且还有没去过的，则优先安排这些景点
        if "attraction_names" in query and (not set(query["attraction_names"]) <= set(attraction_names_visiting)):
            attr_info = poi_info["attractions"]

            # print(query["attraction_names"])

            for res_i, attr_name in enumerate(attr_info["name"]):

                attr_weight[res_i] *= max(int(attr_name in query["attraction_names"] and (
                    not attr_name in attraction_names_visiting)), 1e-5)
        # 排序
        ranking_idx = np.argsort(-np.array(attr_weight))
        # 遍历候选景点

        for r_i in ranking_idx:

            attr_idx = r_i
            # print(ranking_idx, r_i)

            # print("visiting: ", restaurants_visiting, res_idx)

            if not (attr_idx in attractions_visiting):

                poi_sel = poi_info["attractions"].iloc[attr_idx]

                # print(current_position, poi_sel["name"])
                # 计算从当前位置到该景点的交通方式和到达时间
                transports_sel = goto(city=query["target_city"], start=current_position,
                                      end=poi_sel["name"], start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
                # 处理交通方式的特殊字段
                if len(transports_sel) == 3:
                    transports_sel[1]["tickets"] = query["people_number"]
                elif transports_sel[0]["mode"] == "taxi":
                    transports_sel[0]["car"] = int(
                        (query["people_number"] - 1) / 4) + 1
                arrived_time = transports_sel[-1]["end_time"]

                # 获取景点开放时间
                opentime, endtime = poi_sel["opentime"],  poi_sel["endtime"]

                # 如果到达时间太晚（21:00后），跳过   合理吗？？？？
                if time_compare_if_earlier_equal("21:00", arrived_time):
                    continue
                # 如果到达时景点已关门，跳过
                if time_compare_if_earlier_equal(endtime, arrived_time):
                    continue
                # 如果到达时间早于开放时间，则活动开始时间为开放时间，否则为到达时间
                if time_compare_if_earlier_equal(arrived_time, opentime):
                    act_start_time = opentime
                else:
                    act_start_time = arrived_time
                # 动持续90分钟，计算结束时间
                act_end_time = add_time_delta(act_start_time, 90)
                # 如果结束时间超过关门时间，则以关门时间为准
                if time_compare_if_earlier_equal(endtime, act_end_time):
                    act_end_time = endtime
                # 构造本次活动
                activity_i = {
                    "position": poi_sel["name"],
                    "type": poi_type,
                    "transports": transports_sel,
                    "cost": int(poi_sel["price"]),
                    "start_time": act_start_time,
                    "end_time": act_end_time
                }
                # 将活动加入当天行程
                plan[current_day]["activities"].append(activity_i)
                # 更新递归参数
                new_time = act_end_time
                new_position = poi_sel["name"]
                # 记录已访问的景点、类型、名称，防止重复
                attractions_visiting.append(attr_idx)
                spot_type_visiting.append(poi_sel["type"])
                attraction_names_visiting.append(poi_sel["name"])
                # 递归调用，继续安排后续活动
                success, plan = search_poi(
                    query, poi_plan, plan, new_time, new_position, current_day, verbose)

                if success:
                    return True, plan
                # 如果后续失败，回溯撤销本次安排，尝试下一个景点
                plan[current_day]["activities"].pop()
                attractions_visiting.pop()
                spot_type_visiting.pop()
                attraction_names_visiting.pop()

        # The last event in a day: hotel or go-back
         # 如果已经是最后一天，安排返程
        if current_day == query["days"] - 1:
            # go back
            transports_sel = goto(city=query["target_city"],
                                  start=current_position, end=poi_plan["back_transport"]["From"],
                                  start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
            # 判断交通方式，补充票数或车辆数
            # 例如地铁+步行+公交等多段交通，第二段补充票数
            if len(transports_sel) == 3:
                transports_sel[1]["tickets"] = query["people_number"]
             # 例如地铁+步行+公交等多段交通，第二段补充票数
            elif transports_sel[0]["mode"] == "taxi":
                transports_sel[0]["car"] = int(
                    (query["people_number"] - 1) / 4) + 1
            # 判断返程交通工具类型，分别处理火车和飞机
            if "TrainID" in poi_plan["back_transport"]:
                plan[current_day]["activities"].append(
                    {
                        "start_time": poi_plan["back_transport"]["BeginTime"],
                        "end_time": poi_plan["back_transport"]["EndTime"],
                        "start": poi_plan["back_transport"]["From"],
                        "end": poi_plan["back_transport"]["To"],
                        "TrainID": poi_plan["back_transport"]["TrainID"],
                        "type": "train",
                        "transports": transports_sel,
                        "cost": poi_plan["back_transport"]["Cost"],
                        "tickets": query["people_number"]
                    })
            else:
                plan[current_day]["activities"].append(
                    {
                        "start_time": poi_plan["back_transport"]["BeginTime"],
                        "end_time": poi_plan["back_transport"]["EndTime"],
                        "start": poi_plan["back_transport"]["From"],
                        "end": poi_plan["back_transport"]["To"],
                        "FlightID": poi_plan["back_transport"]["FlightID"],
                        "type": "airplane",
                        "transports": transports_sel,
                        "cost": poi_plan["back_transport"]["Cost"],
                        "tickets": query["people_number"]
                    })
            res_bool, res_plan = constraints_validation(query, plan, poi_plan)

            if res_bool:
                return True, res_plan
            else:
                plan[current_day]["activities"].pop()

                print(
                    "[Try the go back transport], but constraints_validation failed...")

                return False, plan
        else:
            # go to hotel
            hotel_sel = poi_plan["accommodation"]

            transports_sel = goto(city=query["target_city"],
                                  start=current_position, end=hotel_sel["name"],
                                  start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
            arrived_time = transports_sel[-1]["end_time"]
            if len(transports_sel) == 3:
                transports_sel[1]["tickets"] = query["people_number"]
            elif transports_sel[0]["mode"] == "taxi":
                transports_sel[0]["car"] = int(
                    (query["people_number"] - 1) / 4) + 1
            # activity_i = {
            #             "position": hotel_sel["name"],
            #             "type": "accommodation",
            #             "room_type": hotel_sel["numbed"],
            #             "transports": transports_sel,
            #             "cost": int(hotel_sel["price"]),
            #             "start_time": arrived_time,
            #             "end_time": "24:00"
            #         }

            activity_i = {
                "position": hotel_sel["name"],
                "type": "accommodation",
                "room_type": hotel_sel["numbed"],
                "transports": transports_sel,
                "cost": hotel_sel["price"],
                "start_time": arrived_time,
                "end_time": "24:00",
                # "rooms": int((query["people_number"] - 1) / hotel_sel["numbed"]) + 1
                "rooms": query["required_rooms"]
            }

            plan[current_day]["activities"].append(activity_i)

            new_time = "00:00"
            new_position = hotel_sel["name"]

            success, plan = search_poi(
                query, poi_plan, plan, new_time, new_position, current_day + 1, verbose)

            if success:
                return True, plan

            else:
                print("Try the go back hotel, failed...")

                plan[current_day]["activities"].pop()

                return False, plan
    else:
        raise Exception("Not Implemented.")

    return False, plan

# 主规划函数，整合交通、酒店等选项，尝试生成可行行程
# query: 查询条件，poi_plan: 规划细节


def search_plan(query, poi_plan):
    source_city = query["start_city"]
    target_city = query["target_city"]

    print(source_city, target_city)

    train_go = intercity_transport.select(
        start_city=source_city, end_city=target_city, intercity_type="train")
    train_back = intercity_transport.select(
        start_city=target_city, end_city=source_city, intercity_type="train")

    # print(train_go)
    # print(train_back)

    flight_go = intercity_transport.select(
        start_city=source_city, end_city=target_city, intercity_type="airplane")
    flight_back = intercity_transport.select(
        start_city=target_city, end_city=source_city, intercity_type="airplane")

    # print(flight_go)
    # print(flight_back)

    flight_go_num = 0 if flight_go is None else flight_go.shape[0]
    train_go_num = 0 if train_go is None else train_go.shape[0]
    flight_back_num = 0 if flight_back is None else flight_back.shape[0]
    train_back_num = 0 if train_back is None else train_back.shape[0]

    # flight_go_num, train_go_num = flight_go.shape[0], train_go.shape[0]
    # flight_back_num, train_back_num = flight_back.shape[0], train_back.shape[0]

    print("from {} to {}: {} flights, {} trains".format(
        source_city, target_city, flight_go_num, train_go_num))
    print("from {} to {}: {} flights, {} trains".format(
        target_city, source_city, flight_back_num, train_back_num))

    if "hotel_feature" in query:
        # hotel_info = hotel_info[hotel_info["featurehoteltype"] in query["hotel_feature"]]
        hotel_info = accommodation.select(
            target_city, "featurehoteltype", lambda x: x in query["hotel_feature"])
    else:
        hotel_info = accommodation.select(target_city, "name", lambda x: True)

    if "hotel_names" in query:
        hotel_info = accommodation.select(
            target_city, "name", lambda x: x == list(query["hotel_names"])[0])

        # print(query["hotel_names"])
        # print(hotel_info)
        # exit(0)

    if "room_type" in query:
        hotel_info = hotel_info[hotel_info["numbed"] == query["room_type"]]

    if "hotel_price" in query:
        hotel_info = hotel_info[hotel_info["price"] <= query["hotel_price"]]

    num_hotel = hotel_info.shape[0]
    # print(hotel_info)

    print("{} accommmodation, {} hotels (satisfied requirments)".format(
        target_city, num_hotel))

    # set_intercity_trannsport(train_go, train_back, flight_go, flight_back)

    global poi_info
    global restaurants_visiting
    global attractions_visiting
    global food_type_visiting
    global spot_type_visiting
    global attraction_names_visiting
    global restaurant_names_visiting

    poi_info = {}
    restaurants_visiting = []
    attractions_visiting = []
    food_type_visiting = []
    spot_type_visiting = []
    attraction_names_visiting = []
    restaurant_names_visiting = []

    poi_info["restaurants"] = restaurants.select(
        target_city, "name", lambda x: True)
    poi_info["attractions"] = attractions.select(
        target_city, "name", lambda x: True)

    poi_info["restaurants_num"] = poi_info["restaurants"].shape[0]
    poi_info["attractions_num"] = poi_info["attractions"].shape[0]

    poi_plan["transport_preference"] = query["transport_preference"]

    for go_i in range(train_go_num + flight_go_num):

        if go_i >= train_go_num:
            poi_plan["go_transport"] = flight_go.iloc[go_i - train_go_num]

            if "intercity_transport_type" in query and query["intercity_transport_type"] != "airplane":
                continue
            if "train_type" in query:
                continue
        else:
            poi_plan["go_transport"] = train_go.iloc[go_i]

            if "intercity_transport_type" in query and query["intercity_transport_type"] != "train":
                continue

            # print(poi_plan["go_transport"])
            # train_id[0]
            # exit(0)
            if "train_type" in query and query["train_type"] != poi_plan["go_transport"]["TrainID"][0]:
                continue

        for back_i in range(flight_back_num + train_back_num - 1, -1, -1):

            # print("go idx: ", go_i, "back id: ", back_i)

            if back_i >= flight_back_num:
                poi_plan["back_transport"] = train_back.iloc[back_i -
                                                             flight_back_num]

                if "intercity_transport_type" in query and query["intercity_transport_type"] != "train":
                    continue

                if "train_type" in query and query["train_type"] != poi_plan["back_transport"]["TrainID"][0]:
                    continue
            else:
                poi_plan["back_transport"] = flight_back.iloc[back_i]

                if "intercity_transport_type" in query and query["intercity_transport_type"] != "airplane":
                    continue

                if "train_type" in query:
                    continue

            print(poi_plan)

            if "cost" in query:

                intercity_cost = (poi_plan["go_transport"]["Cost"] +
                                  poi_plan["back_transport"]["Cost"]) * query["people_number"]

                print("intercity_cost: ", intercity_cost)
                if intercity_cost >= query["cost"]:
                    continue

                else:
                    cost_wo_inter_trans = query["cost"] - intercity_cost

                    # if query["days"] > 1:
                    #     hotel_cost = int(hotel_info.iloc[hotel_i]["price"]) * query["rooms"] * (query["days"] > 1 - 1)

                    #     print("hotel cost: ", hotel_cost)

                    #     query["cost_wo_intercity"] =  query["cost_wo_intercity"] - hotel_cost

                    # if query["cost_wo_intercity"] <= 0:
                    #     continue

                    # print("in-city budget: ", query["cost_wo_intercity"])

            if query["days"] > 1:

                # print(num_hotel)
                for hotel_i in range(num_hotel):

                    # print(hotel_i)

                    poi_plan["accommodation"] = hotel_info.iloc[hotel_i]

                    if "room_type" in query and query["room_type"] != poi_plan["accommodation"]["numbed"]:
                        continue

                    room_type = poi_plan["accommodation"]["numbed"]

                    required_rooms = int(
                        (query["people_number"] - 1) / room_type) + 1

                    if "rooms" in query:

                        if "room_type" in query:
                            pass
                        else:
                            if required_rooms > query["rooms"]:
                                print("Not enough bed")
                                continue

                        required_rooms = query["rooms"]

                    query["required_rooms"] = required_rooms

                    if ("hotel_price" in query) and int(hotel_info.iloc[hotel_i]["price"]) * required_rooms > query["hotel_price"]:
                        continue

                    # print(hotel_info.iloc[hotel_i])
                    # if ("hotel_feature" in query) and (not set(hotel_info.iloc[hotel_i]["featurehoteltype"]) <= query["hotel_feature"]):
                    #     continue

                    if "cost" in query:

                        hotel_cost = int(
                            hotel_info.iloc[hotel_i]["price"]) * required_rooms * (query["days"] - 1)
                        print("hotel cost: ", hotel_cost)
                        query["cost_wo_intercity"] = cost_wo_inter_trans - hotel_cost

                        if query["cost_wo_intercity"] <= 0:
                            continue

                        print("in-city budget: ", query["cost_wo_intercity"])

                    print("search: ...")
                    success, plan = search_poi(
                        query, poi_plan, plan=[], current_time="", current_position="")

                    # exit(0)

                    if success:
                        return True, plan
            else:

                if time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], poi_plan["go_transport"]["EndTime"]):
                    continue

                if "cost" in query:
                    query["cost_wo_intercity"] = cost_wo_inter_trans
                    print("in-city budget: ", query["cost_wo_intercity"])

                print("search: ...")
                success, plan = search_poi(
                    query, poi_plan, plan=[], current_time="", current_position="")

                print(success, plan)
                if success:
                    return True, plan

    return False, {"info": "No Solution"}


# 解析符号化查询，提取约束，调用主规划函数
# query: 查询条件，query_idx: 查询编号
def symbolic_search(query, query_idx):

    global time_before_search
    time_before_search = time.time()

    query["days"] = int(query["hard_logic"][0].split("==")[1])
    query["people_number"] = int(query["hard_logic"][1].split("==")[1])
    query["transport_preference"] = "metro"

    target_city = query["target_city"]
    hotel_info = accommodation.select(target_city, "name", lambda x: True)
    rest_info = restaurants.select(target_city, "name", lambda x: True)
    attr_info = attractions.select(target_city, "name", lambda x: True)

    seen_attr_type_concept = set(attr_info["type"].unique())
    seen_attr_name_concept = set(attr_info["name"].unique())

    seen_hotel_type_concept = set(hotel_info["featurehoteltype"].unique())
    seen_hotel_name_concept = set(hotel_info["name"].unique())
    # print(seen_hotel_type_concept)

    seen_rest_type_concept = set(rest_info["cuisine"].unique())
    seen_rest_name_concept = set(rest_info["name"].unique())

    # for item in query["hard_logic"]:
    #     if item.startswith("rooms=="):
    #         query["rooms"] = int(item.split("==")[1])
    #     if item.startswith("cost<="):
    #         query["cost"] = int(item.split("<=")[1])

    #     if item.startswith("room_type=="):
    #         query["room_type"] = int(item.split("==")[1])

    #     if item.startswith("train_type == "):
    #         query["train_type"] = item.split("'")[1]

    #     if item.endswith(" <= food_type") or item.endswith("food_type"):
    #         ftlist = item.split("<=")[0].split("}")[0].split("{")[1]

    #         food_type_list = []
    #         for f_i in ftlist.split(","):
    #             food_type_list.append(f_i.split("'")[1])

    #         query["food_type"] = set(food_type_list)

    #     # print(item)
    #     if item.startswith("transport_type"):

    #         if ("<=") in item:
    #             str_set = item.split("transport_type<=")[1]
    #         elif ("==") in item:
    #             str_set = item.split("transport_type==")[1]

    #         if "taxi" in str_set:
    #             query["transport_preference"] = "taxi"
    #         elif "metro" in str_set:
    #             query["transport_preference"] = "metro"
    #         else:
    #             query["transport_preference"] = "walk"

    #     if item.startswith("hotel_price<="):
    #         query["hotel_price"] = int(item.split("<=")[1])

    #     if item.endswith("intercity_transport") or item.startswith("intercity_transport"):
    #         query["intercity_transport_type"] = item.split("'")[1]

    #     if item.endswith("spot_type"):
    #         stlist = item.split("<=")[0].split("}")[0].split("{")[1]

    #         spot_type_list = []
    #         for s_i in stlist.split(","):
    #             spot_type_list.append(s_i.split("'")[1])

    #         query["spot_type"] = set(spot_type_list)

    #     if item.endswith("attraction_names"):
    #         stlist = item.split("<=")[0].split("}")[0].split("{")[1]

    #         spot_name_list = []
    #         for s_i in stlist.split(","):
    #             spot_name_list.append(s_i.split("'")[1])

    #         query["attraction_names"] = set(spot_name_list)

    #     if item.endswith("restaurant_names"):
    #         stlist = item.split("<=")[0].split("}")[0].split("{")[1]

    #         res_name_list = []
    #         for s_i in stlist.split(","):
    #             res_name_list.append(s_i.split("'")[1])

    #         query["restaurant_names"] = set(res_name_list)

    #     if item.endswith("hotel_feature"):
    #         stlist = item.split("<=")[0].split("}")[0].split("{")[1]

    #         hf_list = []
    #         for s_i in stlist.split(","):
    #             hf_list.append(s_i.split("'")[1])

    #         query["hotel_feature"] = set(hf_list)
    query["hard_logic_unseen"] = []

    for idx, item in enumerate(query["hard_logic"]):
        if item.startswith("rooms=="):

            # print(item)
            query["rooms"] = int(item.split("==")[1])

        if item.startswith("cost<="):
            query["cost"] = int(item.split("<=")[1])

        if item.startswith("room_type=="):
            query["room_type"] = int(item.split("==")[1])

        if item.startswith("train_type == "):
            query["train_type"] = item.split("'")[1]

        if item.endswith(" <= food_type") or item.endswith("food_type"):
            ftlist = item.split("<=")[0].split("}")[0].split("{")[1]

            seen_list = []
            unseen_list = []
            for s_i in ftlist.split(","):
                str_i = s_i.split("'")[1]
                if str_i in seen_rest_type_concept:
                    seen_list.append(str_i)
                else:
                    unseen_list.append(str_i)

            if len(seen_list) > 0:
                query["food_type"] = set(seen_list)
                query["hard_logic"][idx] = str(
                    query["food_type"]) + "<=food_type"
            else:
                query["hard_logic"][idx] = " 3 < 33"

            if len(unseen_list) > 0:
                query["food_type_unseen"] = set(unseen_list)
                query["hard_logic_unseen"].append(
                    str(query["food_type_unseen"]) + "<=food_type")

        # print(item)
        if item.startswith("transport_type"):

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

        if item.startswith("hotel_price<="):
            query["hotel_price"] = int(item.split("<=")[1])

        if item.endswith("intercity_transport") or item.startswith("intercity_transport"):
            query["intercity_transport_type"] = item.split("'")[1]

        if item.endswith("spot_type"):
            stlist = item.split("<=")[0].split("}")[0].split("{")[1]

            spot_type_list = []

            seen_list = []
            unseen_list = []

            # for s_i in stlist.split(","):
            #     spot_type_list.append(s_i.split("'")[1])

            #     str_i = s_i.split("'")[1]

            # query["spot_type"] = set(spot_type_list)

            # spot_name_list = []
            for s_i in stlist.split(","):
                # spot_name_list.append(s_i.split("'")[1])
                str_i = s_i.split("'")[1]
                # res_name_list.append()

                if str_i in seen_attr_type_concept:
                    seen_list.append(str_i)
                else:
                    unseen_list.append(str_i)

            # query["attraction_names"] = set(spot_name_list)
            if len(seen_list) > 0:
                query["spot_type"] = set(seen_list)
                query["hard_logic"][idx] = str(
                    query["spot_type"]) + "<=spot_type"
            else:
                query["hard_logic"][idx] = " 3 < 33"

            if len(unseen_list) > 0:
                query["spot_type_unseen"] = set(unseen_list)
                query["hard_logic_unseen"].append(
                    str(query["spot_type_unseen"]) + "<=spot_type")

        if item.endswith("attraction_names"):
            stlist = item.split("<=")[0].split("}")[0].split("{")[1]

            seen_list = []
            unseen_list = []

            spot_name_list = []
            for s_i in stlist.split(","):
                # spot_name_list.append(s_i.split("'")[1])
                str_i = s_i.split("'")[1]
                # res_name_list.append()

                if str_i in seen_attr_name_concept:
                    seen_list.append(str_i)
                else:
                    unseen_list.append(str_i)

            # query["attraction_names"] = set(spot_name_list)
            if len(seen_list) > 0:
                query["attraction_names"] = set(seen_list)
                query["hard_logic"][idx] = str(
                    query["attraction_names"]) + "<=attraction_names"
            else:
                query["hard_logic"][idx] = " 3 < 33"

            if len(unseen_list) > 0:
                query["attraction_names_unseen"] = set(unseen_list)
                query["hard_logic_unseen"].append(
                    str(query["attraction_names_unseen"]) + "<=attraction_names")

        if item.endswith("restaurant_names"):
            stlist = item.split("<=")[0].split("}")[0].split("{")[1]

            res_name_list = []

            seen_list = []
            unseen_list = []

            for s_i in stlist.split(","):
                str_i = s_i.split("'")[1]
                # res_name_list.append()

                if str_i in seen_rest_name_concept:
                    seen_list.append(str_i)
                else:
                    unseen_list.append(str_i)

            # query["restaurant_names"] = set(res_name_list)

            if len(seen_list) > 0:
                query["restaurant_names"] = set(seen_list)
                query["hard_logic"][idx] = str(
                    query["restaurant_names"]) + "<=restaurant_names"
            else:
                query["hard_logic"][idx] = " 3 < 33"

            if len(unseen_list) > 0:
                query["restaurant_names_unseen"] = set(unseen_list)
                query["hard_logic_unseen"].append(
                    str(query["restaurant_names_unseen"]) + "<=restaurant_names")

            # seen_rest_name_concept

        if item.endswith("hotel_names"):
            stlist = item.split("<=")[0].split("}")[0].split("{")[1]

            res_name_list = []

            seen_list = []
            unseen_list = []

            for s_i in stlist.split(","):
                str_i = s_i.split("'")[1]
                # res_name_list.append()

                if str_i in seen_hotel_name_concept:
                    seen_list.append(str_i)
                else:
                    unseen_list.append(str_i)

            # query["restaurant_names"] = set(res_name_list)

            if len(seen_list) > 0:
                query["hotel_names"] = set(seen_list)
                query["hard_logic"][idx] = str(
                    query["hotel_names"]) + "<=hotel_names"
            else:
                query["hard_logic"][idx] = " 3 < 33"

            if len(unseen_list) > 0:
                query["hotel_names_unseen"] = set(unseen_list)
                query["hard_logic_unseen"].append(
                    str(query["hotel_names_unseen"]) + "<=hotel_names")

            # seen_rest_name_concept

        if item.endswith("hotel_feature"):
            stlist = item.split("<=")[0].split("}")[0].split("{")[1]

            hf_list = []

            seen_list = []
            unseen_list = []
            for s_i in stlist.split(","):
                str_i = s_i.split("'")[1]
                if str_i in seen_hotel_type_concept:
                    seen_list.append(str_i)
                else:
                    unseen_list.append(str_i)

            if len(seen_list) > 0:
                query["hotel_feature"] = set(seen_list)
                query["hard_logic"][idx] = str(
                    query["hotel_feature"]) + "<=hotel_feature"
            else:
                query["hard_logic"][idx] = " 3 < 33"

            if len(unseen_list) > 0:
                query["hotel_feature_unseen"] = set(unseen_list)
                query["hard_logic_unseen"].append(
                    str(query["hotel_feature_unseen"]) + "<=hotel_feature")

    print(query)

    # exit(0)

    success, plan = search_plan(query, {"query_idx": query_idx})

    print(success, plan)

    return success, plan

    # Env = ReactEnv(None, "")
    # cmd_str = "attractions_keys('{}')".format(target_city)
    # res = Env.run(cmd_str)
    # print(res)
    # cmd_str = "attractions_select('{}', '{}', {})".format(target_city, "name", "lambda x:True")
    # res = Env.run(cmd_str)
    # print(res)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# 主程序入口，命令行参数解析与批量测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--level', '-l', type=str, default="easy", choices=[
                        "easy", "medium", "human", "medium_modified"], help="query subset")
    parser.add_argument('--index', '-i', type=int,
                        default=None, help="query index")
    parser.add_argument('--start', '-s', type=int,
                        default=None, help="start query index")

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

    test_input = load_json_file("../data/{}.json".format(query_level))

    success_count, total = 0, 0

    if not os.path.exists("results/{}".format(query_level)):
        os.makedirs("results/{}".format(query_level))

    result_dir = "results/{}".format(query_level)

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

    with open("results/{}/fail_list.txt".format(query_level), "a+") as dump_f:
        dump_f.write(datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')+"\n")
        dump_f.write("testing range: {}".format(str(test_idx))+"\n")

    for idx in test_idx:

        query_i = test_input[idx]
        # print(query_i)

        # query_i = load_json_file("results/{}/query_{}.json".format(query_level, idx))

        # print(query_i)
        # exit(0)

        # with open("results/{}/query_{}.json".format(query_level, idx), "w", encoding="utf8") as dump_f:
        #     json.dump(query_i, dump_f, ensure_ascii=False, indent=4)

        print("query {}/{}".format(idx, len(test_input)))

        sys.stdout = Logger(
            result_dir + "/plan_{}.log".format(idx), sys.stdout)
        sys.stderr = Logger(
            result_dir + "/plan_{}.error".format(idx), sys.stderr)

        global avialable_plan
        avialable_plan = {}

        # try:
        success, plan = symbolic_search(query_i, idx)

        success_count += int(success)
        total += 1

        if not success:
            fail_list.append(idx)

            with open(result_dir + "/fail_list.txt".format(query_level), "a+") as dump_f:
                dump_f.write(str(idx)+"\n")

            plan = avialable_plan

            with open(result_dir + "/plan_{}.json".format(idx), "w", encoding="utf8") as dump_f:
                json.dump(plan, dump_f, ensure_ascii=False,
                          indent=4,  cls=NpEncoder)

        else:
            with open(result_dir + "/plan_{}.json".format(idx), "w", encoding="utf8") as dump_f:
                json.dump(plan, dump_f, ensure_ascii=False,
                          indent=4,  cls=NpEncoder)

        # except:
        #     total +=1
        #     fail_list.append(idx)

        # if success_count < total:
        #     print("fail: ", idx)
        #     break

    print("success rate [{}]: {}/{}".format(query_level, success_count, total))

    res_stat = {
        "success": success_count,
        "total": total,
        "fail_list": fail_list,
    }

    with open("results/{}/result_stat_{}.json".format(query_level, str(time.time())), "w", encoding="utf8") as dump_f:
        json.dump(res_stat, dump_f, ensure_ascii=False, indent=4)

    with open("results/{}/fail_list.txt".format(query_level), "a+") as dump_f:
        dump_f.write(
            "success rate [{}]: {}/{}".format(query_level, success_count, total)+"\n")
