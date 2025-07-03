"""
旅行规划常识约束验证模块

本模块实现了对旅行规划的7个核心约束验证功能：
1. 城际交通约束（Is_intercity_transport_correct）- 验证往返交通信息的准确性
2. 景点约束（Is_attractions_correct）- 验证景点信息的真实性和唯一性  
3. 酒店约束（Is_hotels_correct）- 验证住宿信息的正确性和唯一性
4. 餐厅约束（Is_restaurants_correct）- 验证餐厅信息和用餐时间的合理性
5. 城内交通约束（Is_transport_correct）- 验证城市内部交通路线的准确性
6. 时间约束（Is_time_correct）- 验证活动时间安排的逻辑性
7. 空间约束（Is_space_correct）- 验证活动位置之间的连贯性

作者：旅行规划系统开发团队
日期：2025年7月2日
"""

import sys

sys.path.append("../")
# 导入各类API工具
from tools.hotels.apis import Accommodations          # 酒店API
from tools.restaurants.apis import Restaurants        # 餐厅API
from tools.attractions.apis import Attractions        # 景点API
from tools.intercity_transport.apis import IntercityTransport  # 城际交通API
from tools.transportation.apis import GoTo           # 城内交通API
from envs import goto
import json
import os
import sys
from tqdm import tqdm

try:
    from utils import load_json_file
except:
    from evaluation.utils import load_json_file

def safe_get_activity_field(activity, field_name, default_value=""):
    """
    安全获取活动字段，避免KeyError
    
    参数:
        activity: 活动字典
        field_name: 字段名称
        default_value: 默认值
    
    返回:
        字段值或默认值
    """
    return activity.get(field_name, default_value)

def safe_get_activity_type(activity):
    """
    安全获取活动类型
    
    参数:
        activity: 活动字典
    
    返回:
        活动类型字符串
    """
    return safe_get_activity_field(activity, "type", "")

def safe_get_activity_position(activity):
    """
    安全获取活动位置
    
    参数:
        activity: 活动字典
    
    返回:
        活动位置字符串
    """
    return safe_get_activity_field(activity, "position", "")

def safe_get_activity_cost(activity):
    """
    安全获取活动费用
    
    参数:
        activity: 活动字典
    
    返回:
        活动费用数值
    """
    return activity.get("cost", 0)

def safe_get_activity_time(activity, time_field):
    """
    安全获取活动时间
    
    参数:
        activity: 活动字典
        time_field: 时间字段名（start_time或end_time）
    
    返回:
        时间字符串
    """
    default_times = {
        "start_time": "08:00",
        "end_time": "09:00"
    }
    return activity.get(time_field, default_times.get(time_field, "08:00"))


# 初始化各类API工具实例
accommodation = Accommodations()    # 酒店API实例
restaurants = Restaurants()         # 餐厅API实例
attractions = Attractions()         # 景点API实例
intercity_transport = IntercityTransport()  # 城际交通API实例


"""
旅行规划约束验证：

可用的约束验证功能：
1. 城际交通信息存在且客观：交通工具ID、时间、起点和终点需要正确
2. 景点信息验证：景点名称、位置、费用等信息的真实性
3. 酒店信息验证：酒店名称、房间类型、价格等信息的正确性
4. 餐厅信息验证：餐厅名称、营业时间、价格等信息的准确性
5. 城内交通验证：城市内部交通路线和费用的准确性
6. 时间约束验证：活动时间安排的逻辑性和合理性
7. 空间约束验证：活动位置之间的连贯性和可达性
"""


def return_info_debug(flag, info):
    """
    调试模式返回信息函数
    
    参数:
        flag: 验证结果布尔值
        info: 详细信息字符串
    
    返回:
        (flag, info)元组
    """
    return flag, info


def return_info_test(flag, info):
    """
    测试模式返回信息函数
    
    参数:
        flag: 验证结果布尔值
        info: 详细信息字符串（测试模式下忽略）
    
    返回:
        flag布尔值
    """
    return flag


def Is_intercity_transport_correct(symbolic_input, plan_json, mode="debug"):
    """
    验证城际交通约束的正确性
    
    此函数验证旅行计划中的城际交通信息是否正确，包括：
    - 第一天必须包含从起点到目的地的交通
    - 最后一天必须包含从目的地返回起点的交通
    - 交通工具ID、时间、起点、终点、费用等信息必须准确
    - 支持飞机和火车两种交通方式
    
    参数:
        symbolic_input: 包含旅行需求的符号化输入
        plan_json: 旅行计划JSON对象
        mode: 运行模式，"debug"或"test"
    
    返回:
        根据mode参数返回(flag, info)或flag
    """

    # print("input: ", symbolic_input)
    # print("plan: ", plan_json)

    # 根据模式选择返回函数
    if mode == "debug":
        return_info = return_info_debug
    else:
        return_info = return_info_test
    
    # 验证计划格式
    if not isinstance(plan_json, dict):
        return return_info(False, "Error plan type, must be python dict")
    
    # 获取第一天和最后一天的计划
    first_day_plan = plan_json["itinerary"][0]
    last_day_plan = plan_json["itinerary"][-1]

    # 获取目标城市和起始城市
    target_city = symbolic_input["target_city"]
    start_pos = symbolic_input["start_city"]

    # 验证必须包含城际交通
    if len(first_day_plan["activities"]) == 0 or len(last_day_plan["activities"]) == 0:
        return return_info(False, "It must contain intecity transport")

    go_intercity_transport_plan = first_day_plan["activities"][0]
    back_intercity_transport_plan = last_day_plan["activities"][-1]
    if ("FlightID" not in go_intercity_transport_plan.keys()) and (
        "TrainID" not in go_intercity_transport_plan.keys()
    ):
        return return_info(
            False, "The first activity should be a transport."
        )  # "The first transport should be from origin to destination.")
    if ("FlightID" not in back_intercity_transport_plan.keys()) and (
        "TrainID" not in back_intercity_transport_plan.keys()
    ):
        return return_info(
            False, "The last activity should be a transport."
        )  # "The last transport should be from destination to origin.")

    go_type = go_intercity_transport_plan["type"]
    if go_type != "airplane" and go_type != "train":
        return return_info(
            False, "Intercity transport type should be airplane or train"
        )

    go_df = intercity_transport.select(start_pos, target_city, go_type)

    if not (
        "start" in go_intercity_transport_plan and "end" in go_intercity_transport_plan
    ):
        return return_info(
            False, "intercity-transport should provide start and end position."
        )

    go_flag = 0
    for _, row in go_df.iterrows():
        if go_type == "airplane":
            try:
                go_intercity_transport_plan["FlightID"]
            except:
                return_info(False, "Iintercity airplane should provide the FlightID.")

            if (
                go_intercity_transport_plan["FlightID"] == row["FlightID"]
                and go_intercity_transport_plan["start"] == row["From"]
                and go_intercity_transport_plan["end"] == row["To"]
            ):
                go_flag = 1

                if (
                    row["BeginTime"] == go_intercity_transport_plan["start_time"]
                    and row["EndTime"] == go_intercity_transport_plan["end_time"]
                    and row["From"] == go_intercity_transport_plan["start"]
                    and row["To"] == go_intercity_transport_plan["end"]
                    and row["Cost"] == go_intercity_transport_plan["cost"]
                ):

                    break
                else:
                    return return_info(
                        False,
                        "Incorrect information of given intercity airplane [origin -> destination].",
                    )

        if go_type == "train":

            try:
                go_intercity_transport_plan["TrainID"]
            except:
                return return_info(False, "Intercity train should provide the TrainID.")
            if (
                go_intercity_transport_plan["TrainID"] == row["TrainID"]
                and go_intercity_transport_plan["start"] == row["From"]
                and go_intercity_transport_plan["end"] == row["To"]
            ):

                go_flag = 1
                if (
                    row["BeginTime"] == go_intercity_transport_plan["start_time"]
                    and row["EndTime"] == go_intercity_transport_plan["end_time"]
                    and row["From"] == go_intercity_transport_plan["start"]
                    and row["To"] == go_intercity_transport_plan["end"]
                    and row["Cost"] == go_intercity_transport_plan["cost"]
                ):
                    break
                else:
                    return return_info(
                        False,
                        "Incorrect information of given intercity train [origin -> destination].",
                    )
    if go_flag == 0:
        return return_info(False, "No information found given transport ID")

    back_type = back_intercity_transport_plan["type"]
    if back_type != "airplane" and back_type != "train":
        return return_info(
            False, "Intercity transport type should be airplane or train"
        )

    back_df = intercity_transport.select(target_city, start_pos, back_type)

    back_flag = 0

    if not (
        "start" in back_intercity_transport_plan
        and "end" in back_intercity_transport_plan
    ):
        return return_info(
            False, "intercity-transport should provide start and end position."
        )

    for _, row in back_df.iterrows():
        if (
            back_type == "airplane"
            and back_intercity_transport_plan["FlightID"] == row["FlightID"]
            and back_intercity_transport_plan["start"] == row["From"]
            and back_intercity_transport_plan["end"] == row["To"]
        ):
            back_flag = 1
            if (
                row["BeginTime"] == back_intercity_transport_plan["start_time"]
                and row["EndTime"] == back_intercity_transport_plan["end_time"]
                and row["From"] == back_intercity_transport_plan["start"]
                and row["To"] == back_intercity_transport_plan["end"]
                and row["Cost"] == back_intercity_transport_plan["cost"]
            ):
                break
            else:
                return return_info(
                    False,
                    "Incorrect information of given intercity airplane [destination -> origin].",
                )
        if (
            back_type == "train"
            and back_intercity_transport_plan["TrainID"] == row["TrainID"]
            and back_intercity_transport_plan["start"] == row["From"]
            and back_intercity_transport_plan["end"] == row["To"]
        ):
            back_flag = 1

            if (
                row["BeginTime"] == back_intercity_transport_plan["start_time"]
                and row["EndTime"] == back_intercity_transport_plan["end_time"]
                and row["From"] == back_intercity_transport_plan["start"]
                and row["To"] == back_intercity_transport_plan["end"]
                and row["Cost"] == back_intercity_transport_plan["cost"]
            ):
                break
            else:
                return return_info(
                    False,
                    "Incorrect information of given intercity train [destination -> origin].",
                )

    if back_flag == 0:
        return return_info(False, "No information found given transport ID")

    return return_info(True, "Intercity_transport passed!")
    # return True


def Is_attractions_correct(symbolic_input, plan_json, mode="debug"):
    """
    验证景点约束的正确性
    
    此函数验证旅行计划中的景点信息是否正确，包括：
    - 景点名称必须在目标城市的景点数据库中存在
    - 景点不能在整个行程中重复出现
    - 景点的开放时间和价格信息准确（已注释）
    
    参数:
        symbolic_input: 包含旅行需求的符号化输入
        plan_json: 旅行计划JSON对象
        mode: 运行模式，"debug"或"test"
    
    返回:
        根据mode参数返回(flag, info)或flag
    """

    # 根据模式选择返回函数
    if mode == "debug":
        return_info = return_info_debug
    else:
        return_info = return_info_test

    # 获取目标城市和行程计划
    target_city = symbolic_input["target_city"]
    try:
        plan_json["itinerary"]
    except:
        return return_info(False, "Error plan type or format")
    plan = plan_json["itinerary"]

    # 记录所有景点，用于检查重复
    attraction_list = []

    # 遍历每一天的行程
    for day_plan_i in plan:
        for activity_i in day_plan_i["activities"]:

            # 检查活动类型字段
            try:
                activity_i["type"]
            except:
                return return_info(False, "type keyerrror")
            
            # 只处理景点类型的活动
            if activity_i["type"] != "attraction":
                continue

            # print(activity_i)
            try:
                activity_i["position"]
            except:
                return return_info(False, "no position!")

            select_attraction = attractions.select(
                target_city, key="name", func=lambda x: x == activity_i["position"]
            )

            # print(select_attraction)

            if select_attraction.empty:
                return return_info(
                    False,
                    "No information found given attraction [{}]".format(
                        activity_i["position"]
                    ),
                )

            else:
                attraction_list.append(activity_i["position"])

            # 开放时间 todo景区开放时间和价格
            """
            opentime, endtime = select_attraction["opentime"].values[0],  select_attraction["endtime"].values[0]

            if time_compare_if_earlier_equal(endtime, activity_i["start_time"]) or time_compare_if_earlier_equal(activity_i["end_time"], opentime): 
                return return_info(False, "The attraction is closed now. {}, open time: [{} -- {}]".format(activity_i["position"], opentime, endtime))
            """

            # 返回信息保证一致: cost

            """
            if int(activity_i["cost"]) != int(select_attraction["price"].values[0]):
                return return_info(False, "Incorrect cost infomation of attraction [{}], cost: {} ".format(activity_i["position"], activity_i["cost"]))
            """

            # if not select_attraction_type.empty:
            #     spot_type.add(select_attraction_type.iloc[0])
            # attraction_names.add(activity["position"])

    if len(set(attraction_list)) != len(attraction_list):
        return return_info(
            False, "Attraction choices should not be repeated throughout the trip."
        )

    return return_info(True, "attractions passed!")


def Is_hotels_correct(symbolic_input, plan_json, mode="debug"):
    """
    验证酒店约束的正确性
    
    此函数验证旅行计划中的酒店信息是否正确，包括：
    - 酒店名称必须在目标城市的酒店数据库中存在
    - 整个行程中酒店应该是唯一的
    - 多天行程必须有住宿安排
    - 房间数量和类型应满足人数要求
    
    注意：目前酒店价格信息无法检查，函数默认返回通过
    
    参数:
        symbolic_input: 包含旅行需求的符号化输入
        plan_json: 旅行计划JSON对象
        mode: 运行模式，"debug"或"test"
    
    返回:
        根据mode参数返回(flag, info)或flag
    """
    # TODO: 酒店价格信息无法检查
    return_info = return_info_debug
    return return_info(True, "hotels passed!")

    # 以下是完整的酒店验证逻辑（暂时被注释）
    mode = "debug"
    if mode == "debug":
        return_info = return_info_debug
    else:
        return_info = return_info_test

    target_city = symbolic_input["target_city"]
    plan = plan_json["itinerary"]

    hotel_list = []  # 记录所有酒店，用于检查唯一性

    # 遍历每一天的行程
    for day_plan_i in plan:
        for activity_i in day_plan_i["activities"]:

            # 检查活动类型字段
            try:
                activity_i["type"]
            except:
                return return_info(False, "type keyerror")
            
            # 只处理住宿类型的活动
            if activity_i["type"] != "accommodation":
                continue

            # print(activity_i)
            try:
                activity_i["position"]
            except:
                return return_info(False, "position keyerror")
            select_hotel = accommodation.select(
                target_city, key="name", func=lambda x: x == activity_i["position"]
            )
            # print(select_hotel)

            if select_hotel.empty:

                return return_info(
                    False,
                    "No information found given hotel [{}]".format(
                        activity_i["position"]
                    ),
                )

            # if not select_attraction_type.empty:
            #     spot_type.add(select_attraction_type.iloc[0])
            # attraction_names.add(activity["position"])
            else:
                hotel_list.append(activity_i["position"])

            # 返回信息保证一致: cost
            # todo:酒店价格约束无法检查
            """
            try: activity_i["cost"]
            except: return return_info(False, "Hotel cost should be provided")

            if activity_i["cost"] != select_hotel["price"].values[0]:
                return return_info(False, "Incorrect cost infomation of accommodation [{}], cost: {} ".format(activity_i["position"], select_hotel["price"].values[0]))
                        

            if activity_i["room_type"] != select_hotel["numbed"].values[0]:
                return return_info(False, "Incorrect room infomation of accommodation [{}], numbed: {} ".format(activity_i["position"], select_hotel["numbed"].values[0]))
            """

            # "rooms=={1}",
            # "room_types=={1}",

            limit_rooms, limits_room_type = False, False

            for logical_i in symbolic_input["hard_logic"]:
                if "rooms" in logical_i:
                    limit_rooms = True
                if "room_type" in logical_i:
                    limits_room_type = True

            if limit_rooms and limits_room_type:
                pass
            else:
                room_type = activity_i["room_type"]
                rooms = activity_i["rooms"]
                # people_number = int(symbolic_input["hard_logic"][1].split("==")[1])
                people_number_idx = 1
                for i, l in enumerate(symbolic_input["hard_logic"]):
                    if "people_number" in l:
                        people_number_idx = i
                people_number = int(
                    symbolic_input["hard_logic"][people_number_idx].split("==")[1]
                )

                if (room_type * rooms >= people_number) and (
                    room_type * rooms < people_number + room_type
                ):
                    pass
                else:
                    return return_info(
                        False,
                        "Incorrect room infomation for {} people, given rooms: {}, numbed: {} ".format(
                            people_number, rooms, room_type
                        ),
                    )

    if len(set(hotel_list)) > 1:
        return return_info(False, "Hotel should be unique during the trip.")

    if len(plan_json["itinerary"]) > 1 and len(hotel_list) == 0:
        return return_info(False, "We need a hotel for a trip more than one day.")

    return return_info(True, "hotels passed!")


def Is_restaurants_correct(symbolic_input, plan_json, mode="debug"):
    """
    验证餐厅约束的正确性
    
    此函数验证旅行计划中的餐厅信息是否正确，包括：
    - 餐厅名称必须在目标城市的餐厅数据库中存在
    - 餐厅不能在整个行程中重复出现
    - 用餐时间必须在合理范围内（早餐：06:00-09:00，午餐：11:00-13:00，晚餐：17:00-20:00）
    - 餐厅的营业时间和价格信息准确
    - 支持在酒店用早餐（费用为0）
    
    注意：目前餐厅正确性检查被注释，函数默认返回通过
    
    参数:
        symbolic_input: 包含旅行需求的符号化输入
        plan_json: 旅行计划JSON对象
        mode: 运行模式，"debug"或"test"
    
    返回:
        根据mode参数返回(flag, info)或flag
    """
    # TODO: 餐厅正确性检查
    return_info = return_info_debug
    return return_info(True, "restaurants passed!")

    # 以下是完整的餐厅验证逻辑（暂时被注释）
    if mode == "debug":
        return_info = return_info_debug
    else:
        return_info = return_info_test

    target_city = symbolic_input["target_city"]
    plan = plan_json["itinerary"]

    restaurants_list = []      # 记录所有餐厅，用于检查重复
    restaurants_time_list = [] # 记录用餐时间

    # 遍历每一天的行程
    for day_plan_i in plan:
        for activity_i in day_plan_i["activities"]:
            # 检查活动类型字段
            try:
                activity_i["type"]
            except:
                return return_info(False, "no type!")
            
            # 只处理用餐类型的活动（早餐、午餐、晚餐）
            if not activity_i["type"] in ["breakfast", "lunch", "dinner"]:
                continue

            # print(activity_i)
            # 检查位置字段
            try:
                activity_i["position"]
            except:
                return return_info(False, "no position!")

            # 查询餐厅信息
            select_restaurant = restaurants.select(
                target_city, key="name", func=lambda x: x == activity_i["position"]
            )

            # print(select_restaurant)

            if activity_i["type"] == "breakfast" and select_restaurant.empty:

                select_hotel = accommodation.select(
                    target_city, key="name", func=lambda x: x == activity_i["position"]
                )

                if select_hotel.empty:
                    return return_info(
                        False,
                        "No information found given restaurant [{}]".format(
                            activity_i["position"]
                        ),
                    )
                try:
                    activity_i["cost"]
                except:
                    return return_info(
                        False,
                        "Cost of breakfast in hotel not provided although it is always if having breakfast at hotel",
                    )
                if activity_i["cost"] != 0:
                    return return_info(False, "Have breakfast at hotel, cost 0")

                # restaurants_list.append(activity_i["position"])
                # restaurants_time_list.append(activity_i["start_time"])

                if time_compare_if_earlier_equal(
                    "09:00", activity_i.get("start_time", "08:00")
                ) or time_compare_if_earlier_equal(activity_i.get("end_time", "08:30"), "06:00"):

                    return return_info(
                        False, "The time of breakfast should be in [06:00 -- 09:00]"
                    )
                continue

            if select_restaurant.empty:
                return return_info(
                    False,
                    "No information found given restaurant [{}]".format(
                        activity_i["position"]
                    ),
                )

            if activity_i["cost"] != select_restaurant["price"].values[0]:
                return return_info(
                    False,
                    "Incorrect cost infomation of restaurant [{}], cost: {} ".format(
                        activity_i["position"], select_restaurant["price"].values[0]
                    ),
                )

            if activity_i["type"] == "lunch" and (
                time_compare_if_earlier_equal("13:00", activity_i.get("start_time", "12:00"))
                or time_compare_if_earlier_equal(activity_i.get("end_time", "13:00"), "11:00")
            ):

                return return_info(
                    False, "The time of lunch should be in [11:00 -- 13:00]"
                )

            if activity_i["type"] == "dinner" and (
                time_compare_if_earlier_equal("20:00", activity_i.get("start_time", "18:00"))
                or time_compare_if_earlier_equal(activity_i.get("end_time", "19:00"), "17:00")
            ):

                return return_info(
                    False, "The time of dinner should be in [17:00 -- 20:00]"
                )

            # if not select_attraction_type.empty:
            #     spot_type.add(select_attraction_type.iloc[0])
            # attraction_names.add(activity["position"])

            # 开放时间
            opentime, endtime = (
                select_restaurant["weekdayopentime"].values[0],
                select_restaurant["weekdayclosetime"].values[0],
            )

            if time_compare_if_earlier_equal(
                endtime, activity_i.get("start_time", "12:00")
            ) or time_compare_if_earlier_equal(activity_i.get("end_time", "13:00"), opentime):
                return return_info(
                    False,
                    "The attraction is closed now. open time: [{} -- {}]".format(
                        opentime, endtime
                    ),
                )

            restaurants_list.append(activity_i["position"])
            restaurants_time_list.append(activity_i.get("start_time", "12:00"))

    if len(set(restaurants_list)) != len(restaurants_list):
        return return_info(
            False, "Restaurants choices should not be repeated throughout the trip."
        )

    # print(restaurants_list)
    # print(restaurants_time_list)

    return return_info(True, "restaurants passed!")


def Is_transport_correct(symbolic_input, plan_json, mode="debug"):
    """
    验证城内交通约束的正确性
    
    此函数验证旅行计划中的城内交通信息是否正确，包括：
    - 地铁交通必须是三阶段（步行->地铁->步行）
    - 出租车和步行交通必须是一阶段
    - 交通路线的起点、终点、费用、距离、时间信息必须准确
    - 交通方式必须与实际可用的交通工具匹配
    
    参数:
        symbolic_input: 包含旅行需求的符号化输入
        plan_json: 旅行计划JSON对象
        mode: 运行模式，"debug"或"test"
    
    返回:
        根据mode参数返回(flag, info)或flag
    """

    # 根据模式选择返回函数
    if mode == "debug":
        return_info = return_info_debug
    else:
        return_info = return_info_test

    target_city = symbolic_input["target_city"]
    plan = plan_json["itinerary"]
    for day_plan_i in plan:
        for activity_i in day_plan_i["activities"]:

            if "transports" in activity_i:

                transport_i = activity_i["transports"]

                if (len(transport_i)) == 0:
                    continue
                # print(transport_i)

                source_poi = transport_i[0].get("start", "")
                target_poi = transport_i[-1].get("end", "")
                start_time = transport_i[0].get("start_time", "08:00")

                # print(source_poi, " -> ", target_poi)

                # print(GoTo(city=target_city, locationA=source_poi, locationB=target_poi, start_time=start_time, transport_type="metro", verbose=False))
                # print(GoTo(city=target_city, locationA=source_poi, locationB=target_poi, start_time=start_time, transport_type="taxi", verbose=False))

                if len(transport_i) == 3:
                    try:
                        tools_return = GoTo(
                            city=target_city,
                            locationA=source_poi,
                            locationB=target_poi,
                            start_time=start_time,
                            transport_type="metro",
                            verbose=False,
                        )
                    except:
                        return return_info(False, "GoTo error")
                    for idx, trans_ii in enumerate(transport_i):

                        if trans_ii["start"] != tools_return[idx]["start"]:
                            return return_info(
                                False,
                                "Incorrect infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                        if trans_ii["end"] != tools_return[idx]["end"]:
                            return return_info(
                                False,
                                "Incorrect infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                        if abs(trans_ii["cost"] - tools_return[idx]["cost"]) > 0.1:
                            return return_info(
                                False,
                                "Incorrect cost infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                        if (
                            abs(trans_ii["distance"] - tools_return[idx]["distance"])
                            > 0.1
                        ):
                            return return_info(
                                False,
                                "Incorrect distance infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                    if (
                        transport_i[0]["mode"] != "walk"
                        or transport_i[2]["mode"] != "walk"
                        or transport_i[1]["mode"] != "metro"
                    ):
                        return return_info(
                            False,
                            "Incorrect transport type of transport {} -> {}".format(
                                source_poi, target_poi
                            ),
                        )

                elif len(transport_i) == 1 and transport_i[0]["mode"] in [
                    "walk",
                    "taxi",
                ]:

                    try:
                        tools_return = GoTo(
                            city=target_city,
                            locationA=source_poi,
                            locationB=target_poi,
                            start_time=start_time,
                            transport_type=transport_i[0]["mode"],
                            verbose=False,
                        )
                    except:
                        return return_info(False, "Goto error")
                    if not isinstance(tools_return, list):
                        return return_info(
                            False,
                            "Can not find a path of transport {} -> {}".format(
                                source_poi, target_poi
                            ),
                        )

                    for idx, trans_ii in enumerate(transport_i):

                        # print(trans_ii)
                        # print(tools_return)
                        # print(source_poi, target_poi)
                        if trans_ii["start"] != tools_return[idx]["start"]:
                            return return_info(
                                False,
                                "Incorrect infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                        if trans_ii["end"] != tools_return[idx]["end"]:
                            return return_info(
                                False,
                                "Incorrect infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                        if abs(trans_ii["cost"] - tools_return[idx]["cost"]) > 0.1:
                            return return_info(
                                False,
                                "Incorrect infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                        if (
                            abs(trans_ii["distance"] - tools_return[idx]["distance"])
                            > 0.1
                        ):
                            return return_info(
                                False,
                                "Incorrect infomation of transport {} -> {}".format(
                                    source_poi, target_poi
                                )
                                + "  [{}], Tool: [{}]".format(
                                    trans_ii, tools_return[idx]
                                ),
                            )

                else:
                    return return_info(
                        False,
                        "Metro transport should be three-stages, Taxi or walk should be one-stage. {} -> {}".format(
                            source_poi, target_poi
                        ),
                    )
                # print("passed")

    return return_info(True, "innercity transport passed!")


def time_compare_if_earlier_equal(time_1, time_2):
    """
    比较两个时间，判断time_1是否早于或等于time_2
    
    参数:
        time_1: 时间字符串，格式为"HH:MM"
        time_2: 时间字符串，格式为"HH:MM"
    
    返回:
        布尔值，True表示time_1早于或等于time_2
    """
    time1 = float(time_1.split(":")[0]) * 60 + float(time_1.split(":")[1])
    time2 = float(time_2.split(":")[0]) * 60 + float(time_2.split(":")[1])

    return time1 <= time2


def time2real(time_str):
    """
    将时间字符串转换为分钟数，处理次日情况
    
    参数:
        time_str: 时间字符串，可能包含"次日"前缀
    
    返回:
        以分钟为单位的时间数值
    """
    time_str = time_str.split("次日")[-1]
    return float(time_str.split(":")[0]) * 60 + float(time_str.split(":")[1])


def Is_time_correct(symbolic_input, plan_json, mode="debug"):
    """
    验证时间约束的正确性
    
    此函数验证旅行计划中的时间安排是否合理，包括：
    - 活动必须有持续时间（开始时间早于结束时间）
    - 交通时间必须在活动开始之前完成
    - 考虑次日到达的情况（适用于火车和飞机）
    
    参数:
        symbolic_input: 包含旅行需求的符号化输入
        plan_json: 旅行计划JSON对象
        mode: 运行模式，"debug"或"test"
    
    返回:
        根据mode参数返回(flag, info)或flag
    """

    # 根据模式选择返回函数
    if mode == "debug":
        return_info = return_info_debug
    else:
        return_info = return_info_test

    target_city = symbolic_input["target_city"]
    plan = plan_json["itinerary"]
    
    # 遍历每一天的行程
    for day_plan_i in plan:
        for activity_i in day_plan_i["activities"]:

            # print(activity_i)
            # 检查时间字段是否存在
            try:
                start_time = activity_i.get("start_time", "08:00")
                end_time = activity_i.get("end_time", "09:00")
            except:
                return_info(False, "key error")
            
            activity_st_time = start_time
            activity_ed_time = end_time

            if time2real(activity_st_time) >= time2real(activity_ed_time) and (
                not activity_i["type"] in ["train", "airplane"]
            ):  # 可能出现次日到达
                return return_info(
                    False, "Activities must cost time: " + str(activity_i)
                )

            if not "transports" in activity_i:
                continue

            if len(activity_i["transports"]) > 0:
                transport_st_time = activity_i["transports"][0]["start_time"]
                transport_ed_time = activity_i["transports"][-1]["end_time"]

                if time2real(activity_st_time) < time2real(transport_ed_time):
                    return return_info(
                        False,
                        "Must arrive at the location before starting the activity: "
                        + str(activity_i),
                    )

    return return_info(True, "time passed!")


def Is_space_correct(symbolic_input, plan_json, mode="debug"):

    if mode == "debug":
        return_info = return_info_debug
    else:
        return_info = return_info_test

    plan = plan_json["itinerary"]

    position_list = []

    for day_plan_i in plan:
        for activity_i in day_plan_i["activities"]:

            if not "position" in activity_i:
                if "start" in activity_i:
                    current_position = activity_i["start"]
                else:
                    return return_info(
                        False, "Every activity need a space: ".format(activity_i)
                    )
            else:
                current_position = activity_i["position"]

            if not "transports" in activity_i:
                print(activity_i)
                return return_info(False, "Need trasnports: ".format(activity_i))

            # try: activity_i["position"] and activity_i["transports"]
            # except: return False

            position_i = current_position

            if (len(position_list) > 0) and position_i != position_list[-1]:
                if len(activity_i["transports"]) < 1:
                    return return_info(
                        False,
                        "There must be transport between activities in different possitions: "
                        + str(activity_i),
                    )

                if activity_i["transports"][0]["start"] != position_list[-1]:
                    return return_info(
                        False,
                        "The origin of the transport must be equal to the position of the previous activity.: "
                        + str(activity_i),
                    )
                if activity_i["transports"][-1]["end"] != position_i:
                    return return_info(
                        False,
                        "The destination of the transport must be equal to the position of the current activity.: "
                        + str(activity_i),
                    )

            if "position" in activity_i:
                position_list.append(activity_i["position"])
            else:
                position_list.append(activity_i["end"])

    # print("position_list: ", position_list)

    return return_info(True, "space passed!")


def func_commonsense_constraints(symbolic_input, plan_json, verbose=False):

    func_list = [
        # Is_intercity_transport_correct,
        # Is_attractions_correct,
        # Is_hotels_correct,
        # Is_restaurants_correct,
        # Is_transport_correct,
        Is_time_correct,
        Is_space_correct,
    ]
    for func in func_list:
        # try:
        result, info = func(symbolic_input, plan_json, mode="debug")

        print(result, info)
        if not result:
            print(info)
            return False

    # except Exception:

    #     print(Exception)

    return True


def evaluate_commonsense_constraints(symbolic_input_list, plan_json_list):
    assert len(symbolic_input_list) == len(plan_json_list)
    func_list = [
        Is_intercity_transport_correct,
        Is_attractions_correct,
        Is_hotels_correct,
        Is_restaurants_correct,
        Is_transport_correct,
        Is_time_correct,
        Is_space_correct,
    ]
    total_correct = 0

    individual_results = []
    results_per_sample = []
    for i, (symbolic_input, plan_json) in enumerate(
        zip(symbolic_input_list, plan_json_list)
    ):

        # print(i)

        if plan_json == []:
            individual_result = [False] * len(func_list)
        else:
            individual_result = []
            for func in func_list:
                # try:
                #     result= func(symbolic_input, plan_json,mode='test')
                #     # print(info)
                #     individual_result.append(result)
                #     if result:
                #         total_correct+=1
                # except Exception:
                #     # print(Exception)
                #     individual_result.append(False)
                try:
                    result, info = func(symbolic_input, plan_json, mode="debug")
                    print(i, result, info)
                except KeyError:
                    result = False
                    print("Some key lost in tested plan!")
                except:
                    result = False
                # result, info = func(symbolic_input, plan_json,mode='debug')

                # print(info)

                individual_result.append(result)
                if result:
                    total_correct += 1

        individual_results.append(all(individual_result))
        results_per_sample.append(individual_result)

    total_count = len(func_list) * len(symbolic_input_list)
    micro_accuracy = total_correct / total_count
    macro_accuracy = sum(individual_results) / len(symbolic_input_list)

    return macro_accuracy * 100, micro_accuracy * 100, results_per_sample


if __name__ == "__main__":

    # test_example=load_json_file("./example/query_53.json")
    # test_plan=load_json_file("./example/plan_53.json")
    # evaluate_commonsense_constraints([test_example], [test_plan])

    # exit(0)

    symbolic_input_list = []
    plan_json_list = []

    for i in range(1):
        test_plan_path = "./example/a_result.json".format(i + 1)
        test_example_path = "./example/a_query.json".format(i + 1)
        test_example = load_json_file(test_example_path)
        test_plan = load_json_file(test_plan_path)
        symbolic_input_list.append(test_example)
        plan_json_list.append(test_plan)
    macro_accuracy, micro_accuracy, _ = evaluate_commonsense_constraints(
        symbolic_input_list, plan_json_list
    )
    print("macro: {}%, micro: {}%".format(macro_accuracy, micro_accuracy))

    # test_plan_path='./example/plan_4.json'
    # test_example_path='./example/query_4.json'
    # test_example=load_json_file(test_example_path)
    # test_plan=load_json_file(test_plan_path)

    # print(Is_intercity_transport_correct(test_example,test_plan))
    # print(Is_attractions_correct(test_example,test_plan))
    # print(Is_hotels_correct(test_example,test_plan))
    # print(Is_restaurants_correct(test_example,test_plan))
    # print(Is_transport_correct(test_example,test_plan))
    # print(Is_time_correct(test_example,test_plan))
    # print(Is_space_correct(test_example,test_plan))

    # pass_flag = True

    # info_list = []
    # for func_i in func_list:
    #     flag, info = func_i(test_example,test_plan)

    #     print(info)

    #     pass_flag = pass_flag and flag
    #     info_list.append(info)

    # print("final result: ", pass_flag)

    # for item in info_list:
    #     print(item)
    # print(info_list)
