import json


# from NS_agent_inter.interactive_search_plus import symbolic_search,set_model
from NS_agent_inter.nl2sy import get_answer
from NS_agent_inter.symbolic_search import symbolic_search
from NS_agent_inter.interactive_search_class import Interactive_Search
from llms import deepseek, deepseek_json, deepseek_poi
import os
from numpy import int64, ndarray, integer, floating
from images import get_image_url
from copy import deepcopy
from tools.restaurants.apis import Restaurants
from tools.poi.apis import Poi

restaurants_tool = Restaurants()
poi_tool = Poi()
searcher = Interactive_Search()


class HardLogicError(Exception):
    pass


# set_model("deepseek")


def decode_json(json_obj):
    if isinstance(json_obj, dict):
        return {decode_json(k): decode_json(v) for k, v in json_obj.items()}
    elif isinstance(json_obj, list):
        return [decode_json(i) for i in json_obj]
    elif isinstance(json_obj, int64):
        return int(json_obj)
    elif isinstance(json_obj, ndarray):
        return decode_json(json_obj.tolist())
    elif isinstance(json_obj, integer):
        return int(json_obj)
    elif isinstance(json_obj, floating):
        return float(json_obj)
    else:
        return json_obj


def get_lon_lat(city: str, poi_name: str):
    poi_info = poi_tool.search_loc(city=city, name=poi_name)
    poi_info = [poi_info[1], poi_info[0]]
    return poi_info


def day_cost(day_plan: dict):
    total_cost = 0
    for activity in day_plan["activities"]:
        total_cost += activity["cost"]
        for trans in activity["transports"]:
            total_cost += trans["cost"]
    return int(total_cost)


def day_time(day_plan: dict):
    st_time = 60 * int(day_plan["activities"][0]["start_time"].split(":")[0]) + int(
        day_plan["activities"][0]["start_time"].split(":")[1]
    )
    ed_time = 60 * int(day_plan["activities"][-1]["end_time"].split(":")[0]) + int(
        day_plan["activities"][-1]["end_time"].split(":")[1]
    )
    total_time = ed_time - st_time
    total_time_str = f"{total_time//60}h {total_time%60}min"
    return total_time_str


def plan_to_poi(plan: dict):
    city = plan["target_city"]
    start_city = plan["start_city"]
    start_city_start_station = plan["itinerary"][0]["activities"][0]["start"]
    target_city_start_station = plan["itinerary"][0]["activities"][0]["end"]
    target_city_end_station = plan["itinerary"][-1]["activities"][-1]["start"]
    start_city_end_station = plan["itinerary"][-1]["activities"][-1]["end"]
    # start_city_start_station_lon_lat = get_lon_lat(city=start_city,poi_name=start_city_start_station)
    # start_city_end_station_lon_lat = get_lon_lat(city=start_city,poi_name=start_city_end_station)
    target_city_start_station_lon_lat = get_lon_lat(
        city=city, poi_name=target_city_start_station
    )
    target_city_end_station_lon_lat = get_lon_lat(
        city=city, poi_name=target_city_end_station
    )

    new_itinerary = []
    for day_plan in plan["itinerary"]:
        new_day_plan = {
            "day": day_plan["day"],
            "position": [],
            "cost": 0,
            "time": "0h 0min",
        }
        plan_day_poi = []
        poi_detail = []
        for activity in day_plan["activities"]:
            if "position" in activity:
                poi = activity["position"]
                plan_day_poi.append(poi)
                poi_detail.append(get_lon_lat(city=city, poi_name=poi))
        new_day_plan["position"] = plan_day_poi
        new_day_plan["position_detail"] = poi_detail
        new_day_plan["cost"] = day_cost(day_plan)
        new_day_plan["time"] = day_time(day_plan)
        new_itinerary.append(new_day_plan)
    new_itinerary[0]["position"].insert(0, target_city_start_station)
    new_itinerary[0]["position"].insert(0, start_city_start_station)
    new_itinerary[-1]["position"].append(target_city_end_station)
    new_itinerary[-1]["position"].append(start_city_end_station)
    new_itinerary[0]["position_detail"].insert(0, target_city_start_station_lon_lat)
    # new_itinerary[0]["position_detail"].insert(0, start_city_start_station_lon_lat)
    new_itinerary[-1]["position_detail"].append(target_city_end_station_lon_lat)
    # new_itinerary[-1]["position_detail"].append(start_city_end_station_lon_lat)
    new_plan = deepcopy(plan)
    new_plan.pop("itinerary")
    new_plan["start"] = start_city_start_station
    new_plan["end"] = start_city_end_station
    new_plan["itinerary"] = new_itinerary
    return new_plan


def get_trans_time_total(trans: list):
    total_time = 0
    for t in trans:
        st_time = 60 * int(t["start_time"].split(":")[0]) + int(
            t["start_time"].split(":")[1]
        )
        ed_time = 60 * int(t["end_time"].split(":")[0]) + int(
            t["end_time"].split(":")[1]
        )
        total_time += ed_time - st_time
    return total_time


def get_trans_dist_total(trans: list):
    total_dist = 0
    for t in trans:
        total_dist += t["distance"]
    # 保留两位小数
    return round(total_dist, 2)


def get_trans_cost_total(trans: list):
    total_cost = 0
    for t in trans:
        total_cost += t["cost"]
    return int(total_cost)


def get_trans_type_total(trans: list):
    # taxi = metro > walk
    for t in trans:
        if t["mode"] != "walk":
            return t["mode"]
    return "walk"


def process_after_search(plan: dict):
    p_plan = deepcopy(plan)
    result = {}
    # intercity_transport_start = plan["itinerary"][0]["activities"][0]
    # intercity_transport_end = plan["itinerary"][-1]["activities"][-1]
    intercity_transport_start = p_plan["itinerary"][0]["activities"][0]
    intercity_transport_start["cost"] = int(intercity_transport_start["cost"])
    intercity_transport_end = p_plan["itinerary"][-1]["activities"][-1]
    intercity_transport_end["cost"] = int(intercity_transport_end["cost"])

    city = p_plan["target_city"]
    activities = []
    # 弹出第一个activity
    p_plan["itinerary"][0]["activities"].pop(0)

    for day_plan in p_plan["itinerary"]:
        activities_per_day = []
        for activity in day_plan["activities"]:
            # print(activity)
            act = {}
            if len(activity["transports"]) != 0:
                act["trans_time"] = get_trans_time_total(activity["transports"])
                act["trans_distance"] = get_trans_dist_total(activity["transports"])
                act["trans_cost"] = get_trans_cost_total(activity["transports"])
                act["trans_type"] = get_trans_type_total(activity["transports"])
                act["trans_detail"] = activity["transports"]
                for i in range(len(act["trans_detail"])):
                    act["trans_detail"][i]["distance"] = round(
                        act["trans_detail"][i]["distance"], 2
                    )
            act["position"] = (
                activity["position"] if "position" in activity else activity["start"]
            )
            act["type"] = activity["type"]
            if act["type"] in ["breakfast", "lunch", "dinner"]:
                food_list = restaurants_tool.select(
                    city=city, key="name", func=lambda x: x == act["position"]
                )
                food_list = food_list["recommendedfood"].values.tolist()
                food_list = food_list[0] if len(food_list) > 0 else ""
                food_list = str(food_list).replace(" ", "")
                food_list = food_list.replace(",", "  ")
                act["food_list"] = food_list
            act["cost"] = activity["cost"]
            # 转整数
            act["cost"] = int(act["cost"])
            act["start_time"] = activity["start_time"]
            act["end_time"] = activity["end_time"]
            act["picture"] = get_image_url(
                city=city, poi_type=act["type"], name=act["position"]
            )
            activities_per_day.append(act)
        activities.append(activities_per_day)

    result["intercity_transport_start"] = intercity_transport_start
    result["intercity_transport_end"] = intercity_transport_end
    result["activities"] = activities

    return result


def generate_plan(request: dict, task_id, debug_mode=False):
    print(request)
    days = request["daysCount"]
    start_city = request["startCity"]
    target_city = request["destinationCity"]
    people_number = request["peopleCount"]
    addition_info = request["additionalRequirements"]

    nature_language = (
        f"当前位置{start_city},打算{people_number}个人一起去{target_city}旅游{days}天,"
    )
    if addition_info != "":
        nature_language += f"此外{addition_info},"
    nature_language += "请给我一个旅行规划."
    query = {}
    if debug_mode:
        print(nature_language)
    try:
        query = get_answer(nature_language=nature_language, model=deepseek_json())
    except Exception as e:
        print(e)
        return {"success": 0, "message": "fail to transform nl to symbol"}

    query["start_city"] = start_city
    query["target_city"] = target_city
    query["days"] = days
    query["people_number"] = people_number
    query["hard_logic"] = [
        logic_str
        for logic_str in query["hard_logic"]
        if not (logic_str.startswith("days") or logic_str.startswith("people_number"))
    ]
    query["hard_logic"].insert(0, f"days=={days}")
    query["hard_logic"].insert(0, f"people_number=={people_number}")
    query["nature_language"] = nature_language
    query_idx = task_id
    result_dir = f"query_results/{task_id}"

    flag, plan = searcher.symbolic_search(query=query, query_idx=query_idx)
    assert flag, "fail to generate plan"
    # _, plan = symbolic_search(query=query,query_idx=query_idx)
    plan = decode_json(plan)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=4, ensure_ascii=False)
    brief_plan = plan_to_poi(plan=plan)
    with open(os.path.join(result_dir, "brief_plan.json"), "w", encoding="utf-8") as f:
        json.dump(brief_plan, f, indent=4, ensure_ascii=False)
    plan = process_after_search(plan=plan)
    with open(os.path.join(result_dir, "plan_to_web.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=4, ensure_ascii=False)
    return {"success": 1, "plan": plan, "brief_plan": brief_plan}


def modify_plan(modify_str: str, task_id: int, debug_mode=False):
    """
    修改计划
    Args:
        modify_str: 修改内容
        task_id: 任务id
    Returns:
        dict: 修改后的计划
    """
    result_dir = f"query_results/{task_id}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, "request.json"), "r", encoding="utf-8") as f:
        request = json.load(f)
    request["additionalRequirements"] += f'，同时要求"{modify_str}"'
    print(f"从{result_dir}/request.json中读取请求, 修改后请求为: \n{request}")
    return generate_plan(request=request, task_id=task_id, debug_mode=debug_mode)


def plan_main(request: dict, task_id: int, debug_mode=False):
    """
    首次生成计划
    Args:
        request: 请求
        task_id: 任务id
    Returns:
        dict: 计划
    """
    result_dir = f"query_results/{task_id}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, "request.json"), "w", encoding="utf-8") as f:
        json.dump(request, f, indent=4, ensure_ascii=False)
    print(f"将请求保存到{result_dir}/request.json")
    return generate_plan(request=request, task_id=task_id, debug_mode=debug_mode)


if __name__ == "__main__":
    request_data = {
        "startCity": "深圳",
        "destinationCity": "上海",
        "peopleCount": 1,
        "daysCount": 2,
        "additionalRequirements": "",
    }
    print(plan_main(request=request_data, task_id=0, debug_mode=True))
    # print(modify_plan(modify_str="也要吃烧烤", task_id=0, debug_mode=True))
