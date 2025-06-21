from tools.attractions.apis import Attractions
from tools.hotels.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.transportation.apis import GoTo
from tools.notebook.apis import Notebook
from tools.planner.apis import Planner
from tools.poi.apis import Poi
from tools.intercity_transport.apis import IntercityTransport
import pandas as pd
from pandas import DataFrame
from typing import Callable


class Result:
    def __init__(self, data, idx, tool_call_id):
        self.data = data
        self.index = 0
        self.idx = idx
        self.tool_call_id = tool_call_id
        self.addition_text = ""

    def next_page(self):
        if not isinstance(self.data, DataFrame):
            return "next_page() is not supported for this data type:" + str(type(self.data)) + "\nMake sure you are using the correct index. -1 is the lastest result."
        self.index += 1
        return self

    def str__10data(self):
        res = ""
        if isinstance(self.data, DataFrame):
            if len(self.data) == 0:
                return "No data."
            if (len(self.data) - 1) // 10 < self.index:
                return "No more data."
            header_str = self.data.columns.values
            res += str(header_str) + "\n"
            for i in range(10):
                if self.index * 10 + i >= len(self.data):
                    break
                res += str(self.data.iloc[self.index * 10 + i].values) + "\n"
        else:
            res += str(self.data)
        # res = "Please note down what is useful using notedown method.\n" + res
        return res

    def __str__(self):
        res = f"Results[{self.idx}]:\n"
        if isinstance(self.data, DataFrame):
            if len(self.data) == 0:
                return "No data."
            if self.index * 10 >= len(self.data)+10:
                return "No more data."
            header_str = self.data.columns.values
            res += str(header_str) + "\n"
            for i in range(10):
                if self.index * 10 + i >= len(self.data):
                    break
                res += str(self.data.iloc[self.index * 10 + i].values) + "\n"
            res += "Page/Total: " + \
                str(self.index + 1) + "/" + str(len(self.data) //
                                                10 + (1 if len(self.data) % 10 != 0 else 0))
        else:
            res += str(self.data)
        # res = "Please note down what is useful using notedown method.\n" + res
        res = self.addition_text + '\n' + res
        return res


class Success:
    def __init__(self):
        pass


class DirectEnv:
    def __init__(self):
        global attractions, accommodations, restaurants, transportation, notebook, poi, intercitytransport
        attractions = Attractions()
        accommodations = Accommodations()
        restaurants = Restaurants()
        notebook = Notebook()
        intercitytransport = IntercityTransport()
        poi = Poi()

    def run(self, commands: str):
        result = eval(commands)
        result = Result(result, 0, 0)
        return result


class ReactEnv:
    def __init__(self, planner_llm, planner_prompt):
        global attractions, accommodations, restaurants, transportation, notebook, planner, poi, intercitytransport
        attractions = Attractions()
        accommodations = Accommodations()
        restaurants = Restaurants()
        notebook = Notebook()
        intercitytransport = IntercityTransport()
        planner = Planner(planner_llm, planner_prompt, notebook, Success())
        poi = Poi()

        global Results
        Results = [Result("Task started.", 0, 0)]

        self.error_num = 0
        self.finished = False
        self.note_num = 0
        self.next_page_num = 0
        self.ans = ""
        self.success_status = False

    def run(self, command: str):
        try:
            if "Action" in command:
                res_str = "Your action is refused because of starting with 'Action'."
                Results.append(Result(res_str, len(Results), 0))
                self.error_num += 1
                return Results[-1]
            command = command.replace("<STOP>", "")
            self.parse_exec(command)
            self.error_num = 0

        except Exception as e:
            res_msg = "调用失败，错误信息：\n"
            res_msg += str(e.with_traceback(None))
            res_msg += "请思考错误原因以及如何修改。"
            self.error_num += 1
            Results.append(Result(res_msg, len(Results), 0))

        if self.error_num > 3:
            self.finished = True
            Results[-1] = Result("连续错误次数过多，任务结束。", len(Results), "0")
        if command == "next_page()":
            self.next_page_num += 1
            if self.next_page_num >= 2:
                Results[-1].addition_text = "连续使用了太多次next_page，请确保你是在没有获得合适的数据的情况下使用next_page。"
                self.next_page_num = 1
        else:
            self.next_page_num = 0

        if str(Results[-1]) == "NoteBook updated.":
            self.note_num += 1
            if self.note_num >= 3:
                Results[-1] = Result("NoteBook updated." +
                                     "\n连续使用了太多次notebook，请你尽可能一次性写完所有有用的信息，如果你认为你的调用是合理的，请忽略这条提示，无论如何此次写入是有效的。", len(Results), Results[-1].tool_call_id)
                self.note_num = 1
        else:
            self.note_num = 0
        # 返回所有新增的结果
        return Results[-1]

    def parse_exec(self, command):
        call_result = eval(command)
        if isinstance(call_result, Success):
            self.finished = True
            self.success_status = True
            self.ans = planner.get_ans()
            Results.append(
                Result("Task finished. The answer is: " + self.ans, len(Results), 0))
        elif isinstance(call_result, Result):
            call_result.idx = len(Results)
            Results.append(call_result)
        else:
            Results.append(Result(call_result, len(Results), 0))

    def reset(self):
        self.finished = False
        self.success_status = False
        self.error_num = 0
        self.note_num = 0
        global Results
        Results = [Result("Task started.", 0, "0")]
        self.ans = ""
        planner.reset()
        notebook.reset()

    def is_finished(self):
        return self.finished

    def get_ans(self):
        return self.ans

    def is_success(self):
        return self.success_status


city_list = ["上海", "北京", "深圳", "广州", "重庆",
             "苏州", "成都", "杭州", "武汉", "南京"]


def attractions_keys(city: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return attractions.keys(city)


def accommodations_keys(city: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return accommodations.keys(city)


def restaurants_keys(city: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return restaurants.keys(city)


def attractions_select(city: str, key: str = "", func: Callable = lambda x: True):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    if key == "":
        key = "name"
    res = attractions.select(city, key, func)
    if len(res) == 0 and key == "type":
        return "Maybe you need use attractions_types(city) to learn the type."
    return res


def attractions_id_is_open(city: str, id: int, time: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return attractions.id_is_open(city, id, time)


def attractions_nearby(city: str, point: str, topk: int, dist: float = 2):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    try:
        lat, lon = poi.search(city, point)
        return attractions.nearby(city, lat, lon, topk, dist)
    except:
        return "No such point in the city. Check the point name."


def attractions_types(city: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return attractions.get_type_list(city)


def accommodations_select(city: str, key: str = "", func: Callable = lambda x: True):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    if key == "":
        key = "name"
    if key == "type":
        return "Maybe you need use accommodations_types(city) to learn the type."
    res = accommodations.select(city, key, func)
    return res


def accommodations_nearby(city: str, point: str, topk: int, dist: float = 5):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    try:
        lat, lon = poi.search(city, point)
        return accommodations.nearby(city, lat, lon, topk, dist)
    except:
        return "No such point in the city. Check the point name."


def restaurants_select(city: str, key: str = "", func: Callable = lambda x: True):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    if key == "":
        key = "name"
    if key == "type":
        return "Maybe you need use restaurants_cuisine(city) to learn the type."
    res = restaurants.select(city, key, func)
    return res


def restaurants_nearby(city: str, point: str, topk: int, dist: float = 2):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    try:
        lat, lon = poi.search(city, point)
        return restaurants.nearby(city, lat, lon, topk, dist)
    except:
        return "No such point in the city. Check the point name."


def restaurants_id_is_open(city: str, id: int, time: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return restaurants.id_is_open(city, id, time)


def restaurants_restaurants_with_recommended_food(city: str, food: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return restaurants.restaurants_with_recommended_food(city, food)


def restaurants_cuisine(city: str):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    return restaurants.get_cuisine_list(city)


def goto(city: str, start: str, end: str, start_time: str, method: str, verbose=False):
    if city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    # position_1 = Poi().search(city, start)
    # position_2 = Poi().search(city, end)
    # print(position_1, position_2)
    # return GoTo(city, position_1, position_2, start_time, method)
    if method not in ["metro", "walk", "taxi"]:
        return "Invalid method. Only support ['metro', 'walk', 'taxi']"
    res = GoTo(city, start, end, start_time, method, verbose)
    if res == "Location must be a tuple of (lat, lon)":
        res = "Error location name."
    return res


def notedown(description: str, content: str):
    return notebook.write(description, content)


def next_page(idx: int):
    if idx >= len(Results):
        return "Invalid index."
    return Results[idx].next_page()


def plan(query: str):
    return planner(query)


def next_page(idx: int = -1):
    if idx >= len(Results):
        return "Invalid index."
    return Results[idx].next_page()


def intercity_transport_select(start_city: str, end_city: str, intercity_type: str, earliest_leave_time: str = None):
    if start_city not in city_list or end_city not in city_list:
        return "Only support cities in " + str(city_list) + "." + "必须使用中文城市名。"
    if earliest_leave_time is None:
        return intercitytransport.select(start_city, end_city, intercity_type)
    else:
        tmp = intercitytransport.select(
            start_city, end_city, intercity_type)
        bool_list = [False for i in range(len(tmp.values))]

        def time_to_float(x): return float(
            x.split(":")[0]) + float(x.split(":")[1])/60
        earliest_leave_time = time_to_float(earliest_leave_time)
        for i in range(len(tmp)):
            # ["BeginTime"]
            if time_to_float(tmp["BeginTime"][i]) >= earliest_leave_time:
                bool_list[i] = True
        return tmp[bool_list]


if __name__ == "__main__":
    a = DirectEnv()
    # 获取所有城市的attractioins, accommodations, restaurants的type, featurehoteltype, cuisine的取值集合
    city_list = ["上海", "北京", "深圳", "广州", "重庆",
                    "苏州", "成都", "杭州", "武汉", "南京"]
    type_list = []
    hotel_feature_list = []
    cuisine_list = []
    for city in city_list:
        tmp_type = attractions.data[city]["type"].unique()
        type_list.extend(tmp_type)
        tmp_feature = accommodations.data[city]["featurehoteltype"].unique()
        hotel_feature_list.extend(tmp_feature)
        tmp_cuisine = restaurants.data[city]["cuisine"].unique()
        cuisine_list.extend(tmp_cuisine)
    type_list = list(set(type_list))
    hotel_feature_list = list(set(hotel_feature_list))
    cuisine_list = list(set(cuisine_list))
    print(type_list)
    print(hotel_feature_list)
    print(cuisine_list)
