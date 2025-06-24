

import argparse
from llms import deepseek, deepseek_json, deepseek_poi, model_GPT_poi
import random
from NS_agent_inter.retrieval import Retriever
import numpy as np
import pandas as pd
import llms
import datetime
import time
from evaluation.hard_constraint import evaluate_constraints as evaluate_logical_constraints
from evaluation.hard_constraint import get_symbolic_concepts, calc_cost_from_itinerary_wo_intercity
from evaluation.commonsense_constraint import func_commonsense_constraints
from evaluation.utils import load_json_file
import os
import json
from envs import goto
from tools.intercity_transport.apis import IntercityTransport
from tools.attractions.apis import Attractions
from tools.restaurants.apis import Restaurants
from tools.hotels.apis import Accommodations
import sys
sys.path.append("../")


random.seed(0)
ret = Retriever()
llm_time_cost = 0


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def __del__(self):
        self.log.close()


TIME_CUT = 60 * 10
time_before_search = 0

accommodation = Accommodations()
restaurants = Restaurants()
attractions = Attractions()
intercity_transport = IntercityTransport()


interactive_mode = "human"
llm_model = None


def set_model(model_name):
    model_name = model_name + "_poi"
    global llm_model, interactive_mode
    llm_model = getattr(llms, model_name)()
    interactive_mode = model_name


def get_avialable_plan():
    global avialable_plan

    return avialable_plan


def constraints_validation(query, plan, poi_plan):

    global avialable_plan
    res_plan = {"people_number": query["people_number"],
                "start_city": query["start_city"],
                "target_city": query["target_city"],
                "itinerary": plan,
                }
    print("validate the plan [for query {}]: ".format(poi_plan["query_idx"]))
    print(res_plan)

    bool_result = func_commonsense_constraints(query, res_plan)

    if bool_result:
        avialable_plan = res_plan

    try:
        extracted_vars = get_symbolic_concepts(query, res_plan)

    except:
        extracted_vars = None

    print(extracted_vars)

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

    # if logical_result:
    #     print("Logical passed!")

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


def time_compare_if_earlier_equal(time_1, time_2):

    time1 = float(time_1.split(":")[0])*60 + float(time_1.split(":")[1])
    time2 = float(time_2.split(":")[0])*60 + float(time_2.split(":")[1])

    return time1 <= time2


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


def react_prompt(info_list, action_prompt, return_history_message=False, history_message=[], verbose=False):

    global llm_time_cost
    global llm_step
    global llm_model
    # Think
    st_time = time.time()
    llm_step += 1

    scratchpad = ""
    for info_i in info_list:
        scratchpad = scratchpad + info_i + "\n"

    scratchpad += f'Thought[{llm_step}]:  '

    json_scratchpad = history_message
    json_scratchpad.append({"role": "user", "content": scratchpad})

    # thought = self.llm(self.prompt+self.scratchpad)
    if verbose:
        print("LLM query: ", json_scratchpad)
    thought = llm_model(json_scratchpad)
    scratchpad = scratchpad + ' ' + thought

    json_scratchpad.append(
        {"role": "assistant", "content": thought})
    # if self.need_print:
    if verbose:
        print(f"Thought[{llm_step}]:", thought)
    # Act
    scratchpad = action_prompt + f'\nAction[{llm_step}]: '

    json_scratchpad.append(
        {"role": "user", "content": scratchpad})

    if verbose:
        print("LLM query: ", json_scratchpad)
    # action = self.llm(self.prompt+self.scratchpad)
    action = llm_model(json_scratchpad)

    json_scratchpad.append(
        {"role": "assistant", "content": action})
    # print(action)

    scratchpad += ' ' + str(action)
    # # if self.need_print:
    if verbose:
        print(f"Action[{llm_step}]:", str(action))

    if return_history_message:
        return action, json_scratchpad
    llm_time_cost += time.time() - st_time
    return action

    # self.json_scratchpad.append(
    #     {"role": "assistant", "content": str(action)})
    # self.json_log.append({f'Action[{self.curr_step}]': str(action)})


def reason_prompt(info_list, return_history_message=True, history_message=[], verbose=False):

    global llm_time_cost
    global llm_step
    global llm_model
    st_time = time.time()
    # print(llm_model)
    # Think
    llm_step += 1

    scratchpad = ""

    for info_i in info_list:
        scratchpad = scratchpad + info_i + "\n"

    scratchpad += f'请简洁的回复。Thought[{llm_step}]:  '

    json_scratchpad = history_message
    json_scratchpad.append({"role": "user", "content": scratchpad})

    # thought = self.llm(self.prompt+self.scratchpad)
    if verbose:
        print("LLM query: ", json_scratchpad)
    thought = llm_model(json_scratchpad)
    scratchpad = scratchpad + ' ' + thought

    json_scratchpad.append(
        {"role": "assistant", "content": thought})
    # if self.need_print:

    if verbose:
        print(f"Thought[{llm_step}]:", thought)

    if return_history_message:
        return thought, json_scratchpad
    llm_time_cost += time.time() - st_time
    return thought

    # self.json_scratchpad.append(
    #     {"role": "assistant", "content": str(action)})
    # self.json_log.append({f'Action[{self.curr_step}]': str(action)})


def act_prompt(info_list, action_prompt, return_history_message=False, history_message=[], verbose=False):

    global llm_step
    global llm_time_cost
    # Think
    st_time = time.time()
    llm_step += 1

    scratchpad = ""
    for info_i in info_list:
        scratchpad = scratchpad + info_i + "\n"

    scratchpad += "\n" + action_prompt + "\n" + f'Action[{llm_step}]: '

    json_scratchpad = history_message
    json_scratchpad.append({"role": "user", "content": scratchpad})

    # thought = self.llm(self.prompt+self.scratchpad)
    if verbose:
        print("LLM query: ", json_scratchpad)

    json_model = deepseek_json()
    action = json_model(json_scratchpad)
    json_scratchpad.append(
        {"role": "assistant", "content": action})

    if verbose:
        print(f"Action[{llm_step}]:", str(action))

    if return_history_message:
        return action, json_scratchpad
    llm_time_cost += time.time() - st_time
    return action


def get_poi_type_from_time_sym(current_time, candidates_type, back_transport_time):
    if 'back-intercity-transport' in candidates_type:
        if time_compare_if_earlier_equal(back_transport_time, add_time_delta(current_time, 180)):
            return 'back-intercity-transport'

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


def get_poi_type_from_time(current_time, candidates_type, back_transport_time):

    # print("当前时间是 {}，我们现在需要项选择下一个活动的类型，包括：".format(current_time))

    # print("lunch: 在邻近中午的时间找家餐厅吃饭，注意午餐时间要求在[11:00 -- 13:00]，请预留时间用于交通到达相应地点")
    # print("dinner: 在邻近傍晚的时间找家餐厅吃饭，注意午餐时间要求在[17:00 -- 20:00]，请预留时间用于交通到达相应地点")
    # print("hotel: 在晚上结束一天的游玩时我们要回到酒店，注意我们要在 23:00 前到达酒店，请预留时间用于交通到达相应地点")
    # print("attraction: 在一天中游览不同的景点，注意景点的游览和中途的交通要花费一定时间，请预留时间以便在合适时间用餐、在夜间回到酒店")
    # print("back-intercity-transport 在旅程的最后一天，可以选择回家")

    # print("根据现有规划，当前可选择的候选类型包括: {}".format(candidates_type))

    info_list = [
        "当前时间是{}，我们现在需要项选择下一个活动的类型，包括：".format(current_time),
    ]
    if "lunch" in candidates_type:
        info_list.append(
            "lunch: 在邻近中午的时间找家餐厅吃饭，注意午餐时间要求在[11:00 -- 13:00]，请预留时间用于交通到达相应地点")
    if "dinner" in candidates_type:
        info_list.append(
            "dinner: 在邻近傍晚的时间找家餐厅吃饭，注意午餐时间要求在[17:00 -- 20:00]，请预留时间用于交通到达相应地点")
    if "attraction" in candidates_type:
        info_list.append(
            "在一天中游览不同的景点，注意景点的游览和中途的交通要花费一定时间，请预留时间以便在合适时间用餐、在夜间回到酒店")

    if "back-intercity-transport" in candidates_type:
        info_list.append(
            "back-intercity-transport: 今天是旅程的最后一天，可以选择回家,返程车票时间是{}，请预留时间用于交通到达相应地点".format(back_transport_time))
    if "hotel" in candidates_type:
        info_list.append(
            "hotel: 在晚上结束今天的游玩时我们要回到酒店，注意我们要在 23:00 前到达酒店，请预留时间用于交通到达相应地点")

    info_list.append(
        "当前可选择的候选类型包括: {}, 作为一个旅行规划助手，请结合以上信息进行思考".format(candidates_type))

    if interactive_mode == "human":

        for info_i in info_list:
            print(info_i)

        poi_type = input()
        if poi_type in candidates_type:
            pass
        else:
            while 3 < 33:

                print("请在候选类型中做出选择，保证输入类型是当前可选择的候选类型包括: {}".format(candidates_type))
                poi_type = input()

                if poi_type in candidates_type:
                    break

    else:

        action_prompt = "请在候选类型中选择下一个活动的类型，请确保输出的为类型的字符串，如选择XX类型则返回##XX##，并确保你的选择XX在候选类型{}中".format(
            candidates_type)

        result = react_prompt(info_list, action_prompt,
                              return_history_message=False, history_message=[])

        result_pos = result.split("##")[1]

        # print(result_pos)

        for cand_i in candidates_type:
            if cand_i in result_pos:
                poi_type = cand_i

    return poi_type

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


def score_poi(planning_info, poi_info, react=False, history_message=[]):

    if interactive_mode == "human":

        print(planning_info)

        print(poi_info)

        print(
            "请依据上述信息，给每个目标地点打分，分数范围 [0,1] 的实数, 打分越高表示越希望去该目标，请注意你的输出必须有区分度，必须给出不同的分数")

        score_list = []

        for i in range(len(poi_info)):
            score_i = input()

            while 3 < 33:
                try:
                    score_i = float(score_i)
                    break
                except:
                    print("请输入一个实数")
                    score_i = input()
                    score_i = float(score_i)

            score_list.append(score_i)
    else:

        # info_list = planning_info

        # info_list.append("请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的列表，列表长度与给出的信息条数一致一般是10条")

        info_list = []

        # poi_info = poi_info.drop(["latitude", "longitude"])
        for item in poi_info:
            info_list.append(item)

        for p_i in planning_info:
            info_list.append(p_i)

        # info_list.append("请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的列表，列表长度与给出的用**id**标记出的候选信息条数一致")

        # action_prompt = "请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的#列表#，列表长度与给出的用**id**标记出的候选信息条数一致， 比如有10条时得到这样的形式: ##[8.0, 2.1, 3.0, 5.0, 7.0, 8.2, 6.5, 8.7, 9.7, 8.4]##, 请确保你输出的是一行一个实数的列表, 不要输出多余信息"
        action_prompt = "请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请注意你的输出必须有区分度，必须给出不同的分数，请确保你输出的是一个实数的#列表#，列表长度与给出的用**id**标记出的候选信息条数一致， 比如有10条候选时得到这样的形式: [8.0, 2.1, 3.0, 5.0, 7.0, 8.2, 6.5, 8.7, 9.7, 8.4],现在请你给前面这{}个候选打分，请确保你输出的是一行一个实数的列表, 不要输出多余信息".format(
            len(poi_info))
        # result = react_prompt(info_list, action_prompt)

        if react:
            result, history_message = react_prompt(
                info_list, action_prompt, return_history_message=True, history_message=[])
        else:
            result, history_message = act_prompt(
                info_list, action_prompt, return_history_message=True, history_message=[])
        print("the result is: ", result)

        # result = result.split("##")[1]

        # str_res = result.strip("[").strip("]")
        str_res = result.split["["][1].spilt["]"][0]
        score_list = []
        for item in str_res.split(", "):

            score_list.append(float(item))

        # print("score_list: ", score_list)

    return score_list, history_message


def rewrite_plan(query, verbose=False):
    global llm_time_cost
    global llm_step
    global llm_model
    st_time = time.time()
    # Think
    llm_step += 1

    scratchpad = query + "\n请将上述内容中涉及到的名字（景点、餐厅）提取出来，并用逗号隔开。"

    scratchpad += f'Thought[{llm_step}]:  '

    json_scratchpad = []
    json_scratchpad.append({"role": "user", "content": scratchpad})

    # thought = self.llm(self.prompt+self.scratchpad)
    if verbose:
        print("LLM query: ", json_scratchpad)
    thought = llm_model(json_scratchpad)

    if verbose:
        print(f"Thought[{llm_step}]:", thought)
    llm_time_cost += time.time() - st_time
    return thought


def score_poi_think_overall_act_page(planning_info, poi_info, react=False, history_message=[]):

    # info_list = planning_info

    # info_list.append("请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的列表，列表长度与给出的信息条数一致一般是10条")

    info_list = []

    # poi_info = poi_info.drop(["latitude", "longitude"])

    new_poi_info = poi_info[0:30]
    for item in new_poi_info:
        info_list.append(item)

    for p_i in planning_info:
        info_list.append(p_i)

    overall_plan, history_message_think_overall = reason_prompt(
        info_list, return_history_message=True, history_message=[])
    # print(history_message_think_overall)

    # Suggestions of LLM
    # LLM_suggestion_text=history_message_think_overall[-1]['content']
    # print(LLM_suggestion_text)

    # print(poi_name_list)
    # select_index=ret.retrieval_index(poi_name_list,LLM_suggestion_text)
    # new_poi_info = [poi_info[index] for index in select_index]
    # poi_info=new_poi_info
    # print(poi_info)
    # exit(0)
    # history_message_think_overall = history_message_think_overall[1:]

    current_page = 1
    item_per_page = 20
    score_list = []

    # rewrite the plan
    rewrite_ans = rewrite_plan(overall_plan)
    # print(rewrite_ans)
    # exit(0)

    querys = rewrite_ans.split(', ')
    scores = []
    for q in querys:
        score = ret.get_score(poi_info, q)
        scores.append(score)

    scores = np.array(scores)
    score_list = np.max(scores, axis=0)

    # indexed_scores = [(i, score) for i, score in enumerate(score_list)]
    # sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    # print([poi_info[int(i)] for i,_ in sorted_scores[:50]])
    # exit(0)
    # indexed_scores = [(i, score) for i, score in enumerate(score_list)]
    # sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    # indexes=[Poi[int(i)] for i,_ in sorted_scores[:50]])

    # while current_page * item_per_page - item_per_page < len(poi_info):

    #     page_l = current_page * item_per_page - item_per_page
    #     page_r = min(current_page * item_per_page, len(poi_info))
    #     print("current page: {}, [{}, {}]".format(current_page, page_l, page_r))

    #     info_list_i = []

    #     for item in poi_info[page_l: page_r]:
    #         # print(item)
    #         info_list_i.append(item)

    #     print(len(info_list_i), "items")
    #     print(len(history_message_think_overall), "messages...")
    #     for p_i in planning_info:
    #         info_list_i.append(p_i)

    #     input_history = deepcopy(history_message_think_overall)

    #     action_prompt = "请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请注意你的输出必须有区分度，必须给出不同的分数，请确保你输出的是一个json格式的dict，比如有10条候选时得到这样的形式: {'score':[8.0, 2.1, 3.0, 5.0, 7.0, 8.2, 6.5, 8.7, 9.7, 8.4]},现在请你给前面这"+str(page_r - page_l)+"个候选打分，请确保你输出的是一行一个实数的列表, 不要输出多余信息"

    #     result = act_prompt(info_list_i, action_prompt, return_history_message=False, history_message=input_history, verbose=False)

    #     result_dict=eval(result)
    #     print("the result is: ", result_dict)
    #     # result = result.split("##")[1]
    #     # str_res = result.strip("[").strip("]")
    #     # str_res = result.split["["][1].spilt["]"][0]
    #     # score_list_i = []
    #     # for item in str_res.split(", "):
    #     #     score_list_i.append(float(item))

    #     # for score_i in score_list_i:
    #     #     score_list.append(score_i)
    #     score_list+=result_dict['score']

    #     current_page += 1

    print("score over")

    # info_list.append("请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的列表，列表长度与给出的用**id**标记出的候选信息条数一致")

    # action_prompt = "请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的#列表#，列表长度与给出的用**id**标记出的候选信息条数一致， 比如有10条时得到这样的形式: ##[8.0, 2.1, 3.0, 5.0, 7.0, 8.2, 6.5, 8.7, 9.7, 8.4]##, 请确保你输出的是一行一个实数的列表, 不要输出多余信息"

    # result = react_prompt(info_list, action_prompt)

    # print("score_list: ", score_list)

    return score_list


def recommend_poi_time(planning_info, poi_name):

    info_list = planning_info

    info_list.append(
        "请你根据以上信息，思考我在{}的游玩时间长度,请注意你只需要思考我在这个景点的游玩即可，不用给出我完整的行程".format(poi_name))

    action_prompt = "请依据上述信息，给出我应该在{}游玩的时间，以XXtimeXXminutes的形式输出，比如要游玩40分钟则输出##40##minutes, 游玩90分钟则输出##90##minutes".format(
        poi_name)

    result = react_prompt(info_list, action_prompt, history_message=[])

    recommend_time = int(result.split("##")[1])

    return recommend_time


def llm_next_page(hisorty_message, previous_score, current_page):

    info_list = ["你已经查了{}个信息，对它们的打分是:{}, 如果以上候选都不满足要求，那么你需要进一步查询更多信息,请思考是否满足了对旅行规规划的要求".format(
        len(previous_score), previous_score)]

    action_prompt = "如果你需要更多候选，请输出#Y#，否则输出#N#，请在满足要求的前提下尽可能不再收集数据"

    if current_page > 5:

        action_prompt = action_prompt + "你已经频繁调用信息，请谨慎选择，减少信息收集"

    result, history_message = react_prompt(
        info_list, action_prompt, return_history_message=True, history_message=hisorty_message)

    if "Y" in result:
        return True

    return False


def select_feature(planning_info):
    if interactive_mode == "human":

        print(planning_info)

        sel_feature = input()
    else:

        # info_list = planning_info

        # info_list.append("请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的列表，列表长度与给出的信息条数一致一般是10条")

        info_list = []

        # poi_info = poi_info.drop(["latitude", "longitude"])

        for p_i in planning_info:
            info_list.append(p_i)

        # info_list.append("请依据上述信息，给每个目标地点打分，分数范围 [0,10] 的实数, 打分越高表示越希望去该目标， 请确保你输出的是一个实数的列表，列表长度与给出的信息条数一致一般是10条")

        action_prompt = "如果你认为描述:XX是最合适的，请按##XX##的格式输出，不要输出多余信息"

        result = react_prompt(info_list, action_prompt)

        print(result)

        sel_feature = result.split("##")[1]

        print("selected_feature: ", sel_feature)

    return sel_feature


def search_poi(query, poi_plan, plan, current_time, current_position, current_day=0, verbose=False):

    global time_before_search

    global fast_mode

    if time.time() > time_before_search + TIME_CUT:

        print("Searching TIME OUT !!!")

        raise Exception("TIME_OUT")

        return False, plan

        return False, {"info": "TIME OUT"}

    if "cost_wo_intercity" in query:
        inner_city_cost = calc_cost_from_itinerary_wo_intercity(
            plan, query["people_number"])

        if inner_city_cost >= query["cost_wo_intercity"]:

            print("budget run out: inner-city budget {}, cost {}".format(
                query["cost_wo_intercity"], inner_city_cost))
            # print(plan)

            # exit(0)
            return False, plan

    global poi_info
    global restaurants_visiting
    global attractions_visiting
    global food_type_visiting
    global spot_type_visiting
    global attraction_names_visiting
    global restaurant_names_visiting

    # intercity_transport - back

    if current_time != "" and time_compare_if_earlier_equal("23:00", current_time):
        print("too late, after 23:00")
        return False, plan

    # if current_time == "24:00":
    #     print(current_time)
    #     print(plan)

    if current_time != "" and current_day == query["days"] - 1:
        # We should go back in time ...
        transports_sel = goto(city=query["target_city"],
                              start=current_position, end=poi_plan["back_transport"]["From"],
                              start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
        arrived_time = transports_sel[-1]["end_time"]

        if time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], arrived_time):

            print("Can not go back source-city in time, current POI {}, station arrived time: {}".format(
                current_position, arrived_time))
            return False, plan

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

    # intercity_transport - go
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

        new_time = poi_plan["go_transport"]["EndTime"]
        new_position = poi_plan["go_transport"]["To"]
        success, plan = search_poi(
            query, poi_plan, plan, new_time, new_position, current_day, verbose)
        if success:
            return True, plan

        else:

            print("No solution for the given Go Transport")
            return False, plan

    # breakfast
    if current_time == "00:00":

        if len(plan) < current_day + 1:
            plan.append({"day": current_day + 1, "activities": []})

        # breakat at hotel
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
        else:
            if time.time() > time_before_search + TIME_CUT:
                print("Searching TIME OUT !!!")
                raise Exception("TIME_OUT")
                return False, plan
        plan[current_day]["activities"].pop()

    haved_lunch_today, haved_dinner_today = False, False

    for act_i in plan[current_day]["activities"]:
        if act_i["type"] == "lunch":
            haved_lunch_today = True
        if act_i["type"] == "dinner":
            haved_dinner_today = True

    # print("info before search poi type", current_day, current_time, current_position, poi_plan, plan)

    candidates_type = ["attraction"]
    # and time_compare_if_earlier_equal(current_time, "12:30"):
    if (not haved_lunch_today):
        candidates_type.append("lunch")
    # and time_compare_if_earlier_equal(current_time, "18:30"):
    if (not haved_dinner_today):
        candidates_type.append("dinner")
    if ("accommodation" in poi_plan) and (current_day < query["days"]-1):
        candidates_type.append("hotel")

    # and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], add_time_delta(current_time, 180)):
    if current_day == query["days"] - 1 and current_time != "":
        candidates_type.append("back-intercity-transport")

    # print(current_day, query["days"] - 1)
    # print("candidates_type: ", candidates_type)

    while len(candidates_type) > 0:

        # print(plan)

        # print(candidates_type)

        poi_type = get_poi_type_from_time_sym(
            current_time, candidates_type, poi_plan["back_transport"]["BeginTime"])

        if verbose:

            print("POI planning, day {} {}, {}, next-poi type: {}".format(current_day,
                  current_time, current_position, poi_type))

        if poi_type == "back-intercity-transport":
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

                if time.time() > time_before_search + TIME_CUT:
                    print("Searching TIME OUT !!!")
                    raise Exception("TIME_OUT")
                    return False, plan

                plan[current_day]["activities"].pop()

                print(
                    "[We choose to go back transport and finish this trip], but constraints_validation failed...")

        elif poi_type in ["lunch", "dinner"]:

            info_list = [query["nature_language"]]

            info_list.append("在这次在{}的旅行中我们之前去过了{}，不要重复去".format(
                query["target_city"], str(restaurant_names_visiting)))
            info_list.append("现在是第{}天 {}, 请帮我选择一个餐厅去吃{}".format(
                current_day + 1, current_time, poi_type))

            rest_info = poi_info["restaurants"]

            if "cost_wo_intercity" in query:

                rest_cost = query["cost_wo_intercity"] - inner_city_cost
                info_list.append(
                    "本次旅行有整体开销的约束，剩余用于餐饮和景点的预算为: {} 元".format(rest_cost))

                upper_budget = rest_cost / query["people_number"]
                rest_info = rest_info[rest_info["price"] <= upper_budget]

            if not fast_mode:

                if "food_type" in query and (not set(query["food_type"]) <= set(food_type_visiting)):

                    req_food_type = set(food_type_visiting) - \
                        set(query["food_type"])

                    info_list.append(
                        "本次旅行有对餐饮类型的需求，这些类型至少需要吃一次 {}, 在选择餐厅的时候请根据餐厅的餐饮类型加以考虑".format(req_food_type))

                # print(res_weight)
                if "restaurant_names" in query and (not set(query["restaurant_names"]) <= set(restaurant_names_visiting)):

                    req_res_type = set(restaurant_names_visiting) - \
                        set(query["restaurant_names"])

                    info_list.append(
                        "本次旅行有对特定餐饮商家的需求，这几家至少需要去一次 {}, 在选择餐厅的时候请根据餐厅的名字加以考虑".format(req_res_type))

                info_list.append("请根据以上信息考虑什么样的餐厅满足我们的需求")

                score_list = []

                current_page = 1
                item_per_page = 20
                history_message = []
                score_list = []

                while current_page * item_per_page - item_per_page < len(rest_info):

                    history_message = []

                    page_l = current_page * item_per_page - item_per_page
                    page_r = min(current_page * item_per_page, len(rest_info))

                    print("current page: {}, [{}, {}]".format(
                        current_page, page_l, page_r))
                    current_info = rest_info.iloc[page_l: page_r, :]

                    poi_info_list = []
                    for idx in range(len(current_info)):

                        res_i = rest_info.iloc[idx]

                        try_transport = goto(city=query["target_city"], start=current_position,
                                             end=res_i["name"], start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
                        arrived_time = try_transport[-1]["end_time"]

                        # poi_info_list.append("**{}** name: {} , arrived_time: {}, cusine: {}, price_per_preson: {}, recommended food: {}".format(idx, res_i["name"], arrived_time, res_i["cuisine"], res_i["price"], res_i["recommendedfood"]))
                        poi_info_list.append("**{}** name: {} , arrived_time: {}, cusine: {}, price_per_preson: {}".format(
                            idx, res_i["name"], arrived_time, res_i["cuisine"], res_i["price"]))

                    score_list_i, history_message = score_poi(
                        info_list, poi_info_list, history_message)

                    print("score over")
                    # score_list_i = score_poi(info_list, current_info)

                    for iitem in score_list_i:
                        score_list.append(iitem)

                    print("next page start")

                    history_message = []
                    if_next_page = llm_next_page(
                        history_message, score_list, current_page)
                    if not if_next_page:
                        break

                    else:
                        current_page += 1

                score_list = np.array(score_list)
            else:
                score_list = rest_info["importance"].values

            ranking_idx = np.argsort(-score_list)

            for r_i in ranking_idx:

                res_idx = r_i
                # print(ranking_idx, r_i)

                # print("visiting: ", restaurants_visiting, res_idx)

                if not (res_idx in restaurants_visiting):

                    # print("in-loop: ", r_i, weight_list[r_i], poi_info["restaurants"].iloc[res_idx]["name"], poi_info["restaurants"].iloc[res_idx]["cuisine"])

                    # exit(0)

                    poi_sel = rest_info.iloc[res_idx]

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

                    if time_compare_if_earlier_equal(act_end_time, act_start_time):
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

                    restaurant_names_visiting.append(poi_sel["name"])
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
                    else:
                        if time.time() > time_before_search + TIME_CUT:
                            print("Searching TIME OUT !!!")
                            raise Exception("TIME_OUT")
                            return False, plan

                    plan[current_day]["activities"].pop()
                    restaurants_visiting.pop()
                    food_type_visiting.pop()
                    restaurant_names_visiting.pop()

                    # print("res {} fail...".format(poi_sel["name"]))

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
            else:
                if time.time() > time_before_search + TIME_CUT:
                    print("Searching TIME OUT !!!")
                    raise Exception("TIME_OUT")
                    return False, plan
            plan[current_day]["activities"].pop()

        elif poi_type == "attraction":

            if not fast_mode:

                info_list = [query["nature_language"]]

                info_list.append("在这次在{}的旅行中我们之前去过了{}，不要重复去".format(
                    query["target_city"], str(attraction_names_visiting)))
                info_list.append("现在是第{}天 {}, 请帮我选择一个景点去游玩".format(
                    current_day + 1, current_time))

                info_list = ["请帮我选择一个景点去游玩"]

                if ("spot_type" in query and (not set(query["spot_type"]) <= set(spot_type_visiting))) or ("spot_type_unseen" in query):

                    req_spot_type = set()

                    if "spot_type" in query:
                        req_spot_type = set.union(req_spot_type, set(
                            query["spot_type"]) - set(spot_type_visiting))

                    if "spot_type_unseen" in query:
                        req_spot_type = set.union(
                            req_spot_type, set(query["spot_type_unseen"]))

                    info_list.append(
                        "本次旅行有对景点类型的需求，这些类型至少需要去一次 {}, 在选择景点的时候请根据景点的景点类型加以考虑".format(req_spot_type))

                if ("attraction_names" in query and (not set(query["attraction_names"]) <= set(attraction_names_visiting))) or ("attraction_names_unseen" in query):

                    req_attr_name = set()
                    if "attraction_names" in query:
                        req_attr_name = set.union(req_attr_name, set(
                            query["attraction_names"]) - set(attraction_names_visiting))

                    if "attraction_names_unseen" in query:
                        req_attr_name = set.union(req_attr_name, set(
                            query["attraction_names_unseen"]))

                    info_list.append(
                        "本次旅行有对特定景点的需求，这几个地方至少需要去一次 {}, 在选择景点的时候请根据景点的名字加以考虑".format(req_attr_name))

                    # print(query["attraction_names"])

                    # for res_i, attr_name in enumerate(attr_info["name"]):

                    #     attr_weight[res_i] *= max(int(attr_name in query["attraction_names"] and (not attr_name in attraction_names_visiting)) , 1e-5)

                info_list.append("请根据以上信息考虑什么样的景点满足我们的需求")
                # info_list.append("请根据以上信息为以下候选目标打分, 打分越高表示越希望去该目标")

                score_list = []

                current_page = 1
                item_per_page = 20
                history_message = []

                while current_page * item_per_page - item_per_page < len(poi_info["attractions"]):

                    history_message = []

                    page_l = current_page * item_per_page - item_per_page
                    page_r = min(current_page * item_per_page,
                                 len(poi_info["attractions"]))

                    current_info = poi_info["attractions"].iloc[page_l: page_r, :]

                    poi_info_list = []
                    for idx in range(len(current_info)):

                        res_i = poi_info["attractions"].iloc[idx]

                        try_transport = goto(city=query["target_city"], start=current_position,
                                             end=res_i["name"], start_time=current_time, method=poi_plan["transport_preference"], verbose=False)
                        arrived_time = try_transport[-1]["end_time"]

                        # poi_info_list.append("**{}** name: {} , arrived_time: {}, cusine: {}, price_per_preson: {}, recommended food: {}".format(idx, res_i["name"], arrived_time, res_i["cuisine"], res_i["price"], res_i["recommendedfood"]))
                        poi_info_list.append("**{}** name: {} , arrived_time: {}, price_per_preson: {}".format(
                            idx, res_i["name"], arrived_time, res_i["price"]))

                    score_list_i, history_message = score_poi(
                        info_list, poi_info_list, history_message)

                    # score_list_i = score_poi(info_list, current_info)

                    for iitem in score_list_i:
                        score_list.append(iitem)

                    if_next_page = llm_next_page(
                        history_message, score_list, current_page)
                    if not if_next_page:
                        break

                    else:
                        current_page += 1

                # while current_page * 10 < len(poi_info["attractions"]):

                #     current_info = poi_info["attractions"].iloc[current_page * 10 - 10: current_page * 10, :]

                #     # print(current_info)

                #     score_list_i = score_poi(info_list, current_info)

                #     for iitem in score_list_i:
                #         score_list.append(iitem)

                #     if interactive_mode == "human":
                #         print("如果这些景点信息仍不足满足上述需求，可以调用 next page 获取更多信息, 请输入 Y 或者 N，表示是否需要翻页")

                #         if_next = input()

                #         if if_next == "Y":

                #             current_page += 1
                #             continue
                #         else:
                #             break
                #     else:
                #         break

                print("score list: ", score_list)

                score_list = np.array(score_list)
            else:
                score_list = poi_info["attractions"]["importance"].values
            ranking_idx = np.argsort(-score_list)

            for r_i in ranking_idx:

                attr_idx = r_i
                # print(ranking_idx, r_i)

                # print("visiting: ", restaurants_visiting, res_idx)

                if not (attr_idx in attractions_visiting):

                    poi_sel = poi_info["attractions"].iloc[attr_idx]

                    # print(current_position, poi_sel["name"])

                    transports_sel = goto(city=query["target_city"], start=current_position,
                                          end=poi_sel["name"], start_time=current_time, method=poi_plan["transport_preference"], verbose=False)

                    if len(transports_sel) == 3:
                        transports_sel[1]["tickets"] = query["people_number"]
                    elif transports_sel[0]["mode"] == "taxi":
                        transports_sel[0]["car"] = int(
                            (query["people_number"] - 1) / 4) + 1
                    arrived_time = transports_sel[-1]["end_time"]

                    # print(transports_sel)
                    # print(poi_sel, arrived_time)

                    # exit(0)
                    # 开放时间
                    opentime, endtime = poi_sel["opentime"],  poi_sel["endtime"]

                    # too late
                    if time_compare_if_earlier_equal("21:00", arrived_time):
                        continue
                    # it is closed ...
                    if time_compare_if_earlier_equal(endtime, arrived_time):
                        continue

                    if time_compare_if_earlier_equal(arrived_time, opentime):
                        act_start_time = opentime
                    else:
                        act_start_time = arrived_time

                    # print("info before search poi type", current_day, current_time, current_position, poi_plan, plan)

                    if ("accommodation" in poi_plan) and (current_day < query["days"]-1):
                        candidates_type.append("hotel")

                    # and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], add_time_delta(current_time, 180)):
                    if current_day == query["days"] - 1 and current_time != "":
                        candidates_type.append("back-intercity-transport")

                    poi_time = 90

                    planning_info = []

                    # print(poi_sel)

                    recommendmintime = int(poi_sel["recommendmintime"] * 60)
                    recommendmaxtime = int(poi_sel["recommendmaxtime"] * 60)

                    planning_info.append(
                        "请作为一个旅行规划助手帮助我想构建行程，我的需求是{}".format(query["nature_language"]))
                    planning_info.append("现在是第{}天{},我到达了{},这个景点的开放时间是[{}--{}]，建议的游玩时间是{}-{}分钟，请帮助我思考我在这个景点游玩多久".format(
                        current_day + 1, current_time, poi_sel["name"], opentime, endtime, recommendmintime, recommendmaxtime))

                    planning_info.append(
                        "在邻近中午的时间找家餐厅吃饭，注意午餐时间要求在[11:00 -- 13:00]，请在游玩后预留时间用于交通到达相应地点")
                    planning_info.append(
                        "在邻近傍晚的时间找家餐厅吃饭，注意午餐时间要求在[17:00 -- 20:00]，请在游玩后预留时间用于交通到达相应地点")

                    if ("accommodation" in poi_plan) and (current_day < query["days"]-1):
                        planning_info.append(
                            "在一天中游览不同的景点，注意景点的游览和中途的交通要花费一定时间，请在游玩后预留时间以便在合适时间用餐、在夜间回到酒店")

                    # and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], add_time_delta(current_time, 180)):
                    if current_day == query["days"] - 1 and current_time != "":
                        planning_info.append("今天是旅程的最后一天，可以选择回家,返程车票时间是{}，请预留时间用于交通到达相应地点".format(
                            poi_plan["back_transport"]["BeginTime"]))

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
                        "cost": int(poi_sel["price"]),
                        "start_time": act_start_time,
                        "end_time": act_end_time
                    }

                    plan[current_day]["activities"].append(activity_i)

                    new_time = act_end_time
                    new_position = poi_sel["name"]

                    attractions_visiting.append(attr_idx)
                    spot_type_visiting.append(poi_sel["type"])
                    attraction_names_visiting.append(poi_sel["name"])

                    success, plan = search_poi(
                        query, poi_plan, plan, new_time, new_position, current_day, verbose)

                    if success:
                        return True, plan
                    else:
                        if time.time() > time_before_search + TIME_CUT:
                            print("Searching TIME OUT !!!")
                            raise Exception("TIME_OUT")
                            return False, plan
                    plan[current_day]["activities"].pop()
                    attractions_visiting.pop()
                    spot_type_visiting.pop()
                    attraction_names_visiting.pop()

        else:
            # raise Exception("Not Implemented.")
            print("incorrect poi type: {}".format(poi_type))
            continue

        candidates_type.remove(poi_type)
        print("try another poi type")

    return False, plan


def search_plan(query, poi_plan, fast_mode=False, result_dir=None):

    global llm_step
    llm_step = 0

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

    # if train_go is None:
    #     train_go_num = 0
    # if flight_back is None:
    #     flight_back_num = 0
    # if train_back is None:
    #     train_back_num = 0

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

    if ("hotel_feature_unseen" in query):

        concept_seen = hotel_info["featurehoteltype"].unique()
        # print(list(concept_seen))

        info_list = [
            "请作为一个旅行规划助手，帮助我挑选酒店，酒店有这些特征：{}, 我需要在以下酒店描述中选择一个最匹配的：{}".format(
                str(query["hotel_feature_unseen"]), concept_seen),
            "请确保你的输出在给出的描述中"
        ]

        sel_feature = select_feature(planning_info=info_list)

        hotel_info = accommodation.select(
            target_city, "featurehoteltype", lambda x: x == sel_feature)

    if "hotel_names" in query:
        hotel_info = accommodation.select(
            target_city, "name", lambda x: x == list(query["hotel_names"])[0])

    if ("hotel_names_unseen" in query):
        concept_seen = hotel_info["name"].unique()

        info_list = [
            "请作为一个旅行规划助手，帮助我挑选酒店，我想去{}酒店, 我需要在以下酒店名称中选择一个最匹配的：{}".format(
                str(query["hotel_names_unseen"]), concept_seen),
            "请确保你的输出在给出的名字中"
        ]
        sel_name = select_feature(planning_info=info_list)

        # print(sel_name)

        hotel_info = accommodation.select(
            target_city, "name", lambda x: x == sel_name)

        print(hotel_info)
        # exit(0)

    # exit(0)

    if "room_type" in query:
        hotel_info = hotel_info[hotel_info["numbed"] == query["room_type"]]

    if "hotel_price" in query:
        hotel_info = hotel_info[hotel_info["price"]
                                * query["rooms"] <= query["hotel_price"]]

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

    if fast_mode:
        attr_path = os.path.join(
            result_dir, "score_{}_attr.csv".format(poi_plan["query_idx"]))
        rest_path = os.path.join(
            result_dir, "score_{}_rest.csv".format(poi_plan["query_idx"]))
        try:

            attr_info = pd.read_csv(attr_path)
            print("load <<< ", attr_path)
        except:
            # attractions:
            info_list = [query["nature_language"]]
            info_list.append(
                "在这次在{}的旅行中，请帮我评估每个景点的游玩重要性，看看是否需要安排到行程里".format(query["target_city"]))

            if ("spot_type" in query) or ("spot_type_unseen" in query):

                req_spot_type = set()

                if "spot_type" in query:
                    req_spot_type = set.union(
                        req_spot_type, set(query["spot_type"]))

                if "spot_type_unseen" in query:
                    req_spot_type = set.union(
                        req_spot_type, set(query["spot_type_unseen"]))

                info_list.append(
                    "本次旅行有对景点类型的需求，这些类型至少需要去一次 {}, 在选择景点的时候请根据景点的景点类型加以考虑".format(req_spot_type))
            # print(info_list)

            if ("attraction_names" in query) or ("attraction_names_unseen" in query):

                req_attr_name = set()
                if "attraction_names" in query:
                    req_attr_name = set.union(
                        req_attr_name, set(query["attraction_names"]))

                if "attraction_names_unseen" in query:
                    req_attr_name = set.union(req_attr_name, set(
                        query["attraction_names_unseen"]))

                info_list.append(
                    "本次旅行有对特定景点的需求，这几个地方至少需要去一次 {}, 在选择景点的时候请根据景点的名字加以考虑".format(req_attr_name))
            info_list.append("请根据以上信息考虑什么样的景点满足我们的需求")
            attr_info = poi_info["attractions"]

            poi_info_list = []
            for idx in range(len(attr_info)):
                res_i = attr_info.iloc[idx]
                poi_info_list.append("**{}** name: {} , price_per_preson: {}, type: {}".format(
                    idx, res_i["name"], res_i["price"], res_i['type']))

            score_list = score_poi_think_overall_act_page(
                info_list, poi_info_list, react=True)
            attr_info["importance"] = score_list
            attr_info = attr_info.sort_values(
                by=["importance"], ascending=False)
            # attr_info.to_csv(attr_path, index=False)
            # print("save  >>> ", attr_path)

        poi_info["attractions"] = attr_info

        # restruants

        try:

            rest_info = pd.read_csv(rest_path)
            print("load <<< ", rest_path)
        except:
            info_list = [query["nature_language"]]

            info_list.append(
                "在这次在{}的旅行中,请帮我选择一些餐厅去吃，评估每个餐厅的相关性，看看是否需要安排到行程里".format(query["target_city"]))

            rest_info = poi_info["restaurants"]

            if "cost_wo_intercity" in query:
                rest_cost = query["cost_wo_intercity"]
                info_list.append(
                    "本次旅行有整体开销的约束，所有餐饮和景点的预算为: {} 元".format(rest_cost))
                upper_budget = rest_cost / query["people_number"]
                rest_info = rest_info[rest_info["price"] <= upper_budget]

            if ("food_type" in query) or ("food_type_unseen" in query):
                req_food_type = set()

                if "food_type" in query:
                    req_food_type = set.union(
                        req_food_type, set(query["food_type"]))

                if "food_type_unseen" in query:
                    req_food_type = set.union(
                        req_food_type, set(query["food_type_unseen"]))

                info_list.append(
                    "本次旅行有对餐饮类型的需求，这些类型至少需要吃一次 {}, 在选择餐厅的时候请根据餐厅的餐饮类型加以考虑".format(req_food_type))

            if ("restaurant_names" in query) or ("restaurant_names_unseen" in query):

                req_res_name = set()
                if "restaurant_names" in query:
                    req_res_name = set.union(
                        req_res_name, set(query["restaurant_names"]))

                if "restaurant_names_unseen" in query:
                    req_res_name = set.union(req_res_name, set(
                        query["restaurant_names_unseen"]))

                info_list.append(
                    "本次旅行有对特定餐饮商家的需求，这几家至少需要去一次 {}, 在选择餐厅的时候请根据餐厅的名字加以考虑".format(req_res_name))

            info_list.append("请根据以上信息考虑什么样的餐厅满足我们的需求")

            poi_info_list = []
            score_list = []
            for idx in range(len(rest_info)):

                res_i = poi_info["restaurants"].iloc[idx]

                poi_info_list.append("**{}** name: {} , cusine: {}, price_per_preson: {}, recommended food: {}".format(
                    idx, res_i["name"], res_i["cuisine"], res_i["price"], res_i["recommendedfood"]))
                # poi_info_list.append("**{}** name: {} , arrived_time: {}, cusine: {}, price_per_preson: {}".format(idx, res_i["name"], arrived_time, res_i["cuisine"], res_i["price"]))

            score_list = score_poi_think_overall_act_page(
                info_list, poi_info_list, react=True)
            rest_info["importance"] = score_list
            rest_info = rest_info.sort_values(
                by=["importance"], ascending=False)

            # rest_info.to_csv(rest_path, index=False)
            # print("save  >>> ", rest_path)
        poi_info["restaurants"] = rest_info

    # print(poi_info["attractions"])
    # print(poi_info["restaurants"])
    # exit(0)

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

            # print(poi_plan)

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


def symbolic_search(query, query_idx, if_fast=False, result_dir="results/"):

    global avialable_plan

    avialable_plan = {"info": "TIME OUT"}
    global fast_mode
    fast_mode = if_fast

    global time_before_search
    time_before_search = time.time()

    try:
        query["days"] = int(query["hard_logic"][0].split("==")[1])
        query["people_number"] = int(query["hard_logic"][1].split("==")[1])

    except:
        day_flag, people_flag = False, False

        for item in query["hard_logic"]:
            if "days==" in item:
                day_flag = True
                query["days"] = int(item.split("==")[1])
            if "people_number==" in item:
                people_flag = True
                query["people_number"] = int(item.split("==")[1])

        if day_flag and people_flag:
            pass
        else:
            raise Exception(
                "You must provide the information about days and people")

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
    # print(seen_rest_type_concept)

    # exit(0)

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

    success, plan = search_plan(
        query, {"query_idx": query_idx}, fast_mode=fast_mode, result_dir=result_dir)

    print(success, plan)

    global llm_time_cost
    return success, plan, llm_time_cost

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--level', '-l', type=str, default="easy", choices=[
                        "easy", "medium", "medium_plus", "human", "human_small", "example"], help="query subset")
    parser.add_argument('--index', '-i', type=int,
                        default=None, help="query index")
    parser.add_argument('--start', '-s', type=int,
                        default=None, help="start query index")
    parser.add_argument('--mode', '-m', type=str,
                        default="human", help="backend model")
    parser.add_argument('--fast', action="store_true", default=False)

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
        "/lamda/shaojj/codes/TravelPlanner-main/ChinaTravel/data/{}.json".format(query_level))

    success_count, total = 0, 0

    result_path = "results_tmp/"
    if not os.path.exists(result_path+"{}".format(query_level)):
        os.makedirs(result_path+"{}".format(query_level))

    global fast_mode

    model_name = args.mode
    if args.fast:
        model_name = model_name + "_fast"
        fast_mode = True
    else:
        fast_mode = False

    if not os.path.exists(result_path+"{}/{}".format(query_level, model_name)):
        os.makedirs(result_path+"{}/{}".format(query_level, model_name))
    result_dir = result_path+"{}/{}".format(query_level, model_name)

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
        dump_f.write(datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')+"\n")
        dump_f.write("testing range: {}".format(str(test_idx))+"\n")

    for idx in test_idx:

        query_i = test_input[idx]

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
        success, plan, llm_time_cost = symbolic_search(
            query_i, idx, if_fast=fast_mode, result_dir=result_dir)

        print("llm time cost: ", llm_time_cost)
        # except Exception as error_message:
        #     success = False
        #     # plan = {"exception": error_message}
        #     print(error_message)
        #     plan = avialable_plan

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

    with open(result_dir + "/result_stat_{}.json".format(str(time.time())), "w", encoding="utf8") as dump_f:
        json.dump(res_stat, dump_f, ensure_ascii=False, indent=4)

    with open(result_dir + "/fail_list.txt", "a+") as dump_f:
        dump_f.write(
            "success rate [{}]: {}/{}".format(query_level, success_count, total)+"\n")
