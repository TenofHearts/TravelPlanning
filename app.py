from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from agent import plan_main, modify_plan
import random
from datetime import datetime
from copy import deepcopy


app = Flask(__name__)
cors = CORS(app)
executor = ThreadPoolExecutor(max_workers=5)  # 可以根据需要调整线程数
tasks = {}  # 用于存储任务的 Future 对象


def run_async_task(test_request, task_id):
    """在后台线程中执行耗时的任务"""
    try:
        print(f"Running task with ID {task_id}")
        result = plan_main(test_request, task_id)
        print(f"Task {task_id} completed")
        return result  # 直接返回结果，Future 会自动处理
    except Exception as e:
        print(f"Task {task_id} failed with error: {e}")
        return {"error": str(e)}


@app.route("/", methods=["GET"])
def hello():
    return "Hello, World!"


@app.route("/test_plan", methods=["POST"])
def get_request():
    """
    接收旅行规划请求并启动异步任务

    Args:
        请求体JSON格式，包含以下必需字段：
        - daysCount (int): 旅行天数，范围建议1-7天
        - startCity (str): 出发城市，必须使用中文城市名
        - destinationCity (str): 目的地城市，必须使用中文城市名
        - peopleCount (int): 旅行人数，范围建议1-10人
        - additionalRequirements (str, optional): 额外需求描述，如"我想吃火锅"、"喜欢自然风光"等

    Returns:
        JSON响应 (HTTP 200):
        - task_id (str): 任务ID，用于后续查询结果
        - message (str): 固定为"Task is running in the background."

    Notes:
        - 任务为异步执行，需要使用返回的task_id调用get_plan接口获取结果
        - 支持的城市列表：上海、北京、深圳、广州、重庆、苏州、成都、杭州、武汉、南京
        - 任务执行时间取决于旅行天数和复杂度，通常需要几秒到几十秒
    """
    test_request = request.json

    task_id = datetime.now().strftime("%Y%m%d%H%M%S")
    # future = executor.submit(run_async_task, test_request, task_id)
    result = run_async_task(test_request, task_id)
    # tasks[task_id] = future  # 将 Future 对象存储到字典中
    tasks[task_id] = result
    # return jsonify(test_dict)
    return jsonify(
        {"task_id": task_id, "message": "Task is running in the background."}
    )


def modify_plan():
    """
    接收用户的修改请求, 并启动异步重新规划

    Args:
        task_id (str): 任务ID
        request (str): 修改的描述
    """


@app.route("/get_plan", methods=["GET"])
def get_plan():
    """
    获取旅行规划结果

    Args:
        task_id (str): 任务ID，通过URL参数传递，从test_plan接口获得
        day (int, optional): 指定获取第几天的行程，不传则返回完整行程概览

    Returns:
        JSON响应，包含以下可能的key：

        成功情况 (HTTP 200):
        - success (int): 固定为1，表示任务成功
        - plan (dict): 旅行计划详情，包含以下子key：
            - intercity_transport_start (dict): 出发城际交通信息（仅在day=1时存在）
                - start (str): 出发地点
                - end (str): 到达地点
                - start_time (str): 出发时间
                - end_time (str): 到达时间
                - cost (int): 费用
                - mode (str): 交通方式
            - intercity_transport_end (dict): 返回城际交通信息（仅在day=最后一天时存在）
                - start (str): 出发地点
                - end (str): 到达地点
                - start_time (str): 出发时间
                - end_time (str): 到达时间
                - cost (int): 费用
                - mode (str): 交通方式
            - activities (list): 当日活动列表（当指定day时）或每日活动列表（当未指定day时）
                每个活动包含：
                - position (str): 活动地点
                - type (str): 活动类型（breakfast/lunch/dinner/attraction等）
                - start_time (str): 开始时间
                - end_time (str): 结束时间
                - cost (int): 活动费用
                - picture (str): 活动图片URL
                - trans_time (int, optional): 交通时间（分钟）
                - trans_distance (float, optional): 交通距离（公里）
                - trans_cost (int, optional): 交通费用
                - trans_type (str, optional): 交通方式
                - trans_detail (list, optional): 详细交通信息
                - food_list (str, optional): 推荐菜品（仅餐厅类型活动）
            - position_detail (list, optional): 位置坐标列表（当指定day时）
            - target_city (str, optional): 目标城市（当指定day时）

        失败情况 (HTTP 404):
        - error (str): 错误信息
            - "Task failed.": 当任务执行失败时（success=0）
            - "Task ID not found.": 当task_id不存在时
    """
    task_id = request.args.get("task_id")
    task_id = str(task_id)
    need_day = True
    day = request.args.get("day")
    # print(day)
    try:
        day = int(day)
    except:
        need_day = False
    # print(task_id,type(task_id))
    # if task_id in tasks:
    #     future = tasks[task_id]
    #     if future.done():  # 检查任务是否完成
    #         result = future.result()  # 获取结果
    #         # del tasks[task_id]  # 移除已完成的任务
    #         return jsonify({"task_id": task_id, "status": "completed", "result": result})
    #     else:
    #         return jsonify({"task_id": task_id, "status": "running"})
    # else:
    #     return jsonify({"error": "Task ID not found."}), 404
    if task_id in tasks:

        tmp = deepcopy(tasks[task_id])
        print(tmp.keys())
        print(tmp["plan"].keys())
        if tmp["success"] == 0:
            return jsonify({"error": "Task failed."}), 404
        if need_day:
            total_day = len(tmp["plan"]["activities"])
            tmp["plan"]["activities"] = tmp["plan"]["activities"][day - 1]
            if day != 1:
                tmp["plan"].pop("intercity_transport_start")
            if day != total_day:
                tmp["plan"].pop("intercity_transport_end")
            if day == total_day:
                tmp["plan"]["activities"][-1].pop("position")
            tmp["plan"]["position_detail"] = tmp["brief_plan"]["itinerary"][day - 1][
                "position_detail"
            ]
            tmp["plan"]["target_city"] = tmp["brief_plan"]["target_city"]
        else:
            tmp["plan"] = tmp["brief_plan"]
        tmp.pop("brief_plan")
        return jsonify(tmp)
    else:
        return jsonify({"error": "Task ID not found."}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
