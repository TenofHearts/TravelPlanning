from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from agent import main
import random
from datetime import datetime
from copy import deepcopy


app = Flask(__name__)
cors=CORS(app)
executor = ThreadPoolExecutor(max_workers=5)  # 可以根据需要调整线程数
tasks = {}  # 用于存储任务的 Future 对象

def run_async_task(test_request, task_id):
    """在后台线程中执行耗时的任务"""
    try:
        print(f"Running task with ID {task_id}")
        result = main(test_request, task_id)
        print(f"Task {task_id} completed")
        return result  # 直接返回结果，Future 会自动处理
    except Exception as e:
        print(f"Task {task_id} failed with error: {e}")
        return {"error": str(e)}

@app.route("/", methods=["GET"])
def hello():
    return "Hello, World!"

@app.route("/test_plan", methods=["POST"])
def test_plan():
    test_request = request.json
    
    task_id = datetime.now().strftime("%Y%m%d%H%M%S")
    # future = executor.submit(run_async_task, test_request, task_id)
    result = run_async_task(test_request, task_id)
    # tasks[task_id] = future  # 将 Future 对象存储到字典中
    tasks[task_id] = result
    # return jsonify(test_dict)
    return jsonify({"task_id": task_id, "message": "Task is running in the background."})

@app.route("/get_plan", methods=["GET"])
def get_plan():
    task_id = request.args.get("task_id")
    task_id=str(task_id)
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
            tmp["plan"]["activities"] = tmp["plan"]["activities"][day-1]
            if day != 1:
                tmp["plan"].pop("intercity_transport_start")
            if day != total_day:
                tmp["plan"].pop("intercity_transport_end")
            if day == total_day:    
                tmp["plan"]["activities"][-1].pop("position")
            tmp["plan"]["position_detail"] = tmp["brief_plan"]["itinerary"][day-1]["position_detail"]
            tmp["plan"]["target_city"] = tmp["brief_plan"]["target_city"]
        else:
            tmp["plan"]=tmp["brief_plan"]
        tmp.pop("brief_plan")
        return jsonify(tmp)
    else:
        return jsonify({"error": "Task ID not found."}), 404
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081, debug=False)
