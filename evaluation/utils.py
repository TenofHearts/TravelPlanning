import json
import jsonschema
from jsonschema import validate


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_json(json_data, schema):
    try:
        validate(instance=json_data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        return False


def time_diff(time1, time2):
    """
    计算两个时间的时间差, 返回时间差, 单位为分钟
    """
    time1 = time1.split(":")
    time2 = time2.split(":")
    return (int(time1[0]) - int(time2[0])) * 60 + int(time1[1]) - int(time2[1])


def score_go_intercity_transport(transport):
    """
    根据时间和排序, 排序标准如下:
    1. BeginTime 在10:00之后, 越早越优先选择, 在10:00之前, 越晚越优先选择
    2. Duration 越短越优先选择
    3. Cost 越低优先级越高
    4. 以上三个标准不分先后, 可以通过函数评分, 评分越高, 优先级越高
    """
    score = (
        abs(time_diff(transport["BeginTime"], "10:00"))
        + transport["Duration"]
        + transport["Cost"]
    )
    return score


def score_back_intercity_transport(transport):
    """
    根据时间和排序, 排序标准如下:
    1. BeginTime 在16:00之后, 越早越优先选择, 在16:00之前, 越晚越优先选择
    2. Duration 越短越优先选择
    3. Cost 越低优先级越高
    """
    score = (
        abs(time_diff(transport["BeginTime"], "16:00")) * 20
        + transport["Duration"] * 50
        + transport["Cost"]
    )
    return score


if __name__ == "__main__":
    schema_file_path = "./output_schema.json"
    json_file_path_template = "../results/test_20240909091404/query_{}_result.json"

    schema = load_json_file(schema_file_path)
    acc = 0
    for i in range(10):
        try:
            json_data = load_json_file(json_file_path_template.format(i))
            if validate_json(json_data, schema):
                acc += 1
            else:
                print("Error {}".format(i))
        except:
            print("Error {}".format(i))
            continue
    print(acc / 10)
