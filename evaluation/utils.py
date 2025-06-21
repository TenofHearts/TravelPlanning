import json
import jsonschema
from jsonschema import validate


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_json(json_data, schema):
    try:
        validate(instance=json_data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        return False


if __name__ == "__main__":
    schema_file_path = './output_schema.json'
    json_file_path_template = '../results/test_20240909091404/query_{}_result.json'

    schema = load_json_file(schema_file_path)
    acc = 0
    for i in range(10):
        try:
            json_data = load_json_file(json_file_path_template.format(i))
            if validate_json(json_data, schema):
                acc += 1
            else:
                print('Error {}'.format(i))
        except:
            print('Error {}'.format(i))
            continue
    print(acc/10)
