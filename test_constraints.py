#!/usr/bin/env python3
"""
测试约束验证功能
"""

import sys
sys.path.append(".")

from evaluation.hard_constraint import get_symbolic_concepts, evaluate_constraints
from evaluation.commonsense_constraint import func_commonsense_constraints

def test_constraints():
    print("=" * 50)
    print("测试约束验证功能")
    print("=" * 50)
    
    # 测试查询
    test_query = {
        'nature_language': '当前位置上海,打算1个人一起去北京旅游2天,请给我一个旅行规划.',
        'start_city': '上海',
        'target_city': '北京', 
        'people_number': 1,
        'days': 2,
        'hard_logic': ['cost<=2000']
    }
    
    # 测试计划
    test_plan = {
        'people_number': 1,
        'start_city': '上海', 
        'target_city': '北京',
        'itinerary': [
            {
                'day': 1,
                'activities': [
                    {
                        'start_time': '08:00',
                        'end_time': '10:00',
                        'start': '上海虹桥机场',
                        'end': '北京首都机场',
                        'ID': 'CA1234',
                        'type': 'airplane',
                        'transports': [],
                        'cost': 800,
                        'tickets': 1
                    },
                    {
                        'position': '故宫博物院',
                        'type': 'attraction',
                        'transports': [
                            {
                                'mode': 'metro',
                                'cost': 5,
                                'tickets': 1
                            }
                        ],
                        'cost': 60,
                        'start_time': '11:00',
                        'end_time': '13:00'
                    },
                    {
                        'position': '全聚德',
                        'type': 'lunch',
                        'transports': [
                            {
                                'mode': 'taxi',
                                'cost': 20,
                                'tickets': 1
                            }
                        ],
                        'cost': 150,
                        'start_time': '13:30',
                        'end_time': '14:30'
                    }
                ]
            },
            {
                'day': 2,
                'activities': [
                    {
                        'start_time': '16:00',
                        'end_time': '18:00',
                        'start': '北京首都机场',
                        'end': '上海虹桥机场',
                        'ID': 'CA5678',
                        'type': 'airplane',
                        'transports': [],
                        'cost': 800,
                        'tickets': 1
                    }
                ]
            }
        ]
    }
    
    print("1. 测试 get_symbolic_concepts 函数")
    try:
        extracted_vars = get_symbolic_concepts(test_query, test_plan)
        print("✓ 成功提取符号概念:")
        for key, value in extracted_vars.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        print(f"   错误类型: {type(e).__name__}")
        return False
    
    print("\n2. 测试逻辑约束验证")
    try:
        logical_result = evaluate_constraints(extracted_vars, test_query['hard_logic'])
        print(f"✓ 逻辑约束验证结果: {logical_result}")
        for i, result in enumerate(logical_result):
            constraint = test_query['hard_logic'][i]
            status = "通过" if result else "失败"
            print(f"   {constraint}: {status}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        print(f"   错误类型: {type(e).__name__}")
        return False
    
    print("\n3. 测试常识约束验证")
    try:
        commonsense_result = func_commonsense_constraints(test_query, test_plan)
        print(f"✓ 常识约束验证结果: {commonsense_result}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        print(f"   错误类型: {type(e).__name__}")
        return False
    
    print("\n" + "=" * 50)
    print("约束验证功能测试完成!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_constraints()
