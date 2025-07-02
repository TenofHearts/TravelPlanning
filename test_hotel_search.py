#!/usr/bin/env python3
"""
酒店搜索测试脚本
用于诊断和修复酒店搜索为空的问题
"""

import sys
import os
sys.path.append(".")

from tools.hotels.apis import Accommodations

def test_hotel_search():
    """测试酒店搜索功能"""
    
    print("=" * 50)
    print("开始测试酒店搜索功能")
    print("=" * 50)
    
    # 初始化酒店API
    try:
        accommodation = Accommodations()
        print("✓ 酒店API初始化成功")
    except Exception as e:
        print(f"✗ 酒店API初始化失败: {str(e)}")
        return False
    
    # 测试不同的城市和搜索条件
    test_cases = [
        {"city": "北京", "keywords": "酒店"},
        {"city": "上海", "keywords": "酒店"},
        {"city": "南京", "keywords": "酒店"},
        {"city": "北京", "keywords": "温泉酒店"},
        {"city": "北京", "keywords": "酒店 酒店名字为北京饭店"},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"城市: {test_case['city']}")
        print(f"关键词: {test_case['keywords']}")
        
        try:
            result = accommodation.select(
                city=test_case['city'], 
                keywords=test_case['keywords']
            )
            
            if result is not None and not result.empty:
                print(f"✓ 搜索成功，找到 {len(result)} 个酒店")
                print(f"第一个酒店: {result.iloc[0]['name'] if 'name' in result.columns else '未知'}")
            else:
                print("✗ 搜索结果为空")
                
        except Exception as e:
            print(f"✗ 搜索失败: {str(e)}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_hotel_search()
