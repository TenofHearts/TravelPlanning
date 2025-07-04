"""
测试语音输入功能
"""
import requests
import json
import time
import os

# 测试配置
BASE_URL = "http://localhost:5000"
AUDIO_FILE_PATH = "test_audio.wav"  # 你需要准备一个测试音频文件

def test_voice_input():
    """测试语音输入功能"""
    print("开始测试语音输入功能...")
    
    # 检查音频文件是否存在
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"⚠️  测试音频文件 {AUDIO_FILE_PATH} 不存在")
        print("请准备一个包含旅行计划描述的音频文件，例如：")
        print("'我想从上海到北京旅游3天，我一个人，想要体验红色之旅，去一些红色景点，还想吃火锅和北京烤鸭'")
        return
    
    # 发送语音输入请求
    try:
        with open(AUDIO_FILE_PATH, 'rb') as f:
            files = {'audio_file': f}
            data = {
                'language': 'zh',
                'additional_context': '这是一个关于中国旅游的语音输入'
            }
            
            print("📤 发送语音输入请求...")
            response = requests.post(f"{BASE_URL}/plan/voice", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 语音输入成功!")
                print(f"📝 转写文本: {result.get('transcription', '')}")
                print(f"🎯 提取信息: {json.dumps(result.get('extracted_request', {}), ensure_ascii=False, indent=2)}")
                print(f"🆔 任务ID: {result.get('task_id', '')}")
                
                # 等待计划生成
                task_id = result.get('task_id')
                if task_id:
                    return poll_plan_result(task_id)
                    
            else:
                print(f"❌ 语音输入失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False

def poll_plan_result(task_id):
    """轮询获取计划结果"""
    print(f"⏳ 等待计划生成 (任务ID: {task_id})...")
    
    max_attempts = 30  # 最多等待30次
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/plan/result/{task_id}")
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success') == 1:
                    print("✅ 计划生成成功!")
                    plan = result.get('plan', {})
                    
                    # 简单显示计划概览
                    print("\n📋 计划概览:")
                    print(f"出发地: {plan.get('start_city', '')}")
                    print(f"目的地: {plan.get('target_city', '')}")
                    print(f"人数: {plan.get('people_number', '')}人")
                    print(f"天数: {len(plan.get('itinerary', []))}天")
                    
                    # 显示每日行程
                    itinerary = plan.get('itinerary', [])
                    for day_idx, day_plan in enumerate(itinerary):
                        print(f"\n第{day_idx + 1}天:")
                        activities = day_plan.get('activities', [])
                        for activity in activities:
                            activity_type = activity.get('type', '')
                            location = activity.get('position', activity.get('start', ''))
                            time_info = f"{activity.get('start_time', '')} - {activity.get('end_time', '')}"
                            print(f"  {time_info} | {activity_type} | {location}")
                    
                    return True
                    
                elif result.get('success') == 0:
                    print(f"❌ 计划生成失败: {result.get('message', '')}")
                    return False
                else:
                    print(f"⏳ 计划生成中... (尝试 {attempt + 1}/{max_attempts})")
                    time.sleep(2)
                    
            else:
                print(f"❌ 获取计划结果失败: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"⚠️  轮询过程中出错: {e}")
            time.sleep(2)
    
    print("⏰ 等待超时，计划生成可能需要更长时间")
    return False

def test_text_extraction():
    """测试文本提取功能"""
    print("\n开始测试文本提取功能...")
    
    # 模拟不同的语音转写文本
    test_cases = [
        "我想从上海到北京旅游3天，我一个人，想要体验红色之旅",
        "明天开始从广州去深圳玩2天，我们有4个人，想吃当地美食",
        "计划下周从成都到重庆的5天旅行，两个人，喜欢自然风光和火锅",
        "我要去杭州玩一周，从苏州出发，3个人，想看西湖"
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {text}")
        
        # 这里可以直接调用提取函数进行测试
        # 由于函数在app.py中，这里只是展示测试思路
        print("  (需要在app.py中调用extract_travel_request_from_text函数)")

if __name__ == "__main__":
    print("🎙️  语音输入功能测试")
    print("=" * 50)
    
    # 首先测试服务器是否运行
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ 服务器运行正常")
        else:
            print("❌ 服务器响应异常")
            exit(1)
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        print("请确保Flask应用已启动 (python app.py)")
        exit(1)
    
    # 测试语音输入功能
    test_voice_input()
    
    # 测试文本提取
    test_text_extraction()
    
    print("\n测试完成!")
