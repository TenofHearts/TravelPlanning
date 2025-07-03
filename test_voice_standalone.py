"""
独立的语音输入测试脚本
测试语音转文本和信息提取功能，不依赖整个应用
"""
import os
import io
import json
import re
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: OpenAI库未安装，语音识别功能将不可用")

# 语音识别配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-your-api-key-here")
SPEECH_CONFIG = {
    'model': 'whisper-1',
    'language': 'zh',
    'max_file_size': 25 * 1024 * 1024,  # 25MB
    'timeout': 30
}

def simple_extract_travel_info(text):
    """
    简单的基于规则的旅行信息提取
    """
    # 支持的城市列表
    cities = ['上海', '北京', '深圳', '广州', '重庆', '苏州', '成都', '杭州', '武汉', '南京']
    
    # 提取天数
    days_patterns = [
        r'(\d+)天',
        r'(\d+)日',
        r'天数.*?(\d+)',
        r'(\d+).*?天',
        r'一周', r'1周', r'7天',
        r'两周', r'2周', r'14天',
        r'三周', r'3周', r'21天'
    ]
    
    days_count = None
    for pattern in days_patterns:
        if pattern in ['一周', '1周', '7天']:
            if any(p in text for p in ['一周', '1周']):
                days_count = 7
                break
        elif pattern in ['两周', '2周', '14天']:
            if any(p in text for p in ['两周', '2周']):
                days_count = 14
                break
        elif pattern in ['三周', '3周', '21天']:
            if any(p in text for p in ['三周', '3周']):
                days_count = 21
                break
        else:
            match = re.search(pattern, text)
            if match:
                days_count = int(match.group(1))
                break
    
    # 提取城市并分析出发地和目的地
    found_cities = []
    city_positions = []
    
    for city in cities:
        if city in text:
            found_cities.append(city)
            city_positions.append((text.find(city), city))
    
    # 按出现位置排序
    city_positions.sort(key=lambda x: x[0])
    
    # 分析出发地和目的地
    start_city = None
    destination_city = None
    
    # 寻找"从...到..."或"...去..."的模式
    from_to_patterns = [
        r'从(.+?)到(.+?)[玩游旅]',
        r'从(.+?)去(.+?)[玩游旅]',
        r'从(.+?)出发.*?去(.+?)[玩游旅]',
        r'从(.+?)出发.*?到(.+?)[玩游旅]',
        r'(.+?)到(.+?)[玩游旅]',
        r'去(.+?)[玩游旅].*?从(.+?)出发',
        r'要去(.+?)[玩游旅].*?从(.+?)出发'
    ]
    
    for pattern in from_to_patterns:
        match = re.search(pattern, text)
        if match:
            if '去' in pattern and '从' in pattern and '出发' in pattern:
                # 处理"去...从...出发"的情况
                dest_str = match.group(1)
                start_str = match.group(2)
            else:
                start_str = match.group(1)
                dest_str = match.group(2)
            
            # 在匹配的字符串中找城市
            for city in cities:
                if city in start_str:
                    start_city = city
                if city in dest_str:
                    destination_city = city
            
            if start_city and destination_city:
                break
    
    # 如果没有找到明确的方向，使用出现顺序
    if not start_city or not destination_city:
        if len(city_positions) >= 2:
            start_city = city_positions[0][1]
            destination_city = city_positions[1][1]
        elif len(city_positions) == 1:
            destination_city = city_positions[0][1]
            start_city = '北京'  # 默认出发地
    
    # 提取人数
    people_patterns = [
        r'(\d+)人',
        r'(\d+)个人',
        r'人数.*?(\d+)',
        r'一个人', r'1个人',
        r'两个人', r'2个人',
        r'三个人', r'3个人',
        r'四个人', r'4个人',
        r'五个人', r'5个人',
        r'我们有(\d+)人',
        r'我们(\d+)人',
        r'我们有(\d+)个人'
    ]
    
    people_count = 1  # 默认1人
    for pattern in people_patterns:
        if pattern in ['一个人', '1个人']:
            if any(p in text for p in ['一个人', '1个人']):
                people_count = 1
                break
        elif pattern in ['两个人', '2个人']:
            if any(p in text for p in ['两个人', '2个人']):
                people_count = 2
                break
        elif pattern in ['三个人', '3个人']:
            if any(p in text for p in ['三个人', '3个人']):
                people_count = 3
                break
        elif pattern in ['四个人', '4个人']:
            if any(p in text for p in ['四个人', '4个人']):
                people_count = 4
                break
        elif pattern in ['五个人', '5个人']:
            if any(p in text for p in ['五个人', '5个人']):
                people_count = 5
                break
        else:
            match = re.search(pattern, text)
            if match:
                people_count = int(match.group(1))
                break
    
    # 构造结果
    if days_count and start_city and destination_city:
        return {
            'daysCount': days_count,
            'startCity': start_city,
            'destinationCity': destination_city,
            'peopleCount': people_count,
            'additionalRequirements': text
        }
    
    return None

def test_text_extraction():
    """
    测试文本提取功能
    """
    print("🔍 测试文本提取功能...")
    
    test_cases = [
        "我想从上海到北京旅游3天，我一个人，想要体验红色之旅",
        "明天开始从广州去深圳玩2天，我们有4个人，想吃当地美食",
        "计划下周从成都到重庆的5天旅行，两个人，喜欢自然风光和火锅",
        "我要去杭州玩一周，从苏州出发，3个人，想看西湖",
        "北京到上海2天1夜，一个人，想吃小笼包"
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\n📝 测试案例 {i+1}: {text}")
        
        result = simple_extract_travel_info(text)
        if result:
            print(f"✅ 提取成功: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print("❌ 提取失败，无法获取必要信息")

def test_speech_to_text(audio_file_path):
    """
    测试语音转文本功能
    """
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI库未安装，无法测试语音转文本")
        return None
    
    if not os.path.exists(audio_file_path):
        print(f"❌ 音频文件不存在: {audio_file_path}")
        return None
    
    print(f"🎙️ 测试语音转文本: {audio_file_path}")
    
    try:
        # 创建OpenAI客户端
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # 读取音频文件
        with open(audio_file_path, 'rb') as f:
            # 调用Whisper API
            transcript = client.audio.transcriptions.create(
                model=SPEECH_CONFIG['model'],
                file=f,
                language=SPEECH_CONFIG['language'],
                response_format="text"
            )
        
        transcription = transcript.strip()
        print(f"✅ 转写成功: {transcription}")
        
        # 提取旅行信息
        travel_info = simple_extract_travel_info(transcription)
        if travel_info:
            print(f"🎯 提取的旅行信息: {json.dumps(travel_info, ensure_ascii=False, indent=2)}")
        else:
            print("⚠️ 无法从转写文本中提取旅行信息")
        
        return transcription
        
    except Exception as e:
        print(f"❌ 语音转文本失败: {e}")
        return None

def main():
    """
    主测试函数
    """
    print("🎙️ 语音输入功能独立测试")
    print("=" * 50)
    
    # 测试文本提取
    test_text_extraction()
    
    print("\n" + "=" * 50)
    
    # 测试语音转文本（如果有音频文件）
    audio_file = "test_audio.wav"  # 替换为你的音频文件路径
    
    if os.path.exists(audio_file):
        test_speech_to_text(audio_file)
    else:
        print(f"⚠️ 音频文件 {audio_file} 不存在")
        print("请准备一个包含旅行计划描述的音频文件进行测试")
        print("例如录制: '我想从上海到北京旅游3天，我一个人，想要体验红色之旅'")
    
    print("\n✅ 测试完成!")

if __name__ == "__main__":
    main()
