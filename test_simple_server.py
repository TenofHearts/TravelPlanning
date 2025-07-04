#!/usr/bin/env python3
"""
简化的语音输入测试启动脚本
避免复杂依赖问题
"""
import sys
import os
import traceback

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """测试基本导入"""
    try:
        import flask
        print("✅ Flask导入成功")
    except ImportError as e:
        print(f"❌ Flask导入失败: {e}")
        return False
    
    try:
        import openai
        print("✅ OpenAI导入成功")
    except ImportError as e:
        print(f"❌ OpenAI导入失败: {e}")
        return False
    
    return True

def start_simple_server():
    """启动简化的服务器"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def home():
        return "🎙️ 语音输入旅行计划系统 - 测试版"
    
    @app.route('/test')
    def test_page():
        try:
            with open('test_voice_frontend.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return """
            <html>
            <head><title>测试页面</title></head>
            <body>
                <h1>🎙️ 语音输入测试</h1>
                <p>测试页面文件未找到，但服务器正在运行</p>
                <p>API端点: <code>/plan/voice</code></p>
            </body>
            </html>
            """
    
    @app.route('/plan/voice', methods=['POST'])
    def voice_to_plan():
        return jsonify({
            "success": 0,
            "error": "简化测试版本，完整功能需要完整应用启动"
        })
    
    print("🚀 启动简化测试服务器...")
    print("📍 访问地址: http://localhost:5000")
    print("📋 测试页面: http://localhost:5000/test")
    print("🛑 按Ctrl+C停止服务")
    
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    print("🔧 语音输入功能测试")
    print("=" * 40)
    
    if not test_basic_imports():
        print("❌ 基本依赖缺失，请安装必要的包")
        sys.exit(1)
    
    try:
        start_simple_server()
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        traceback.print_exc()
