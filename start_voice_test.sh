#!/bin/bash
echo "🚀 启动旅行计划语音输入测试"
echo "=================================="

# 检查是否设置了OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告：未设置OPENAI_API_KEY环境变量"
    echo "语音识别功能可能无法正常工作"
    echo "请设置环境变量：export OPENAI_API_KEY='your-api-key-here'"
    echo ""
fi

# 检查依赖是否安装
echo "📦 检查依赖..."
python -c "import openai, flask, flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 依赖缺失，正在安装..."
    pip install -r requirements.txt
fi

# 启动Flask应用
echo "🔧 启动Flask应用..."
echo "API地址: http://localhost:5000"
echo "测试页面: http://localhost:5000/test"
echo "--------------------------------"
echo "按Ctrl+C停止服务"
echo ""

# 设置环境变量并启动应用
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
