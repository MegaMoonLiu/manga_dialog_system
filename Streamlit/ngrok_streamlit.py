from pyngrok import ngrok
import subprocess

ngrok.set_auth_token("")

# 启动 Streamlit 应用
subprocess.Popen(["streamlit", "run", "app.py"])

# 使用 ngrok 暴露本地服务
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")

# 保持运行
input("Press Enter to terminate...")
