# 安装：pip install streamlit
# 保存为 app.py，运行：streamlit run app.py

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="AI ChatBot",
    page_icon="🤖",
    layout="wide"
)

# 标题
st.title("🤖 AI ChatBot")
st.caption("基于 LangGraph 的智能对话助手")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    model = st.selectbox("模型", ["gpt-4", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)

    st.divider()
    st.info("""
    💡 **使用说明**
    - 在下方输入框输入消息
    - 按 Enter 发送
    - 查看 AI 响应
    """)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 用户输入
if prompt := st.chat_input("输入你的消息..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 模拟 AI 响应
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            import time
            time.sleep(1)
            response = f"[{model}] 收到消息: {prompt}"
            st.write(response)

    # 添加 AI 响应
    st.session_state.messages.append({"role": "assistant", "content": response})

# 清除历史按钮
if st.button("🗑️ 清除对话历史"):
    st.session_state.messages = []
    st.rerun()