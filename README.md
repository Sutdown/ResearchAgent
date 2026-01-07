# ResearchAgent

一个基于多智能体系统的研究助手项目，通过协调多个专业化代理来完成复杂的研究任务。

## 项目概述

ResearchAgent 采用多智能体架构，将研究任务分解为多个子任务，由不同类型的代理协同工作。系统包含规划代理、研究代理、协调代理和报告代理，通过工作流引擎实现任务流程的自动化管理。

## 项目结构

```
ResearchAgent/
├── main.py                 # 主程序入口
├── func.py                 # 辅助功能函数
├── README.md               # 项目说明文档
├── .gitignore             # Git忽略文件配置
├── docs/                  # 文档目录
│   └── *.md               # 项目相关文档
├── images/                # 图片资源目录
│   └── *.png              # 项目相关图片
├── outputs/               # 输出目录
│   └── *.md               # 生成的报告文档
├── tests/                 # 测试目录
│   └── *.py               # 单元测试文件
├── vector_memory/         # 向量数据库存储
│   └── *.sqlite3          # 本地向量数据库文件
└── RAgents/               # 核心代码模块
    ├── agents/            # 智能代理模块
    │   ├── coordinator.py # 协调代理 - 负责任务协调和代理间通信
    │   ├── planner.py     # 规划代理 - 负责任务分解和计划制定
    │   ├── rapporteur.py  # 报告代理 - 负责结果整理和报告生成
    │   └── researcher.py  # 研究代理 - 负责信息收集和分析
    ├── langsmith/         # LangSmith集成模块
    │   └── langsmith.py   # LangSmith平台集成功能
    ├── llms/              # 语言模型模块
    │   └── *.py           # LLM调用和管理
    ├── prompts/           # 提示词模板
    │   └── *.md           # 各代理使用的提示词模板
    ├── tools/             # 工具模块
    │   └── *.py           # 研究工具和数据获取工具
    ├── utils/             # 工具函数模块
    │   └── *.py           # 通用工具函数
    └── workflow/          # 工作流模块
        ├── graph.py       # 工作流图定义 - 定义任务流程图
        ├── nodes.py       # 工作流节点 - 实现各个工作流节点
        └── state.py       # 状态管理 - 管理工作流状态和数据
```

## 核心功能

- **多智能体协作**：通过不同专业化的代理协同完成研究任务
- **任务分解**：将复杂研究任务自动分解为可执行的子任务
- **工作流管理**：基于状态机的工作流引擎确保任务有序执行
- **向量存储**：使用本地向量数据库存储和管理研究数据
- **报告生成**：自动整理研究结果并生成结构化报告

## 运行环境

- Python 3.8+
- 依赖包：requirements.txt 中列出的所有依赖

## 快速开始

### 方式一：命令行界面（传统）

1. 克隆项目到本地
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行主程序：
   ```bash
   python main.py
   ```

### 方式二：Web 界面（推荐）

1. 安装 Python 和 Node.js 依赖：
   ```bash
   # Python 依赖
   pip install -r requirements.txt
   pip install fastapi uvicorn python-multipart

   # 进入前端目录
   cd frontend
   
   # Node.js 依赖
   npm install
   ```

2. 配置环境变量（在项目根目录创建 `.env` 文件）：
   ```env
   # LLM API 密钥（至少配置一个）
   API_KEY=your_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_claude_key_here
   
   # 搜索 API（可选）
   TAVILY_API_KEY=your_tavily_key_here
   ```

3. 启动 Web 服务：
   ```bash
   # Windows 用户
   cd frontend && start.bat
   
   # Linux/macOS 用户  
   cd frontend && chmod +x start.sh && ./start.sh
   ```

4. 访问界面：http://localhost:5173

详细 Web 界面安装说明请参考：[WEB_SETUP.md](WEB_SETUP.md)

## 使用说明

### 命令行界面
1. 启动程序后，选择功能菜单
2. 输入研究主题或任务
3. 系统会自动分解任务并分配给相应的代理
4. 各代理协同工作完成研究任务
5. 最终生成的报告会保存在 `outputs/` 目录中

### Web 界面
1. 在浏览器中打开 http://localhost:5173
2. 使用图形化界面进行研究任务管理
3. 支持实时监控、人工审批、多轮对话等高级功能
4. 可视化配置系统参数

## 输出文件

- 研究报告：`outputs/` 目录下的 Markdown/HTML 文件
- 向量数据：`vector_memory/` 目录下的数据库文件

## 界面对比

| 功能 | 命令行界面 | Web 界面 |
|------|-----------|----------|
| 基础研究 | ✅ | ✅ |
| 多轮对话 | ✅ | ✅ |
| 实时进度 | ❌ | ✅ |
| 图形化配置 | ❌ | ✅ |
| 人工审批界面 | ❌ | ✅ |
| 报告可视化 | ❌ | ✅ |
| 状态监控 | 基础 | 丰富 |