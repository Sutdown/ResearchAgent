# ResearchAgent

## 项目结构

```
ResearchAgent/
├── main.py                 # 主程序入口
├── README.md               # 项目说明文档
├── .gitignore             # Git忽略文件配置
├── docs/                  # 文档目录
├── images/                # 图片资源目录
└── RAgents/               # 核心代码模块
    ├── agents/            # 智能代理模块
    │   ├── coordinator.py # 协调代理
    │   ├── planner.py     # 规划代理
    │   ├── rapporteur.py  # 报告代理
    │   └── researcher.py  # 研究代理
    ├── langsmith/         # LangSmith集成
    │   └── langsmith.py   # LangSmith相关功能
    ├── llms/              # 语言模型模块
    ├── prompts/           # 提示词模板
    ├── tools/             # 工具模块
    ├── utils/             # 工具函数
    └── workflow/          # 工作流模块
        ├── graph.py       # 工作流图定义
        ├── nodes.py       # 工作流节点
        └── state.py       # 状态管理
```

## 运行说明

```
python main.py
```