### ResearchAgent

#### 设计理念



![1](..\images\1.png)



#### WorkFlow

> 基于LangGraph的工作流引擎，协调各代理之间的交互。



#### Agents

> 实现四个核心agent。

- coordinator.py：协调器。理解用户意图、统筹工作流。

- planner.py：规划器，拆解任务、制定研究计划。

- researcher.py：研究员，调用搜索、论文等工具收集信息。

- rapporteur.py：报告员，生成 Markdown/HTML 报告。



#### Vector

> 向量内存系统，用于持久化存储检索，主要用于优化researcher过程。



#### Tools

> 提供各种搜索和信息收集工具。



#### llms

> 语言模型抽象层，提供了统一的接口。

采用抽象工厂模式设计。

- **基础抽象类（BaseLLM）**：定义所有LLM实现的通用接口
- **具体实现类**：针对不同LLM提供商的具体实现
- **工厂类（LLMFactory）**：负责创建和管理不同LLM实例



#### Prompts

> 统一管理所有agent的提示词模板。



#### LangSmith

> LangSmith监控和追踪集成。



- [ ] 利用DSpy优化提示词操作





