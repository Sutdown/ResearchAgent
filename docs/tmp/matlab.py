# 安装：pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 代码示例：Agent 性能监控图表
print("=== Matplotlib 可视化示例 ===\n")

# 模拟数据
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sessions = [120, 150, 180, 160, 200, 90, 85]
response_time = [1.2, 1.1, 0.9, 1.0, 0.8, 1.5, 1.6]
success_rate = [0.85, 0.88, 0.92, 0.90, 0.94, 0.82, 0.80]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('AI Agent Performance Dashboard', fontsize=16, fontweight='bold')

# 1. 会话数柱状图
axes[0, 0].bar(days, sessions, color='skyblue', edgecolor='navy')
axes[0, 0].set_title('Daily Sessions')
axes[0, 0].set_ylabel('Sessions')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. 响应时间折线图
axes[0, 1].plot(days, response_time, marker='o', color='coral', linewidth=2)
axes[0, 1].set_title('Response Time')
axes[0, 1].set_ylabel('Time (seconds)')
axes[0, 1].grid(True, alpha=0.3)

# 3. 成功率区域图
axes[1, 0].fill_between(range(len(days)), success_rate, alpha=0.3, color='green')
axes[1, 0].plot(days, success_rate, marker='s', color='darkgreen', linewidth=2)
axes[1, 0].set_title('Success Rate')
axes[1, 0].set_ylabel('Rate')
axes[1, 0].set_ylim([0.7, 1.0])
axes[1, 0].grid(True, alpha=0.3)

# 4. 综合散点图
x = sessions
y = success_rate
sizes = [t * 200 for t in response_time]
axes[1, 1].scatter(x, y, s=sizes, alpha=0.5, c=range(len(days)), cmap='viridis')
axes[1, 1].set_title('Sessions vs Success (size=response_time)')
axes[1, 1].set_xlabel('Sessions')
axes[1, 1].set_ylabel('Success Rate')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('agent_dashboard.png', dpi=150, bbox_inches='tight')
print("✅ 图表已保存为 agent_dashboard.png")

print("\n=== Matplotlib 常用图表类型 ===")
print("""
1. 线图: plt.plot()
2. 柱状图: plt.bar()
3. 散点图: plt.scatter()
4. 饼图: plt.pie()
5. 直方图: plt.hist()
6. 箱线图: plt.boxplot()
7. 热力图: plt.imshow()
8. 子图: plt.subplots()
""")