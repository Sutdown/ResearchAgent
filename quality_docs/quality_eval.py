import os
import json
import glob

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pathlib import Path

load_dotenv()

# 配置DeepSeek API
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def load_prompt(prompt_file):
    """读取评估提示词"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def load_article(file_path):
    """读取文章内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def evaluate_article(prompt, article_content):
    """调用DeepSeek API评估文章质量"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"请评估以下文章：\n\n{article_content}"}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"评估过程中出错: {e}")
        return None

def parse_evaluation_result(eval_text):
    """解析评估结果为JSON格式"""
    try:
        # 尝试直接解析JSON
        result = json.loads(eval_text)
        return result
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试提取JSON部分
        try:
            start_idx = eval_text.find('{')
            end_idx = eval_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = eval_text[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            print(f"解析评估结果失败: {e}")
            print(f"原始结果: {eval_text}")
            return None

def evaluate_all_articles(outputs_dir, prompt_file):
    """评估outputs目录中的所有文章"""
    # 读取提示词
    prompt = load_prompt(prompt_file)

    # 获取所有md文件
    md_files = glob.glob(os.path.join(outputs_dir, "*.md"))

    results = []

    for md_file in sorted(md_files):
        file_name = os.path.basename(md_file)
        print(f"\n正在评估: {file_name}")

        # 读取文章
        article_content = load_article(md_file)

        # 调用API评估
        eval_result = evaluate_article(prompt, article_content)

        if eval_result:
            # 解析结果
            parsed_result = parse_evaluation_result(eval_result)

            if parsed_result:
                result = {
                    '文件名': file_name,
                    '内容专业性与深度(40分)': parsed_result['content_professionalism']['score'],
                    '专业性理由': parsed_result['content_professionalism']['reason'],
                    '逻辑结构与连贯性(30分)': parsed_result['logical_structure']['score'],
                    '逻辑性理由': parsed_result['logical_structure']['reason'],
                    '信息准确性与可信度(30分)': parsed_result['information_accuracy']['score'],
                    '准确性理由': parsed_result['information_accuracy']['reason'],
                    '总分': parsed_result['total_score']
                }
                results.append(result)
                print(f"评估完成，总分: {result['总分']}")
            else:
                print(f"解析评估结果失败: {file_name}")
        else:
            print(f"评估失败: {file_name}")

    return results

def save_results_to_table(results, output_file):
    """将结果保存为表格"""
    if not results:
        print("没有评估结果可保存")
        return

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 保存为Excel文件
    excel_file = output_file.replace('.csv', '.xlsx')
    df.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"\n结果已保存到: {excel_file}")

    # 同时保存为CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {output_file}")

def main():
    # 设置路径
    project_root = Path(__file__).parent.parent
    outputs_dir = os.path.join(project_root, "outputs")
    prompt_file = os.path.join(project_root, "quality_docs", "prompt.txt")
    output_file = os.path.join(project_root, "quality_docs", "evaluation_results.csv")

    print("=" * 50)
    print("文章质量评估系统")
    print("=" * 50)

    # 开始评估
    results = evaluate_all_articles(outputs_dir, prompt_file)

    # 保存结果
    if results:
        save_results_to_table(results, output_file)

        # 打印汇总
        print("\n" + "=" * 50)
        print("评估汇总")
        print("=" * 50)
        for result in results:
            print(f"\n文件: {result['文件名']}")
            print(f"  内容专业性与深度: {result['内容专业性与深度(40分)']}/40")
            print(f"  逻辑结构与连贯性: {result['逻辑结构与连贯性(30分)']}/30")
            print(f"  信息准确性与可信度: {result['信息准确性与可信度(30分)']}/30")
            print(f"  总分: {result['总分']}/100")
    else:
        print("没有完成任何评估")

if __name__ == "__main__":
    main()
