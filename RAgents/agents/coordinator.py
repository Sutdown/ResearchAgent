from RAgents.llms.base import BaseLLM
from RAgents.prompts.loader import PromptLoader


class Coordinator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_loader = PromptLoader()

    # 根据提问进行分类，决定下一步的动作，返回提问类型
    def classify_query(self, user_query: str) -> str:
        prompt = self.prompt_loader.load(
            'coordinator_classify_query',
            user_query=user_query
        )

        query_type = self.llm.generate(prompt).strip().upper()
        if query_type not in ['GREETING', 'INAPPROPRIATE', 'RESEARCH']:
            query_type = 'RESEARCH'

        return query_type