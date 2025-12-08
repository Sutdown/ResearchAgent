from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template

class PromptLoader:
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        self.prompts_dir = Path(prompts_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )

    # 加载并渲染提示
    def load(self, prompt_name: str, **variables: Any) -> str:
        if 'CURRENT_TIME' not in variables:
            variables['CURRENT_TIME'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            template = self.env.get_template(f"{prompt_name}.md")
            rendered = template.render(**variables)
            return rendered
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load prompt '{prompt_name}' from {self.prompts_dir}: {e}"
            )

    # 加载原始提示
    def load_raw(self, prompt_name: str) -> str:
        prompt_path = self.prompts_dir / f"{prompt_name}.md"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}"
            )

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    # 渲染字符串
    def render_string(self, template_str: str, **variables: Any) -> str:
        if 'CURRENT_TIME' not in variables:
            variables['CURRENT_TIME'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        template = Template(template_str)
        return template.render(**variables)

_default_loader = None

def get_default_loader() -> PromptLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader