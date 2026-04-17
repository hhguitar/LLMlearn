from __future__ import annotations

from openai import OpenAI

from .settings import settings


class LLMClient:
    def __init__(self) -> None:
        api_key = settings.resolved_llm_api_key
        base_url = settings.resolved_llm_base_url
        if not api_key:
            raise ValueError('Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY in environment variables.')
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = settings.llm_model

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
            )
            return response.choices[0].message.content or ''
        except Exception as exc:
            return f'[LLM unavailable] {type(exc).__name__}: {exc}'
