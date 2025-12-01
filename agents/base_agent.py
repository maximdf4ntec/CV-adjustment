from abc import ABC, abstractmethod
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from agents.agents_utils import make_llm, _json_from_llm
from models import PipelineState  # Import your state definition


class BaseAgent(ABC):
    """
    An abstract base class (interface) for all agents in the pipeline.
    """

    def __init__(
        self, model: str, prompt_template: ChatPromptTemplate, debug: bool = False
    ):
        # Common functionality can go here
        self.llm = make_llm(model)
        self.prompt_template = prompt_template
        self.debug = debug
        print(f"Initializing {self.__class__.__name__}")

    @abstractmethod
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """
        The method that processes the state and returns updates.
        Must be implemented by all derived classes.
        """
        pass

    def _invoke_llm(
        self, dict: Dict[str, str], node: str, response_type: type[BaseModel]
    ) -> BaseModel:
        msg = self.prompt_template.invoke(dict)
        structured_llm = self.llm.with_structured_output(response_type)
        result = structured_llm.invoke(msg.to_messages())
        if result is None or result == "":
            print(
                f"DEBUG ERROR: LLM returned empty data for node: {node} using dict: {dict}"
            )
            raise ValueError("LLM returned empty data")
        else:
            if self.debug:
                print(f"DEBUG {node} LLM output: {result}")
        return result


def _invoke_llm_json(self, dict: Dict[str, str], node: str) -> Dict[str, Any]:
    msg = self.prompt_template.invoke(dict)
    content = self.llm.invoke(msg.to_messages()).content.strip()
    if content is None or content == "":
        print(
            f"DEBUG ERROR: LLM returned empty content for node: {node} using dict: {dict}"
        )
        raise ValueError("LLM returned empty content")
    else:
        if self.debug:
            print(f"DEBUG {node} LLM output: {content}")
        try:
            data = _json_from_llm(content)
        except Exception as e:
            print(
                f"DEBUG ERROR: Failed to parse JSON from LLM output for node: {node} using dict: {dict} LLM output was: {content}"
            )
            raise e
    return data
