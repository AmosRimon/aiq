# pylint: disable=unused-argument
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
except ImportError:
    # pylint: disable=ungrouped-imports
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import LLMResult
    from langchain.schema.document import Document
    from langchain_core.messages import BaseMessage
from pytz import UTC

from deepchecks_llm_client.client import DEFAULT_ENV_TYPE, DeepchecksLLMClient
from deepchecks_llm_client.data_types import EnvType, LogInteraction, Step
from deepchecks_llm_client.exceptions import DeepchecksLLMClientError


class DeepchecksCallbackHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    def __init__(self, *args, app_name, app_version, env_type: EnvType = DEFAULT_ENV_TYPE,
                 host: str = None, api_key: str = None, logger: logging.Logger = None,
                 user_interaction_id: Union[str, None] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._started_at = None
        self.api_key = api_key
        self.host = host
        self.app_name = app_name
        self.app_version = app_version
        self.env_type = env_type
        self.logger = logger or logging.getLogger(__name__)
        self._user_interaction_id = user_interaction_id
        self._information_retrieval = None
        self._input = None
        self._output = None
        self._steps: Dict[str, Step] = defaultdict()  # run_id -> Step

    @property
    def steps(self) -> List[Step]:
        return list(self._steps.values())

    @property
    def input(self) -> str:
        return self._input

    @property
    def output(self) -> str:
        return self._output

    @property
    def user_interaction_id(self) -> Union[str, None]:
        return str(self._user_interaction_id)

    @property
    def started_at(self) -> Union[datetime, None]:
        return self._started_at

    @property
    def information_retrieval(self) -> list[str]:
        return self._information_retrieval

    def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        """Run when ChatModel starts running."""
        self.logger.debug('on_chat_model_start', kwargs)

        _input = ""
        for msg in messages:
            _input += '\n'.join([m.content for m in msg])

        if parent_run_id is None:
            self._input = _input
            self._started_at = datetime.now(tz=UTC)

        self._steps[str(run_id)] = Step(
            name="LLM",
            value=_input,
        )

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.logger.debug('on_llm_start', kwargs)
        run_id = kwargs['run_id']
        _input = '\n'.join(prompts)
        if kwargs.get('parent_run_id', None) is None:
            self._input = _input
            self._started_at = datetime.now(tz=UTC)

        self._steps[str(run_id)] = Step(
            name="LLM",
            value=_input,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.logger.debug('on_llm_end', kwargs)
        step = self._steps[str(kwargs['run_id'])]
        step.value = '\n'.join([a.json() for b in response.generations for a in b])

        if kwargs.get('parent_run_id', None) is None:
            self._output = '\n'.join(generation.text for generations in response.generations for generation in generations)
            self._log_interaction()

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        self.logger.debug('on_llm_error', kwargs)
        self._on_error(error, kwargs)

    def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        if parent_run_id is None:
            self._started_at = datetime.now(tz=UTC)
            self.logger.debug('on_chain_start', kwargs)
            self._input = inputs.get('input') or inputs.get('question') or inputs.get('text') or '\n'.join([str(v) for v in inputs.values()]) \
                if isinstance(inputs, dict) \
                else inputs

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        self.logger.debug('on_chain_end', kwargs)
        if kwargs.get('parent_run_id', None) is None:
            self._output = outputs.get('text') or outputs.get('output_text') or '\n'.join([str(v) for v in outputs.values()]) \
                if isinstance(outputs, dict) \
                else outputs

            self._log_interaction()

    def on_retriever_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """Run when Retriever errors."""
        self.logger.debug('on_retriever_error', kwargs)
        self._on_error(error, kwargs)

    def _on_error(self, error, kwargs):
        self._log_interaction()

    def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        """Run when Retriever starts running."""
        self.logger.debug('on_retriever_start', kwargs)
        self._information_retrieval = []

    def on_retriever_end(
            self,
            documents: Sequence[Document],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        self.logger.debug('on_retriever_end', kwargs)
        self._information_retrieval.append('\n'.join(d.page_content for d in documents))

    def _log_interaction(self):

        host = self.host or os.getenv('DEEPCHECKS_LLM_HOST', None)
        api_token = self.api_key or os.getenv('DEEPCHECKS_LLM_API_KEY', None)
        if not api_token or not host:
            raise DeepchecksLLMClientError('API key or host not defined. Please either pass them as arguments or'
                                           ' set them as environment variables.')

        _client: DeepchecksLLMClient = DeepchecksLLMClient(host=host,
                                                           api_token=api_token,
                                                           silent_mode=True)

        result = _client.log_interaction(
            app_name=self.app_name,
            version_name=self.app_version,
            env_type=self.env_type,
            interaction=LogInteraction(
                output=self.output,
                user_interaction_id=self.user_interaction_id,
                input=self.input,
                steps=self.steps,
                information_retrieval=self.information_retrieval,
                history=None,
                started_at=self.started_at.timestamp() if self.started_at else datetime.now(tz=UTC).timestamp(),
                finished_at=datetime.now(tz=UTC).timestamp()
            )
        )
        if not result:
            raise self.logger.error('Interaction logging failed')

    @staticmethod
    def to_str_dict(attributes):
        return {k: str(v) for k, v in attributes.items()}
