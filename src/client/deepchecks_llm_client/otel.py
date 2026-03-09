import abc
import json
import logging
import typing as t
from datetime import datetime
from pathlib import Path

from deepchecks_llm_client.client import DeepchecksLLMClient
from deepchecks_llm_client.data_types import EnvType

logging.basicConfig()
logger = logging.getLogger(__name__)

try:
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter, SpanExportResult

    from deepchecks_llm_client.isolated_tracer import apply_instrumentor_isolation, enable_threading_propagation  # pylint: disable=ungrouped-imports
except ImportError:
    logger.error("OpenTelemetry dependencies are not installed. please install deepchecks-llm-client with otel extra - "
                 "\"pip install deepchecks-llm-client[otel]\"")
    raise


class BaseExporter(SpanExporter, abc.ABC):
    """Abstract base class for span exporters that adds agent framework information to spans"""

    def __init__(self, agent_framework_name: str = None):
        """
        Initialize the base exporter with agent framework name.

        Args:
            agent_framework_name: The agent framework name (e.g., crewai, langgraph).
        """
        self._agent_framework_name = agent_framework_name

    def _add_agent_framework_to_spans(self, spans: t.Sequence[ReadableSpan]) -> None:
        """
        Protected method to add agent framework name to span attributes.

        Args:
            spans: The list of ReadableSpan objects to modify
        """
        if self._agent_framework_name:
            for span in spans:
                if hasattr(span, '_attributes') and span._attributes is not None: # pylint: disable=protected-access
                    span._attributes["deepchecks.agent.framework"] = self._agent_framework_name # pylint: disable=protected-access

    @abc.abstractmethod
    def export(self, spans: t.Sequence[ReadableSpan]) -> "SpanExportResult":
        """Abstract method for exporting spans - must be implemented by subclasses"""
        pass

    def shutdown(self) -> None:
        """Shuts down the exporter.

        Called when the SDK is shut down.
        """
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool: # pylint: disable=unused-argument
        """Nothing is buffered in this exporter, so this method does nothing."""
        return True


class DCSpanExporter(BaseExporter):
    def __init__(
            self,
            dc_client: DeepchecksLLMClient,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            agent_framework_name: str = None,
    ):
        """
        Initializes the DCSpanExporter with the given host and API key.

        Args:
            dc_client: An instance of DeepchecksLLMClient.
            app_name: The name of the application.
            version_name: The name of the application version.
            env_type: The environment type (e.g., PROD, EVAL).
            agent_framework_name: The agent framework name (e.g., crewai, langgraph).
             If not provided, a new instance will be created.
        """
        super().__init__(agent_framework_name)
        self._dc_client = dc_client
        self._app_name = app_name
        self._version_name = version_name
        self._env_type = env_type

    def export(
        self, spans: t.Sequence[ReadableSpan]
    ) -> "SpanExportResult":
        """Exports a batch of telemetry data.

        Args:
            spans: The list of ReadableSpan objects to be exported

        Returns:
            The result of the export
        """
        try:
            # Add agent framework name to span attributes if provided
            self._add_agent_framework_to_spans(spans)

            self._dc_client._api.send_spans( # pylint: disable=protected-access
                app_name=self._app_name,
                version_name=self._version_name,
                env_type=self._env_type,
                spans=spans,
            )
            return SpanExportResult.SUCCESS
        except Exception as e: # pylint: disable=broad-except
            # Handle export failure
            logger.error(f"Export failed: {e}")
            return SpanExportResult.FAILURE


class JsonExporter(BaseExporter):
    def __init__(self, output_folder: str, agent_framework_name: str = None):
        """
        Initializes the JsonExporter with the output folder where spans will be saved.

        Args:
            output_folder: The folder where the spans will be dumped in JSON format.
            agent_framework_name: The agent framework name (e.g., crewai, langgraph).
        """
        super().__init__(agent_framework_name)
        self.output_folder = Path(output_folder)
        if not self.output_folder.exists():
            logger.info(f"Creating output folder: {self.output_folder}")
            self.output_folder.mkdir(parents=True, exist_ok=True)
        self.json_file = self.output_folder / f'spans{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    def export(
        self, spans: t.Sequence[ReadableSpan],
    ) -> "SpanExportResult":
        """Exports a batch of telemetry data by appending to a JSON file as a list of dicts."""
        try:
            # Add agent framework name to span attributes if provided
            self._add_agent_framework_to_spans(spans)

            # Load existing data if file exists, else start with empty list
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            logger.error(f"Existing JSON file {self.json_file} does not contain a list. Aborting export.")
                            return SpanExportResult.FAILURE
                    except json.JSONDecodeError:
                        logger.error(f"Existing JSON file {self.json_file} is not valid JSON. Aborting export.")
                        return SpanExportResult.FAILURE
            else:
                data = []

            # Append new spans
            data.extend([json.loads(span.to_json()) for span in spans])

            # Write back to file
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return SpanExportResult.SUCCESS
        except Exception as e: # pylint: disable=broad-except
            # Handle export failure
            logger.error(f"Export failed: {e}")
            return SpanExportResult.FAILURE


class OtelIntegration:
    """Abstract base class for OpenTelemetry integrations"""

    def __init__(self, agent_framework_name: str, instrumentors: t.List[BaseInstrumentor]):
        """
        Initialize the integration with framework name and instrumentor classes.

        Args:
            agent_framework_name: The name of the agent framework (e.g., "crewai", "langgraph")
            instrumentors: List of instrumentor objects to use for this framework
        """
        self.agent_framework_name = agent_framework_name
        self.instrumentors = instrumentors

    def register_dc_exporter(
        self,
        host: str,
        api_key: str,
        app_name: str,
        version_name: str,
        env_type: t.Union[EnvType, str],
        dc_client: t.Optional[DeepchecksLLMClient] = None,
        log_to_console: bool = False,
        isolated: bool = True,
    ) -> TracerProvider:
        """
        Registers the OpenTelemetry tracer with the provided host and API key.
        This function sets up the tracer provider with a console exporter for debugging
        and a DCSpanExporter for sending spans to Deepchecks LLM.
        Parameters
        ----------
        host : str
            The host URL for the Deepchecks LLM API.
        api_key : str
            The API key for authenticating with the Deepchecks LLM API.
        app_name : str
            The name of the application for which spans are being sent.
        version_name : str
            The name of the application version for which spans are being sent.
        env_type : EnvType | str
            The environment type (e.g., PROD, EVAL) for which spans are being sent.
        dc_client : DeepchecksLLMClient, optional
            An instance of DeepchecksLLMClient to use for sending spans.
            If not provided, a new instance will be created using the host and api_key.
        log_to_console : bool, optional
            If True, a console exporter will be added for debugging purposes.
            Defaults to False.
        isolated : bool, optional
            If True, enables context isolation to prevent trace mixing with other
            OTEL instrumentation (e.g., Langfuse, Traceloop). Defaults to True.
        """
        # create a DeepchecksLLMClient instance if not provided:
        dc_client = dc_client or DeepchecksLLMClient(
            host=host,
            api_token=api_key,
        )
        dc_span_exporter = DCSpanExporter(dc_client=dc_client, app_name=app_name,
                                           version_name=version_name, env_type=env_type,
                                           agent_framework_name=self.agent_framework_name)

        tracer_provider = self._add_exporter(
            dc_span_exporter, log_to_console=log_to_console, isolated=isolated
        )
        # Return the tracer provider for further use
        return tracer_provider

    def register_json_exporter(
        self,
        output_folder: str,
        log_to_console: bool = False,
        isolated: bool = True,
    ) -> TracerProvider:
        """
        Registers an OpenTelemetry exporter to dump spans to a JSON file.
        ----------
        output_folder : str
            The folder where the spans will be dumped in JSON format.
        log_to_console : bool, optional
            If True, a console exporter will be added for debugging purposes.
            Defaults to False.
        isolated : bool, optional
            If True, enables context isolation to prevent trace mixing with other
            OTEL instrumentation (e.g., Langfuse, Traceloop). Defaults to True.
        """
        # JsonExporter for dumping spans to JSON:
        dc_span_exporter = JsonExporter(output_folder=output_folder, agent_framework_name=self.agent_framework_name)
        tracer_provider = self._add_exporter(
            dc_span_exporter, log_to_console=log_to_console, isolated=isolated
        )
        # Return the tracer provider for further use
        return tracer_provider

    def uninstrument(self) -> None:
        """Uninstrument all instrumentors in this integration"""
        for instrumentor in self.instrumentors:
            instrumentor.uninstrument()

    def _add_exporter(
        self,
        span_exporter: SpanExporter,
        log_to_console: bool = False,
        isolated: bool = True,
    ) -> TracerProvider:
        tracer_provider = TracerProvider()

        # Enable threading context propagation unconditionally.
        # Uses copy_context() to propagate all ContextVars across thread boundaries.
        enable_threading_propagation()

        # Instrument with all the instrumentor classes for this framework
        for instrumentor in self.instrumentors:
            instrumentor.instrument(tracer_provider=tracer_provider)

            # Apply context isolation
            if isolated:
                apply_instrumentor_isolation(instrumentor)

        # console exporter for debugging if needed:
        if log_to_console:
            logger.info("Adding console exporter for debugging")
            # Add a console exporter to the tracer provider for debugging
            console_exporter = ConsoleSpanExporter()
            console_span_processor = SimpleSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_span_processor)

        # add the provided exporter:
        span_processor = BatchSpanProcessor(span_exporter)
        tracer_provider.add_span_processor(span_processor)
        # Return the tracer provider for further use
        return tracer_provider


class CrewaiIntegration(OtelIntegration):
    """CrewAI integration for OpenTelemetry"""

    def __init__(self):
        """Initialize CrewAI integration with CrewAI and LiteLLM instrumentors"""
        from openinference.instrumentation.crewai import CrewAIInstrumentor  # pylint: disable=import-outside-toplevel
        from openinference.instrumentation.litellm import LiteLLMInstrumentor  # pylint: disable=import-outside-toplevel
        try:
            # make sure crewai use LiteLLM and not native providers
            from crewai.llm import SUPPORTED_NATIVE_PROVIDERS  # pylint: disable=import-outside-toplevel
            SUPPORTED_NATIVE_PROVIDERS.clear()
        except ImportError:
            pass
        super().__init__(
            agent_framework_name="crewai",
            instrumentors=[CrewAIInstrumentor(), LiteLLMInstrumentor()]
        )


class LanggraphIntegration(OtelIntegration):
    """LangGraph integration for OpenTelemetry"""

    def __init__(self):
        """Initialize LangGraph integration with LangChain instrumentor"""
        from openinference.instrumentation.langchain import LangChainInstrumentor  # pylint: disable=import-outside-toplevel
        super().__init__(
            agent_framework_name="langgraph",
            instrumentors=[LangChainInstrumentor()]
        )


class GoogleAdkIntegration(OtelIntegration):
    """GoogleAdk integration for OpenTelemetry"""

    def __init__(self):
        """Initialize GoogleAdk integration with GoogleAdk instrumentor"""
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor  # pylint: disable=import-outside-toplevel
        super().__init__(
            agent_framework_name="google_adk",
            instrumentors=[GoogleADKInstrumentor()]
        )

