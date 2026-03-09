import enum
import json
import logging
import typing as t
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pytz import UTC

from deepchecks_llm_client.exceptions import DeepchecksLLMClientError
from deepchecks_llm_client.utils import check_topic

__all__ = ["EnvType", "AnnotationType", "Interaction", "Step", "Application",
           "ApplicationType", "ApplicationVersion", "ApplicationVersionSchema",
           "LogInteraction", "InteractionType", "BuiltInInteractionType", "UserValueProperty",
           "PropertyColumnType", "UserValuePropertyType", "InteractionCompleteEvents",
           "InteractionTypeVersionData", "CreateInteractionTypeVersionData", "UpdateInteractionTypeVersionData",
           "Span", "SpanKind", "SpanEvent", "Dataset", "DatasetSample", "DatasetType", "Deployment",
           "DeploymentHeader", "DatasetRunResult", "DatasetRunSampleResult"]

logging.basicConfig()
logger = logging.getLogger(__name__)


def _timestamp_to_iso_format(timestamp: float) -> str:
    """Convert a Unix timestamp to ISO 8601 format with Z suffix.

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        ISO 8601 formatted timestamp string with Z suffix (e.g., '2024-01-18T10:30:00.000Z')
    """
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace('+00:00', 'Z')


class EnvType(str, enum.Enum):
    PROD = "PROD"
    EVAL = "EVAL"
    PENTEST = "PENTEST"


class AnnotationType(str, enum.Enum):
    GOOD = "good"
    BAD = "bad"
    UNKNOWN = "unknown"


class PropertyColumnType(str, enum.Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


@dataclass
class Step:
    name: str
    value: str

    def to_json(self):
        return {
            self.name: self.value
        }

    @classmethod
    def as_jsonl(cls, steps):
        if steps is None:
            return None
        return [step.to_json() for step in steps]


@dataclass
class UserValueProperty:
    """Data class representing user provided property"""
    name: str
    value: t.Any
    reason: t.Optional[str] = None


@dataclass
class Interaction:
    user_interaction_id: t.Union[str, int]
    input: str
    output: str
    information_retrieval: t.Union[str, t.List[str]]
    history: t.Union[str, t.List[str]]
    full_prompt: str
    expected_output: str
    is_completed: bool
    metadata: t.Dict[str, str]
    tokens: int
    input_tokens: t.Optional[int]
    output_tokens: t.Optional[int]
    model: t.Optional[str]
    model_provider: t.Optional[str]
    input_cost: t.Optional[float]
    output_cost: t.Optional[float]
    cost: t.Optional[float]
    properties: t.Dict[str, t.Any]
    properties_reasons: t.Dict[str, t.Any]
    created_at: datetime
    interaction_datetime: datetime
    interaction_type: str
    topic: str
    session_id: t.Union[str, int]
    annotation: t.Optional[AnnotationType]
    annotation_reason: t.Optional[str]


@dataclass
class InteractionUpdate:
    """A dataclass representing an update interaction object.

        Attributes
        ----------
        input : str
            Input data
        output : str
            Output data
        expected_output : str, optional
            Full expected output data, defaults to None
        full_prompt : str, optional
            Full prompt data, defaults to None
        information_retrieval : str, optional
            Information retrieval, defaults to None
        history : str, optional
            History (for instance "chat history"), defaults to None
        annotation : AnnotationType, optional
            Annotation type of the interaction, defaults to None
        steps : list of Step, optional
            List of steps taken during the interaction, defaults to None
        user_value_properties : list of UserValueProperty, optional
            Additional user value properties, defaults to None
        annotation_reason : str, optional
            Reason for the annotation, defaults to None
        started_at : datetime or float, optional
            Timestamp the interaction started at. Datetime format is deprecated, use timestamp instead
        finished_at : datetime or float, optional
            Timestamp the interaction finished at. Datetime format is deprecated, use timestamp instead
        is_completed : bool, optional
            Indicates if the interaction is completed, defaults to True
        metadata : dict, optional
            Metadata for the interaction, defaults to None
        tokens : int, optional
            Token count for the interaction, defaults to None
        input_tokens : int, optional
            Input token count for the interaction, defaults to None
        output_tokens : int, optional
            Output token count for the interaction, defaults to None
        model : str, optional
            The model name used for this interaction (e.g., 'gpt-4'), used for cost calculation, defaults to None
        model_provider : str, optional
            The model provider used for this interaction (e.g., 'openai'), used for cost calculation, defaults to None
        """
    input: t.Optional[str] = None
    output: t.Optional[str] = None
    information_retrieval: t.Optional[t.Union[str, t.List[str]]] = None
    history: t.Optional[t.Union[str, t.List[str]]] = None
    full_prompt: t.Optional[str] = None
    expected_output: t.Optional[str] = None
    is_completed: bool = True
    metadata: t.Optional[t.Dict[str, str]] = None
    tokens: t.Optional[int] = None
    input_tokens: t.Optional[int] = None
    output_tokens: t.Optional[int] = None
    model: t.Optional[str] = None
    model_provider: t.Optional[str] = None
    annotation: t.Optional[t.Union[AnnotationType, str]] = None
    annotation_reason: t.Optional[str] = None
    steps: t.Optional[t.List[Step]] = None
    user_value_properties: t.Optional[t.List[UserValueProperty]] = None
    started_at: t.Optional[t.Union[datetime, float]] = None
    finished_at: t.Optional[t.Union[datetime, float]] = None

    def to_json(self):
        if isinstance(self.started_at, datetime) or isinstance(self.finished_at, datetime):
            logger.warning(
                "Deprecation Warning: Usage of datetime for started_at/finished_at is deprecated, use timestamp instead."
            )
            self.started_at = (self.started_at.timestamp() if isinstance(self.started_at, datetime) else self.started_at) \
                if self.started_at else datetime.now(tz=UTC).timestamp()
            self.finished_at = (self.finished_at.timestamp() if isinstance(self.finished_at, datetime) else self.finished_at) \
                if self.finished_at else None

        data = {
            "input": self.input,
            "output": self.output,
            "expected_output": self.expected_output,
            "full_prompt": self.full_prompt,
            "information_retrieval": self.information_retrieval
            if self.information_retrieval is None or isinstance(self.information_retrieval, list)
            else [self.information_retrieval],
            "history": self.history
            if self.history is None or isinstance(self.history, list)
            else [self.history],
            "annotation": (
                None if self.annotation is None else
                self.annotation.value if isinstance(self.annotation, AnnotationType)
                else str(self.annotation).lower().strip()
            ),
            "steps": [step.to_json() for step in self.steps] if self.steps else None,
            "custom_properties": {prop.name: prop.value for prop in self.user_value_properties} if self.user_value_properties else None,
            "custom_properties_reasons": {
                prop.name: prop.reason for prop in self.user_value_properties if prop.reason
            } if self.user_value_properties else None,
            "annotation_reason": self.annotation_reason,
            "is_completed": self.is_completed,
            "metadata": self.metadata,
            "tokens": self.tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "model_provider": self.model_provider,
        }
        if self.started_at:
            data["started_at"] = self.started_at
        if self.finished_at:
            data["finished_at"] = self.finished_at

        return data


@dataclass
class LogInteraction(InteractionUpdate):
    """A dataclass representing a new interaction object.

    Attributes
    ----------
    input : str
        Input data
    output : str
        Output data
    expected_output : str, optional
        Full expected output data, defaults to None
    full_prompt : str, optional
        Full prompt data, defaults to None
    annotation : AnnotationType, optional
        Annotation type of the interaction, defaults to None
    user_interaction_id : str, optional
        Unique identifier of the interaction, defaults to None
    steps : list of Step, optional
        List of steps taken during the interaction, defaults to None
    user_value_properties : list of UserValueProperty, optional
        Additional user value properties, defaults to None
    information_retrieval : str, optional
        Information retrieval, defaults to None
    history : str, optional
        History (for instance "chat history"), defaults to None
    annotation_reason : str, optional
        Reason for the annotation, defaults to None
    started_at : datetime or float, optional
        Timestamp the interaction started at. Datetime format is deprecated, use timestamp instead
    finished_at : datetime or float, optional
        Timestamp the interaction finished at. Datetime format is deprecated, use timestamp instead
    vuln_type : str, optional
        Type of vulnerability (Only used in case of EnvType.PENTEST and must be sent there), defaults to None
    vuln_trigger_str : str, optional
        Vulnerability trigger string (Only used in case of EnvType.PENTEST and is optional there), defaults to None
    session_id: str, optional
        The identifier for the session associated with this interaction.
        If not provided, a session ID will be automatically generated.
    interaction_type: str, optional
        The type of interaction.
        None is deprecated. If not provided, the interaction type will default to the application's default type.
    metadata: t.Dict[str, str], optional
        Metdata for the interaction.
    tokens: int, optional
        Token count for the interaction.
    """
    user_interaction_id: t.Optional[t.Union[str, int]] = None
    vuln_type: t.Optional[str] = None
    vuln_trigger_str: t.Optional[str] = None
    topic: t.Optional[str] = None
    interaction_type: t.Optional[str] = None
    session_id: t.Optional[t.Union[str, int]] = None

    def to_json(self):
        data = super().to_json()
        if self.interaction_type is None:
            logger.warning(
                "Deprecation Warning: The value 'None' for 'interaction_type' is deprecated. "
                "Please specify an explicit interaction type."
            )

        # rename custom_properties to custom_props:
        data["custom_props"] = data.pop("custom_properties", None)
        data["custom_props_reasons"] = data.pop("custom_properties_reasons", None)

        data.update({
            "user_interaction_id": str(self.user_interaction_id) if self.user_interaction_id is not None else None,
            "vuln_type": self.vuln_type,
            "vuln_trigger_str": self.vuln_trigger_str,
            "session_id": str(self.session_id) if self.session_id is not None else None,
            "interaction_type": self.interaction_type,
        })
        check_topic(self.topic)
        if self.topic is not None:
            data["topic"] = self.topic

        return data


class SpanKind(str, enum.Enum):
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    AGENT = "AGENT"
    RETRIEVAL = "RETRIEVER"


@dataclass
class SpanEvent:
    """A dataclass representing an event that occurred during a span's execution.

    Attributes
    ----------
    name : str
        The name of the event
    timestamp : float
        The timestamp when the event occurred (Unix timestamp)
    attributes : dict, optional
        Additional attributes associated with the event
    """
    name: str
    timestamp: float
    attributes: t.Optional[t.Dict[str, t.Any]] = None


@dataclass
class Span:
    """A dataclass representing a span within a trace for tracking nested operations.

    A Span represents a unit of work within a distributed trace, allowing you to track
    hierarchical relationships between operations. This is designed to work with the
    OtelParser system for converting spans into interactions.

    Attributes
    ----------
    span_id : str
        The unique identifier for this span
    span_name : str
        The name of this span, describing the operation being tracked
    span_kind : SpanKind
        The type of span (Root is CHAIN without parent_id)
    trace_id : str
        The unique identifier for the trace this span belongs to. All spans
        in the same trace share this ID
    parent_id : str or None
        The ID of the parent span for tree structure. None if this is the root span
    started_at : float
        Timestamp when the span started (numeric value for ordering)
    finished_at : float
        Timestamp when the span finished (numeric value for ordering)
    status_code : {'OK', 'ERROR'}, optional
        The status code indicating whether the span completed successfully.
        'OK' indicates success, 'ERROR' indicates failure. Defaults to 'OK'
    status_description : str or None, optional
        Human-readable description providing additional context about the status,
        particularly useful for explaining error conditions. Defaults to None
    input : str, optional
        Input data for this span's operation, defaults to None
    output : str, optional
        Output data from this span's operation, defaults to None
    full_prompt : str, optional
        Full prompt data, defaults to None
    expected_output : str, optional
        Expected output data, defaults to None
    information_retrieval : str, optional
        Information retrieval data, defaults to None
    tokens : int, optional
        Token count for aggregation purposes, defaults to None
    input_tokens : int, optional
        Input token count for the span, defaults to None
    output_tokens : int, optional
        Output token count for the span, defaults to None
    model : str, optional
        The model name used for this span (e.g., 'gpt-4'), used for cost calculation, defaults to None
    model_provider : str, optional
        The model provider used for this span (e.g., 'openai'), used for cost calculation, defaults to None
    graph_parent_name : str, optional
        Graph metadata indicating "who triggered me" (the span_name of another span), defaults to None
    session_id : str, optional
        The identifier for the session associated with this span, defaults to None
    metadata : t.Dict[str, str], optional
        Additional metadata for the span, defaults to None
    events : list of SpanEvent, optional
        List of events that occurred during this span's execution, defaults to None
    user_value_properties : list of UserValueProperty, optional
        User-provided custom properties for the span, defaults to None
    """

    span_id: str
    span_name: str
    trace_id: str
    span_kind: SpanKind
    parent_id: t.Optional[str]
    started_at: float
    finished_at: float
    status_code: t.Literal['OK', 'ERROR'] = 'OK'
    status_description: t.Optional[str] = None
    input: t.Optional[str] = None
    output: t.Optional[str] = None
    full_prompt: t.Optional[str] = None
    expected_output: t.Optional[str] = None
    information_retrieval: t.Optional[t.List[str]] = None
    tokens: t.Optional[int] = None
    input_tokens: t.Optional[int] = None
    output_tokens: t.Optional[int] = None
    model: t.Optional[str] = None
    model_provider: t.Optional[str] = None
    graph_parent_name: t.Optional[str] = None
    session_id: t.Optional[str] = None
    metadata: t.Optional[t.Dict[str, str]] = None
    events: t.Optional[t.List[SpanEvent]] = None
    user_value_properties: t.Optional[t.List[UserValueProperty]] = None

    def to_span_data(self):
        attributes = {
            "deepchecks.agent.framework": "deepchecks_sdk",
            "openinference.span.kind": self.span_kind
        }

        # Text fields with .value suffix in attributes
        if self.input is not None:
            attributes["input.value"] = self.input
        if self.output is not None:
            attributes["output.value"] = self.output
        if self.full_prompt is not None:
            attributes["full_prompt.value"] = self.full_prompt
        if self.expected_output is not None:
            attributes["expected_output.value"] = self.expected_output
        if self.information_retrieval is not None:
            attributes["information_retrieval.value"] = self.information_retrieval

        # Token count (optional, for aggregation)
        # If only input_tokens and output_tokens are provided, calculate total
        if self.tokens is not None:
            attributes["llm.token_count.total"] = self.tokens
        elif self.input_tokens is not None and self.output_tokens is not None:
            attributes["llm.token_count.total"] = self.input_tokens + self.output_tokens
        if self.input_tokens is not None:
            attributes["llm.token_count.prompt"] = self.input_tokens
        if self.output_tokens is not None:
            attributes["llm.token_count.completion"] = self.output_tokens

        # Model name and provider (optional, for cost calculation)
        if self.model is not None:
            attributes["llm.model_name"] = self.model
        if self.model_provider is not None:
            attributes["llm.provider"] = self.model_provider

        # Graph parent name (who triggered me)
        if self.graph_parent_name is not None:
            attributes["graph.node.id"] = self.span_name
            attributes["graph.node.parent_id"] = self.graph_parent_name

        # Session ID (optional)
        if self.session_id is not None:
            attributes["session.id"] = self.session_id

        # Metadata (optional)
        if self.metadata is not None:
            if isinstance(self.metadata, dict):
                attributes["metadata"] = json.dumps(self.metadata)
            else:
                attributes["metadata"] = self.metadata

        # User value properties (optional) - stored as dc_user_values attribute
        # Serialize to JSON string to match metadata pattern
        if self.user_value_properties is not None:
            attributes["dc_user_values"] = json.dumps([
                {"name": prop.name, "value": prop.value, "reason": prop.reason}
                for prop in self.user_value_properties
            ])

        # Convert timestamps to ISO 8601 format with Z suffix (CrewAI format)
        start_time_iso = _timestamp_to_iso_format(self.started_at)
        end_time_iso = _timestamp_to_iso_format(self.finished_at)

        # Convert events to OTEL format
        events_data = []
        if self.events:
            for event in self.events:
                events_data.append({
                    "name": event.name,
                    "timestamp": _timestamp_to_iso_format(event.timestamp),
                    "attributes": event.attributes
                })

        # Build the span structure matching OTEL format
        span_data = {
            "name": self.span_name,
            "kind": "SpanKind.INTERNAL",
            "status": {"status_code": self.status_code, "description": self.status_description},
            "events": events_data,
            "context": {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "trace_state": "[]",
            },
            "parent_id": self.parent_id,
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "attributes": attributes,
        }

        return span_data


@dataclass
class UserValuePropertyType:
    display_name: str
    type: t.Union[PropertyColumnType, str]
    description: t.Union[str, None] = None


class BuiltInInteractionType(str, enum.Enum):
    QA = "Q&A"
    OTHER = "Other"
    SUMMARIZATION = "Summarization"
    CLASSIFICATION = "Classification"
    GENERATION = "Generation"
    FEATURE_EXTRACTION = "Feature Extraction"
    RETRIEVAL = "Retrieval"
    CHAT = "Chat"
    CHAIN = "Chain"
    ROOT = "Root"
    LLM = "LLM"
    AGENT = "Agent"
    TOOL = "Tool"


class ApplicationType(str, enum.Enum):
    QA = "Q&A"
    OTHER = "Other"
    SUMMARIZATION = "Summarization"
    CLASSIFICATION = "Classification"
    GENERATION = "Generation"
    FEATURE_EXTRACTION = "Feature Extraction"
    RETRIEVAL = "Retrieval"
    CHAT = "Chat"
    CHAIN = "Chain"
    ROOT = "Root"
    LLM = "LLM"
    AGENT = "Agent"
    TOOL = "Tool"


class DatasetType(str, enum.Enum):
    """Enum for dataset types."""

    SINGLE_TURN = "SINGLE_TURN"
    MULTI_TURN = "MULTI_TURN"


class InteractionCompleteEvents(str, enum.Enum):
    TOPICS_COMPLETED = "topics_completed"
    PROPERTIES_COMPLETED = "properties_completed"
    SIMILARITY_COMPLETED = "similarity_completed"
    LLM_PROPERTIES_COMPLETED = "llm_properties_completed"
    ANNOTATION_COMPLETED = "annotation_completed"
    DC_EVALUATION_COMPLETED = "dc_evaluation_completed"
    BUILTIN_LLM_PROPERTIES_COMPLETED = "builtin_llm_properties_completed"


@dataclass
class ApplicationVersionSchema:
    name: str
    description: t.Optional[str] = None
    additional_fields: t.Optional[t.Dict[str, t.Any]] = None

    def to_json(self):
        return {
            "name": self.name,
            "description": self.description,
            "additional_fields": self.additional_fields if self.additional_fields else {}
        }


@dataclass
class ApplicationVersion:
    """A dataclass representing an Application Version.

    Attributes
    ----------
    id : int
        Version id
    name : str
        Version name
    created_at : datetime
        Version created at timestamp
    updated_at : datetime
        Version updated at timestamp
    custom : list of dict
        Additional details about the version as key-value pairs
        This member is deprecated. It will be removed in future versions. Use additional_fields instead.
    additional_fields : dict
        Additional details about the version as dict
    """

    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    description: t.Optional[str] = None
    additional_fields: t.Optional[t.Dict[str, t.Any]] = None


@dataclass
class Application:
    id: int
    name: str
    kind: ApplicationType
    created_at: datetime
    updated_at: datetime
    in_progress: bool
    versions: t.List[ApplicationVersion]
    interaction_types: t.List[str]
    description: t.Optional[str] = None
    log_latest_insert_time_epoch: t.Optional[int] = None
    n_of_llm_properties: t.Optional[int] = None
    n_of_interactions: t.Optional[int] = None
    notifications_enabled: t.Optional[bool] = None


@dataclass
class InteractionTypeVersionData:
    """A dataclass representing interaction type version data.

    Attributes
    ----------
    id : int
        Interaction type version data id
    interaction_type_id : int
        Interaction type id
    application_version_id : int
        Application version id
    model : str or None
        Model name
    prompt : str or None
        Prompt template
    metadata_params : dict
        Additional metadata parameters
    created_at : datetime
        Created at timestamp
    updated_at : datetime
        Updated at timestamp
    """
    id: int
    interaction_type_id: int
    application_version_id: int
    model: t.Optional[str] = None
    prompt: t.Optional[str] = None
    metadata_params: t.Dict[str, t.Any] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class CreateInteractionTypeVersionData:
    """A dataclass for creating interaction type version data.

    Attributes
    ----------
    interaction_type_id : int
        Interaction type id
    application_version_id : int
        Application version id
    model : str or None
        Model name
    prompt : str or None
        Prompt template
    metadata_params : dict
        Additional metadata parameters
    """
    interaction_type_id: int
    application_version_id: int
    model: t.Optional[str] = None
    prompt: t.Optional[str] = None
    metadata_params: t.Dict[str, t.Any] = None

    def to_json(self):
        return {
            "interaction_type_id": self.interaction_type_id,
            "application_version_id": self.application_version_id,
            "model": self.model,
            "prompt": self.prompt,
            "metadata_params": self.metadata_params or {},
        }


@dataclass
class UpdateInteractionTypeVersionData:
    """A dataclass for updating interaction type version data.

    Attributes
    ----------
    model : str or None
        Model name
    prompt : str or None
        Prompt template
    metadata_params : dict or None
        Additional metadata parameters
    """
    model: t.Optional[str] = None
    prompt: t.Optional[str] = None
    metadata_params: t.Optional[t.Dict[str, t.Any]] = None

    def to_json(self):
        return {
            "model": self.model,
            "prompt": self.prompt,
            "metadata_params": self.metadata_params,
        }


@dataclass
class InteractionType:
    id: int
    name: str


@dataclass
class Dataset:
    """Data class representing a dataset."""
    id: int
    application_id: int
    dataset_name: str
    samples_count: int
    created_at: datetime
    updated_at: datetime
    dataset_type: DatasetType = DatasetType.SINGLE_TURN

    @classmethod
    def from_dict(cls, data: t.Dict[str, t.Any]) -> 'Dataset':
        return cls(
            id=data["id"],
            application_id=data["application_id"],
            dataset_name=data["dataset_name"],
            samples_count=data["samples_count"],
            created_at=pd.to_datetime(data["created_at"]),
            updated_at=pd.to_datetime(data["updated_at"]),
            dataset_type=DatasetType(data.get("dataset_type", "SINGLE_TURN")),
        )


@dataclass
class DatasetSample:
    """Data class representing a dataset sample."""
    id: int
    dataset_id: int
    input: t.Any
    output: t.Optional[t.Any]
    sample_metadata: t.Optional[t.Dict[str, t.Any]]
    created_at: t.Optional[datetime] = None
    updated_at: t.Optional[datetime] = None


@dataclass
class DeploymentHeader:
    """Data class representing a deployment header."""
    id: int
    deployment_id: int
    name: str
    value: str

    @classmethod
    def from_dict(cls, data: t.Dict[str, t.Any]) -> 'DeploymentHeader':
        return cls(
            id=data["id"],
            deployment_id=data["deployment_id"],
            name=data["name"],
            value=data["value"],
        )


@dataclass
class Deployment:
    """Data class representing a deployment configuration."""
    id: int
    application_id: int
    deployment_name: str
    deployment_url: str
    timeout: int
    max_concurrent: int
    max_retries: int
    created_at: datetime
    updated_at: datetime
    headers: t.List[DeploymentHeader]

    @classmethod
    def from_dict(cls, data: t.Dict[str, t.Any]) -> 'Deployment':
        return cls(
            id=data["id"],
            application_id=data["application_id"],
            deployment_name=data["deployment_name"],
            deployment_url=data["deployment_url"],
            timeout=data["timeout"],
            max_concurrent=data["max_concurrent"],
            max_retries=data["max_retries"],
            created_at=pd.to_datetime(data["created_at"]),
            updated_at=pd.to_datetime(data["updated_at"]),
            headers=[DeploymentHeader.from_dict(h) for h in data["headers"]],
        )


@dataclass
class DatasetRunSampleResult:
    """Result of running a single dataset sample."""
    sample_id: int
    sample_input: t.Any
    sample_output: t.Optional[t.Any]
    sample_metadata: t.Optional[t.Dict[str, t.Any]]
    success: bool
    deployment_response: t.Optional[t.Dict[str, t.Any]] = None
    error_message: t.Optional[str] = None
    error_type: t.Optional[str] = None
    duration_seconds: t.Optional[float] = None
    retries: int = 0
    status_code: t.Optional[int] = None


@dataclass
class DatasetRunResult:
    """Result of running a dataset on a deployment."""
    dataset_name: str
    deployment_name: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    duration_seconds: float
    results: t.List[DatasetRunSampleResult]
    # Optional fields for re-running
    app_name: t.Optional[str] = None
    version_name: t.Optional[str] = None
    env_type: t.Optional[str] = None
    additional_headers: t.Optional[t.Dict[str, str]] = None
    verify_ssl: bool = True
    _deployment: t.Optional['Deployment'] = None  # Deployment config for re-running
    _api: t.Any = None  # API client for re-running
    iteration: int = 1  # Number of iterations (1 = initial run, 2+ = after retries)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        return (self.successful_samples / self.total_samples * 100) if self.total_samples > 0 else 0.0

    @property
    def failed_results(self) -> t.List[DatasetRunSampleResult]:
        """Get only the failed sample results."""
        return [r for r in self.results if not r.success]

    @property
    def successful_results(self) -> t.List[DatasetRunSampleResult]:
        """Get only the successful sample results."""
        return [r for r in self.results if r.success]

    async def rerun_failed(self, show_progress: bool = True) -> None:
        """Re-run only the failed samples from this result and update in place.

        This is a convenience method that re-runs failed samples using the same
        configuration (app_name, deployment, version, etc.) from the original run,
        then merges the successful retries back into this result.

        Note: This is an async method and must be called with await:
        >>> await result.rerun_failed()

        Parameters
        ----------
        show_progress : bool, default=True
            Whether to show a live progress widget (if in notebook with ipywidgets installed)

        Examples
        --------
        >>> result = await client.execute_app("my-app", "test-dataset", "prod-deployment")
        >>> print(f"Initial: {result.successful_samples}/{result.total_samples} succeeded")
        >>> if result.failed_samples > 0:
        ...     await result.rerun_failed()  # Updates result in place
        ...     print(f"After retry: {result.successful_samples}/{result.total_samples} succeeded")
        >>> result  # Display result in notebook - shows interactive widget
        """
        if self.failed_samples == 0:
            raise DeepchecksLLMClientError("No failed samples to re-run")

        from deepchecks_llm_client.dataset_operations import rerun_failed_samples as rerun_impl  # pylint: disable=import-outside-toplevel

        retry_result = await rerun_impl(
            previous_result=self,
            show_progress=show_progress,
        )

        # Merge retry results back into this object
        # Create a mapping of sample_id to retry result
        retry_results_map = {r.sample_id: r for r in retry_result.results}

        # Update results list: replace failed samples with their retry results
        updated_results = []
        for original_result in self.results:
            if not original_result.success and original_result.sample_id in retry_results_map:
                # Use the retry result
                updated_results.append(retry_results_map[original_result.sample_id])
            else:
                # Keep original result
                updated_results.append(original_result)

        # Update the dataclass fields
        self.results = updated_results
        self.successful_samples = sum(1 for r in updated_results if r.success)
        self.failed_samples = sum(1 for r in updated_results if not r.success)
        self.duration_seconds += retry_result.duration_seconds
        self.iteration = retry_result.iteration

    def _repr_html_(self) -> str:
        """Display interactive results widget in Jupyter notebooks."""
        # import in function to avoid unnecessary module load
        from deepchecks_llm_client.notebook_widgets import create_interactive_results_widget  # pylint: disable=import-outside-toplevel
        widget = create_interactive_results_widget(self)
        if widget is not None:
            return ""  # Widget is displayed, return empty string

        # Fall back to text representation
        lines = [
            "<div style='font-family: monospace;'>",
            "<b>Dataset Run Result</b><br>",
            f"Dataset: {self.dataset_name}<br>",
            f"Deployment: {self.deployment_name}<br>",
            f"Total samples: {self.total_samples}<br>",
            f"Successful: {self.successful_samples}<br>",
            f"Failed: {self.failed_samples}<br>",
            f"Success rate: {self.success_rate:.1f}%<br>",
            f"Duration: {self.duration_seconds:.2f}s",
            "</div>"
        ]
        return "".join(lines)
