"""Module containing all the client functionality for the Deepchecks LLM Client."""
import io
import json
import logging
import os
import time
import typing as t
from datetime import datetime

import pandas as pd

from deepchecks_llm_client.api import API
from deepchecks_llm_client.data_types import (
    AnnotationType,
    ApplicationType,
    ApplicationVersionSchema,
    BuiltInInteractionType,
    Dataset,
    DatasetRunResult,
    DatasetSample,
    DatasetType,
    Deployment,
    EnvType,
    Interaction,
    InteractionCompleteEvents,
    InteractionType,
    InteractionTypeVersionData,
    InteractionUpdate,
    LogInteraction,
    PropertyColumnType,
    Span,
    Step,
    UserValueProperty,
    UserValuePropertyType,
)
from deepchecks_llm_client.dataset_operations import execute_app as execute_app_impl
from deepchecks_llm_client.exceptions import DeepchecksLLMClientError
from deepchecks_llm_client.utils import (
    HandleExceptions,
    HandleGeneratorExceptions,
    get_application,
    get_application_version_id,
    get_or_create_application_version_id,
    get_timestamp,
    paginate_all,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
init_logger = logging.Logger(__name__ + ".init")

DEFAULT_ENV_TYPE = EnvType.PROD
DEFAULT_DC_HOST = "https://app.llm.deepchecks.com"


class DeepchecksLLMClient:
    """DeepchecksLLMClient is the class through which you can interact with the Deepchecks LLM API."""

    def __init__(self,
                 host: str = None,
                 api_token: str = None,
                 log_level: int = logging.WARNING,
                 silent_mode: bool = False):
        """Init Deepchecks LLM Client. Connect to the server and perform a simple handshake with the API.

        Parameters
        ==========
        host : str
            The URL of the Deepchecks server to communicate with. By default,
            it's set to the Deepchecks LLM host (https://app.llm.deepchecks.com),
            but you can override it by providing a custom URL.
            If not provided, the value is read from the environment variable DEEPCHECKS_LLM_HOST.
        api_token : str
            Deepchecks API Token (can be generated from the UI).
            If not provided, the value is read from the environment variable DEEPCHECKS_LLM_API_TOKEN.
        silent_mode: bool, default=False
            If True, the SDK will print logs upon encountering errors rather than raising them
        log_level: int, default=logging.WARNING
            Set SDK loggers logging level

        Returns
        =======
        None
        """
        self._api: API = None
        self._host = host or os.environ.get("DEEPCHECKS_LLM_HOST", None) \
            or os.environ.get("AWS_PARTNER_APP_URL", DEFAULT_DC_HOST)  # AWS Hadron env var
        self._api_token = api_token or os.environ.get("DEEPCHECKS_LLM_API_TOKEN")

        logger.setLevel(log_level)
        self._log_level = log_level
        self.silent_mode = silent_mode

        self._init_api()

    @HandleExceptions(init_logger)
    def _init_api(self):
        if self._api is None:
            if self._host is not None and self._api_token is not None:
                self._api = API.instantiate(host=self._host, token=self._api_token)
            else:
                raise DeepchecksLLMClientError("host/token parameters must be provided")

    @property
    def api(self) -> API:
        """Get the API object for the client."""
        if self._api:
            return self._api
        raise DeepchecksLLMClientError("deepchecks llm client was not initialized correctly, please re-create it")

    @HandleExceptions(logger)
    def create_application(
            self,
            app_name: str,
            app_type: ApplicationType,
            versions: t.Optional[t.List[ApplicationVersionSchema]] = None,
            description: t.Optional[str] = None,
    ):
        """Create a new application.

        Parameters
        ----------
        app_name
            Application name
        app_type
            The Default Interaction Type for interactions of this application
        versions
            List of versions to create for the application
        description
            Description of the application

        Returns
        -------
            The response from the API
        """
        self.api.create_application(app_name=app_name, app_type=app_type, versions=versions, description=description)
        # Clear caches so subsequent calls get the new application
        get_application.cache_clear()
        get_application_version_id.cache_clear()
        get_or_create_application_version_id.cache_clear()
        return self.api.get_application(app_name=app_name)

    @HandleExceptions(logger)
    def create_app_version(
            self,
            app_name: str,
            version_name: str,
            description: t.Optional[str] = None,
            additional_fields: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        """Create a new application version.

        Parameters
        ----------
        app_name
            Application name
        version_name
            Version name
        description
            Description of the version
        additional_fields
            Additional fields to add to the version
        Returns
        -------
            The response from the API
        """
        app_id = self._get_application_id(app_name)
        result = self.api.create_application_version(
            application_id=app_id,
            version_name=version_name,
            description=description,
            additional_fields=additional_fields
        )
        # Clear caches so subsequent calls get the new version
        get_application.cache_clear()
        get_application_version_id.cache_clear()
        get_or_create_application_version_id.cache_clear()
        return result

    @HandleExceptions(logger)
    def update_application_version(
        self,
        app_version_id: int,
        name: t.Optional[str] = None,
        description: t.Optional[str] = None,
        additional_fields: t.Optional[dict] = None,
    ):
        """Create a new application.

        Parameters
        ----------
        app_version_id
            Application version id to udate
        name
            New version name
        description
            New description of the version
        additional_fields
            Additional fields to add to the version
        Returns
        -------
            The response from the API
        """
        self.api.update_application_version(
            app_version_id=app_version_id,
            name=name,
            description=description,
            additional_fields=additional_fields,
        )
        # Clear caches so subsequent calls get the updated version data
        get_application.cache_clear()
        get_application_version_id.cache_clear()
        get_or_create_application_version_id.cache_clear()

    @HandleExceptions(logger)
    def get_applications(self):
        """Get all applications as Application objects.

        Note that "custom" attribute in ApplicationVersion is deprecated. Please use `additional_fields` instead.

        Returns
        -------
        List[Application]
            List of applications
        """
        return self.api.get_applications()

    @HandleExceptions(logger)
    def get_versions(self, app_name: str):
        """Get all versions of an application as Version objects.

        Note that "custom" attribute in ApplicationVersion is deprecated. Please use `additional_fields` instead.

        Parameters
        ----------
        app_name
            Name of the application to get versions for

        Returns
        -------
        List[ApplicationVersion]
            List of versions
        """
        return self.api.get_versions(app_name=app_name)

    @HandleExceptions(logger)
    def annotate(self, app_name: str, version_name: str, user_interaction_id: str,
                 annotation: t.Union[AnnotationType, str] = None, reason: t.Optional[str] = None):
        """Annotate a specific interaction by its user_interaction_id.

        Parameters
        ----------
        app_name
            Application name
        version_name
            Name of the version to which this interaction belongs
        user_interaction_id
            The user_interaction_id of the interaction to annotate
        annotation
            Could be one of AnnotationType.GOOD, AnnotationType.BAD, AnnotationType.UNKNOWN
            or None to remove annotation
        reason
            String that explains the reason for the annotation

        Returns
        -------
            None
        """
        version_id = self._get_version_id(version_name=version_name, app_name=app_name)
        self.api.annotate(user_interaction_id, version_id, annotation, reason=reason)

    # pylint: disable=redefined-builtin
    @HandleExceptions(logger)
    def log_interaction(
        self,
        app_name: str,
        version_name: str,
        env_type: t.Union[EnvType, str],
        interaction: LogInteraction,
    ) -> str:
        """Log a single interaction.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of the version to which this interaction will be added
        env_type : EnvType or str
            Type of environment. Allowed values: 'eval', 'prod', 'pentest'
        interaction : LogInteraction
            The interaction data to log.
        Returns
        -------
        str
            The uuid of the interaction
        """
        # pylint: disable=redefined-builtin
        result = self.api.log_interaction(
            app_name=app_name,
            version_name=version_name,
            env_type=env_type,
            interaction=interaction,
        )
        return result.json()[0]

    @HandleExceptions(logger)
    def log_batch_interactions(
            self,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            interactions: t.List[LogInteraction],
            ignore_duplicates: bool = False,
    ) -> t.List[str]:
        """Log multiple interactions at once.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of the version to which this interaction will be added
        env_type : EnvType or str
            Type of environment. Could be one of EnvType or 'prod', 'eval', 'pentest'
        interactions : list of LogInteraction
            The list of interaction data to log.
        ignore_duplicates : bool
            If True, the system will ignore interactions with user_interaction_id that already exist in the system.

        Returns
        -------
        list of str
            List of the uuids of the interactions

        """
        result = self.api.log_batch(
            app_name=app_name,
            version_name=version_name,
            env_type=env_type,
            interactions=interactions,
            ignore_duplicates=ignore_duplicates,
        )
        return result.json()

    @HandleExceptions(logger)
    def update_interaction(
        self,
        app_name: str,
        version_name: str,
        user_interaction_id: str,
        annotation: t.Optional[t.Union[AnnotationType, str]] = None,
        annotation_reason: t.Optional[str] = None,
        user_value_properties: t.Union[t.List[UserValueProperty], None] = None,
        steps: t.Union[t.List[Step]] = None,
        information_retrieval: t.Union[t.List[str], str] = None,
        input: t.Union[str, None] = None,
        output: t.Union[str, None] = None,
        expected_output: t.Union[str, None] = None,
        full_prompt: str = None,
        history: t.Union[t.List[str], str] = None,
        is_completed: t.Union[bool, None] = None,
        started_at: t.Union[datetime, float] = None,
        finished_at: t.Union[datetime, float] = None,
        metadata: t.Optional[t.Dict[str, str]] = None,
        tokens: t.Optional[int] = None,
        input_tokens: t.Optional[int] = None,
        output_tokens: t.Optional[int] = None,
    ):
        """Update a specific interaction by its user_interaction_id.

        Parameters
        ----------
        app_name : str
            Application name
        version_name
            Name of the version to which this interaction belongs
        user_interaction_id
            Unique id of the interaction to update
        annotation : AnnotationType or str
            Could be one of AnnotationType or 'good', 'bad', 'unknown'
        annotation_reason: str
            String that explains the reason for the annotation
        user_value_properties: list of UserValueProperty
            List of user value properties for interaction
        steps: list of Step
            List of steps taken during the interaction.
            New steps will be added to existing ones for previously incomplete interactions.
            Cannot be changed for completed interactions
        information_retrieval : str or list of str
            Information retrieval. New data will be added to existing ones for previously incomplete interactions.
            Cannot be changed for completed interactions
        input: str or None
            Input data. Old value will be overridden by the new value.
            Note that once interactions are marked as completed, this value cannot be changed.
        output: str or None
            Output data. Old value will be overridden by the new value.
            It cannot be modified for completed interactions.
        expected_output: str or None
            Expected output data. Old value will be overridden by the new value.
            It cannot be modified for completed interactions.
        full_prompt: str or None
            Full prompt data. Old value will be overridden by the new value.
            It cannot be modified for completed interactions.
        history: str or list of str
            History. New data will be added to existing ones for previously incomplete interactions.
            Cannot be changed for completed interactions
        is_completed: bool or None.
            Completed interactions will be shown in the system,
            and cannot be further modified except for annotations,
            topics and user value properties.
            Interaction completeness status will be not changed if None is sent
        started_at : datetime or float
            Timestamp the interaction started_at at. Datetime format is deprecated, use timestamp instead
        finished_at : datetime or float
            Timestamp the interaction finished at. Datetime format is deprecated, use timestamp instead
        metadata: Dict[str, str], optional
            Metdata for the interaction.
        tokens: int, optional
            Token count for the interaction.
        input_tokens: int, optional
            Input token count for the interaction.
        output_tokens: int, optional
            Output token count for the interaction.
        Returns
        -------
            None
        """
        # pylint: disable=redefined-builtin
        version_id = self._get_version_id(app_name=app_name, version_name=version_name)
        interaction_update = InteractionUpdate(
            annotation=annotation,
            annotation_reason=annotation_reason,
            user_value_properties=user_value_properties,
            steps=steps,
            information_retrieval=information_retrieval,
            input=input,
            output=output,
            expected_output=expected_output,
            full_prompt=full_prompt,
            history=history,
            is_completed=is_completed,
            started_at=started_at,
            finished_at=finished_at,
            metadata=metadata,
            tokens=tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        self.api.update_interaction(
            app_version_id=version_id,
            user_interaction_id=user_interaction_id,
            interaction_update=interaction_update,
        )

    @HandleExceptions(logger)
    def delete_interactions(self, app_name: str, version_name: str, user_interaction_ids: t.List[str]):
        """Delete specific interactions by their user_interaction_ids.

        Parameters
        ----------
        app_name : str
            Application name
        version_name: str
            Name of application version
        user_interaction_ids: List[str]
            List of interaction user ids to delete

        Returns
        -------
            None
        """
        version_id = self._get_version_id(app_name=app_name, version_name=version_name)
        self.api.delete_interactions(
            user_interaction_ids=user_interaction_ids, app_version_id=version_id,
        )

    @HandleExceptions(logger)
    def update_interaction_type_config(self, app_name: str, interaction_type: str, file):
        """Update the auto-annotation YAML configuration file for the interaction type within the application.

        Parameters
        ----------
        app_name : str
            Application name
        interaction_type: str
            Interaction Type you wish to update
        file : str
            The path to the configuration file to update

        Returns
        -------
            None
        """
        app_id = self._get_application_id(app_name)
        self.api.update_interaction_type_config(application_id=app_id, interaction_type=interaction_type, file=file)

    @HandleExceptions(logger)
    def get_interaction_type_config(self, app_name: str, interaction_type: str, file_save_path: t.Union[str, None] = None) -> str:
        """Write the auto-annotation YAML configuration file to the specified path.

        Parameters
        ----------
        app_name : str
            Application name
        interaction_type: str
            Interaction Type you wish to update
        file_save_path : str | None, optional
            The path to save the configuration file to. If None, the file will not be saved.

        Returns
        -------
        str
            The auto-annotation YAML configuration file as a string.
        """
        app_id = self._get_application_id(app_name)
        return self.api.get_interaction_type_config(application_id=app_id, interaction_type=interaction_type, file_save_path=file_save_path)

    @HandleGeneratorExceptions(init_logger)
    def data_iterator(self,
                      app_name: str,
                      version_name: str,
                      env_type: t.Union[EnvType, str],
                      start_time: t.Union[datetime, int, None] = None,
                      end_time: t.Union[datetime, int, None] = None,
                      user_interaction_ids: t.Union[t.List[t.Union[str, int]], None] = None,
                      include_incomplete: bool = False,
                      session_ids: t.Union[t.List[t.Union[str, int]], None] = None,
                      interaction_type: t.Union[str, None] = None,
                      ) -> t.Iterable[Interaction]:
        """
        Fetch all interactions from the specified environment type (PROD/EVAL) as an iterable.

        Supports pagination, so this API is suitable for iterating over large amounts of data.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        env_type : EnvType | str
            The environment type from which to fetch interactions. This can be either "PROD" or "EVAL".
        start_time : datetime | int | None, optional
            The start time from which to fetch interactions. This can be a datetime object or an integer.
            If not provided, interactions will be fetched from the beginning.
        end_time : datetime | int | None, optional
            The end time until which to fetch interactions. This can be a datetime object or an integer.
            If not provided, interactions will be fetched up to the most recent.
        user_interaction_ids: list | None, optional
            user interactions ids to include into the list of interactions.annotation_reason
        include_incomplete : bool
            Return only interactions with completed status or not.
            Only completed are returned by default.
        session_ids: list | None, optional
            List of session IDs to filter by. Maximum 1000 items allowed.
        interaction_type: str | None, optional
            Interaction types to filter by.

        Returns
        -------
        Iterable[Interaction]
            An iterable collection of interactions.

        """
        version_id = self._get_version_id(app_name=app_name, version_name=version_name)

        offset = 0
        limit = 20

        while True:
            interactions = \
                self.api.get_interactions(version_id,
                                          env_type=env_type,
                                          start_time_epoch=get_timestamp(start_time),
                                          end_time_epoch=get_timestamp(end_time),
                                          user_interaction_ids=[str(el) for el in user_interaction_ids] if user_interaction_ids else None,
                                          limit=limit, offset=offset,
                                          include_incomplete=include_incomplete,
                                          session_ids=[str(el) for el in session_ids] if session_ids else None,
                                          interaction_type=interaction_type,
                                          )
            for interaction in interactions:
                yield self._build_interaction_object(interaction)

            # If the size of the data is less than the limit, we"ve reached the end
            if len(interactions) < limit:
                break

            offset += limit

    @HandleExceptions(init_logger)
    def get_data(
            self,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            return_topics: bool = True, return_annotation_data: bool = True,
            return_builtin_props: bool = True,
            return_user_value_properties: bool = True,
            return_custom_prompt: bool = True,
            start_time: t.Union[datetime, int, None] = None,
            end_time: t.Union[datetime, int, None] = None,
            return_similarities: bool = False,
            user_interaction_ids: t.Union[t.List[str], None] = None,
            include_incomplete: bool = False,
            return_steps: bool = True,
            session_ids: t.Union[t.List[str], None] = None,
            interaction_type: t.Union[str, None] = None,
            return_session_related: bool = False,
            return_only_input_output_texts: bool = False,
    ) -> t.Union[pd.DataFrame, None]:
        """
        Fetch all the interactions from the specified environment (PROD/EVAL) as a pandas DataFrame.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        env_type : EnvType | str
            The environment type from which to fetch interactions. This can be either "PROD" or "EVAL".
        return_annotation_data : bool, optional
            Whether to include annotation info in the data.
        return_topics : bool, optional
            Whether to include the topic in the data.
        return_builtin_props : bool, optional
            Whether to include built-in properties in the data.
        return_user_value_properties : bool, optional
            Whether to include user value properties in the data.
        return_custom_prompt : bool, optional
            Whether to include LLM properties in the data.
        return_similarities : bool, optional
            Whether to include similarities in the data.
        start_time : datetime | int | None, optional
            The start time from which to fetch interactions. This can be a datetime object or an integer.
            If not provided, interactions will be fetched from the beginning.
        end_time : datetime | int | None, optional
            The end time until which to fetch interactions. This can be a datetime object or an integer.
            If not provided, interactions will be fetched up to the most recent.
        user_interaction_ids: list | None, optional
            user interactions ids to include in the data.
        include_incomplete : bool
            Return only interactions with completed status or not.
            Only completed are returned by default.
        return_steps : bool, optional
            Whether to include steps in the data.
        session_ids: list | None, optional
            List of session IDs to filter by. Maximum 1000 items allowed.
        interaction_type: str | None, optional
            interaction type to filter by.
        return_session_related: bool, optional
            Whether to include session related data in the data.
        return_only_input_output_texts: bool, optional
            If true the only text fields that would be included are input and output.
        Returns
        -------
        pd.DataFrame | None
            A pandas DataFrame containing the interactions, or None in case of a problem retrieving the data.
        """
        eval_set_version_id = self._get_version_id(app_name=app_name, version_name=version_name)
        csv_as_text = self.api.get_interactions_csv(
            eval_set_version_id,
            env_type=env_type,
            start_time_epoch=get_timestamp(start_time),
            end_time_epoch=get_timestamp(end_time),
            return_topics=return_topics,
            return_annotation_data=return_annotation_data,
            return_user_value_properties=return_user_value_properties,
            return_builtin_props=return_builtin_props,
            return_custom_prompt=return_custom_prompt,
            return_similarities=return_similarities,
            user_interaction_ids=user_interaction_ids,
            include_incomplete=include_incomplete,
            return_steps=return_steps,
            session_ids=session_ids,
            interaction_type=interaction_type,
            return_session_related=return_session_related,
            return_only_input_output_texts=return_only_input_output_texts,
        )
        if not csv_as_text:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(csv_as_text))

    @HandleExceptions(init_logger)
    def get_interaction_by_user_interaction_id(self, app_name: str, version_name: str, user_interaction_id: str) -> Interaction:
        """Get a specific interaction by its user_interaction_id.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        user_interaction_id : str
            Unique id of the interaction to get

        Returns
        -------
        Interaction
            The interaction object, including the input, output, properties and other fields
        """
        version_id = self._get_version_id(app_name=app_name, version_name=version_name)
        interaction = self.api.get_interaction_by_user_interaction_id(version_id, user_interaction_id)
        return self._build_interaction_object(interaction)

    def _get_application_id(self, app_name: str):
        app = get_application(self.api, app_name)
        return app["id"]

    def _get_version_id(self, app_name: str, version_name: str, create_if_not_exist: bool = False):
        if create_if_not_exist:
            return get_or_create_application_version_id(self.api, app_name, version_name)
        else:
            return get_application_version_id(self.api, app_name, version_name)

    def get_pentest_prompts(self, app_name: str, probes: t.Optional[t.List[str]] = None) -> pd.DataFrame:
        """Get pentest prompts for the application.

        Parameters
        ----------
        app_name : str
            Application name
        probes: list of str
            List of probes to get pentest prompts for. If None, all probes will be returned.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the pentest prompts
        """
        app_id = self._get_application_id(app_name)
        csv_content = self.api.get_pentest_prompts(app_id=app_id, probes=probes)
        df = pd.read_csv(io.StringIO(csv_content))
        return df

    def get_user_value_properties(self, app_name: str, interaction_type: t.Optional[str] = None) -> t.List[UserValuePropertyType]:
        """Get the user value properties defined for the current application.

        Parameters
        ----------
        app_name : str
            Application name
        interaction_type : str
            Interaction type

        Returns
        -------
        list[UserValuePropertyType]
            A list of user value properties defined for the application
        """
        app_id = self._get_application_id(app_name)
        user_value_properties = self.api.get_user_value_properties_definitions(application_id=app_id, interaction_type=interaction_type)
        return [
            UserValuePropertyType(
                display_name=prop["property_name"],
                type=prop["column_type"],
                description=prop.get("description")
            ) for prop in user_value_properties
        ]

    def create_user_value_property(
        self,
        app_name: str,
        name: str,
        prop_type: t.Union[PropertyColumnType, str],
        description: t.Optional[str] = None,
        interaction_type: t.Optional[str] = None,
    ) -> None:
        """Define a user value property for the current application.

        Parameters
        ----------
        app_name : str
            Application name
        name : str
            Name of the user value property
        prop_type : PropertyColumnType or str
            Type of the user value property (categorical, numeric)
        description : str, optional
            Description of the user value property
        interaction_type : str, optional
            Interaction type
        Returns
        -------
        None
        """
        app_id = self._get_application_id(app_name)
        self.api.create_user_value_property_definition(
            application_id=app_id,
            name=name,
            prop_type=prop_type,
            description=description,
            interaction_type=interaction_type,
        )

    @HandleExceptions(init_logger)
    def create_interaction_type(
        self,
        app_name: str,
        name: str,
        template_name: t.Optional[BuiltInInteractionType] = None
    ) -> int:
        """Activating Builtin type or Creating a new custom interaction type for the current application.

        Parameters
        ----------
        app_name : str
            Application name
        name : str
            Name of the interaction type
        template_name: BuiltInInteractionType
            Template of the interaction type
        Returns
        -------
        int
            Interaction type id
        """
        app_id = self._get_application_id(app_name)
        create_interaction_response = self.api.create_interaction_type(
            app_id=app_id,
            name=name,
            template_name=template_name,
        )
        return create_interaction_response["id"]

    @HandleExceptions(init_logger)
    def update_interaction_type(
        self,
        app_name: str,
        interaction_type_id: int,
        name: t.Optional[str] = None,
    ):
        """Update interaction type for the current application.

        Parameters
        ----------
        app_name : str
            Application name
        interaction_type_id : int
            Interaction type id
        name : str, optional
            Name of the custom interaction type

        Returns
        -------
        None
        """
        app_id = self._get_application_id(app_name)
        self.api.update_interaction_type(
            application_id=app_id,
            interaction_type_id=interaction_type_id,
            name=name,
        )

    @HandleExceptions(init_logger)
    def get_interaction_types(self, app_name: str) -> t.List[InteractionType]:
        """Get the custom interaction types defined for the current application.

        Parameters
        ----------
        app_name : str
            Application name

        Returns
        -------
        list[InteractionType]
            A list of interaction types defined for the application
        """
        app_id = self._get_application_id(app_name)
        interaction_types = self.api.get_interaction_types(
            app_id=app_id,
        )
        return [
            InteractionType(
                id=interaction_type["id"],
                name=interaction_type["name"],
            ) for interaction_type in interaction_types
        ]

    @HandleExceptions(init_logger)
    def delete_interaction_type(self, app_name: str, interaction_type_id: int):
        """Delete interaction type.

        Parameters
        ----------
        app_name : str
            Application name
        interaction_type_id : int
            Interaction type id
        Returns
        -------
        None
        """
        app_id = self._get_application_id(app_name)
        self.api.delete_interaction_type(
            application_id=app_id,
            interaction_type_id=interaction_type_id,
        )

    @HandleExceptions(logger)
    def create_interaction_type_version_data(
        self,
        app_name: str,
        version_name: str,
        interaction_type: str,
        model: t.Optional[str] = None,
        prompt: t.Optional[str] = None,
        metadata_params: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        """Create interaction type version data.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Version name
        interaction_type : str
            Interaction type name
        model : str, optional
            Model name
        prompt : str, optional
            Prompt template
        metadata_params : dict, optional
            Additional metadata parameters

        Returns
        -------
        InteractionTypeVersionData
            Created interaction type version data
        """
        app_version_id = self._get_version_id(app_name, version_name)

        interaction_type_id = None
        interaction_types = self.get_interaction_types(app_name)
        for it in interaction_types:
            if it.name.lower() == interaction_type.lower():
                interaction_type_id = it.id
                break
        if interaction_type_id is None:
            raise DeepchecksLLMClientError(f"Interaction type '{interaction_type}' not found")

        data = self.api.create_interaction_type_version_data(
            interaction_type_id=interaction_type_id,
            application_version_id=app_version_id,
            model=model,
            prompt=prompt,
            metadata_params=metadata_params,
        )
        return InteractionTypeVersionData(
            id=data["id"],
            interaction_type_id=data["interaction_type_id"],
            application_version_id=data["application_version_id"],
            model=data["model"],
            prompt=data["prompt"],
            metadata_params=data["metadata_params"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @HandleExceptions(logger)
    def get_interaction_type_version_data(
        self,
        app_name: str,
        version_name: str,
        interaction_type: str,
    ):
        """Get interaction type version data.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Version name
        interaction_type : str
            Interaction type name

        Returns
        -------
        InteractionTypeVersionData
            Interaction type version data
        """
        app_version_id = self._get_version_id(app_name, version_name)

        interaction_type_id = None
        interaction_types = self.get_interaction_types(app_name)
        for it in interaction_types:
            if it.name.lower() == interaction_type.lower():
                interaction_type_id = it.id
                break
        if interaction_type_id is None:
            raise DeepchecksLLMClientError(f"Interaction type '{interaction_type}' not found")

        data = self.api.get_interaction_type_version_data(
            application_version_id=app_version_id,
            interaction_type_id=interaction_type_id,
        )
        return InteractionTypeVersionData(
            id=data["id"],
            interaction_type_id=data["interaction_type_id"],
            application_version_id=data["application_version_id"],
            model=data["model"],
            prompt=data["prompt"],
            metadata_params=data["metadata_params"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @HandleExceptions(init_logger)
    def get_interactions_complete_status(
            self,
            app_name: str,
            version_name: str,
            user_interaction_ids: t.List[str],
            events_to_check: t.Optional[t.List[InteractionCompleteEvents]] = None,
    ):
        """Get the completion status of interactions for specified events and user interaction IDs.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        events_to_check : Optional[List[InteractionCompleteEvents]]
            A list of events whose completion status needs to be checked. By default, only annotation_completed is
            checked
        user_interaction_ids : List[str]
            A list of user interaction IDs to check for completion status.

        Returns
        -------
        Dict[InteractionCompleteEvents, bool]
            A dictionary where keys are events from `events_to_check`
            and values are objects that indicating whether interactions for each event are completed and number of
            completed interactions {"annotation_completed": {"all_completed": True, "number_of_completed": 10}}
        """
        if not events_to_check:
            events_to_check = [InteractionCompleteEvents.ANNOTATION_COMPLETED]

        version_id = self._get_version_id(app_name=app_name, version_name=version_name)
        return self.api.get_interactions_complete_status(
            app_version_id=version_id,
            events_to_check=events_to_check,
            user_interaction_ids=user_interaction_ids,
        )

    @staticmethod
    def _validate_retry_interval_in_seconds(retry_interval_in_seconds):
        if retry_interval_in_seconds < 3:
            raise DeepchecksLLMClientError("retry_interval_in_seconds cannot be lower then 3 seconds")

    @HandleExceptions(init_logger)
    def get_data_if_calculations_completed(
            self,
            app_name: str,
            version_name: str,
            env_type: EnvType,
            user_interaction_ids: t.List[str],
            events_to_check: t.Optional[t.List[InteractionCompleteEvents]] = None,
            max_retries: int = 60,
            retry_interval_in_seconds: int = 3,
            return_topics: bool = True, return_annotation_data: bool = True,
            return_builtin_props: bool = True,
            return_user_value_properties: bool = True,
            return_custom_prompt: bool = True,
            return_similarities: bool = False,
    ) -> t.Optional[pd.DataFrame]:
        """Check if calculations for specified events are completed by calling `get_interactions_complete_status`.

        If all user_interaction_ids completed then we will fetch the data for those ids,
        if not - will sleep and retry again. Consider increasing max_retries or retry_interval_in_seconds if running
        the method returns None.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        env_type : EnvType
            Environment type
        events_to_check : Optional[List[InteractionCompleteEvents]]
            A list of events whose completion status needs to be checked. By default, only annotation_completed
            is checked
        user_interaction_ids : List[str]
            A list of user interaction IDs to retrieve data for.
        max_retries : Optional[int]
            Maximum number of retries if calculations are not completed. Defaults to 60.
        retry_interval_in_seconds : Optional[float]
            Interval between retries in seconds. Defaults to 3.
        return_annotation_data : bool, optional
            Whether to include annotation info in the data.
        return_topics : bool, optional
            Whether to include the topic in the data.
        return_annotation_data : bool, optional
            Whether to include annotation info in the data.
        return_builtin_props : bool, optional
            Whether to include built-in properties in the data.
        return_user_value_properties : bool, optional
            Whether to include custom properties in the data.
        return_custom_prompt : bool, optional
            Whether to include Custom Prompt properties in the data.
        return_similarities : bool, optional
            Whether to include similarities in the data.

        Returns
        -------
        Optional[pd.DataFrame]
            A DataFrame containing the retrieved data if calculations are completed, otherwise None.
        """
        self._validate_retry_interval_in_seconds(retry_interval_in_seconds)

        if not events_to_check:
            events_to_check = [InteractionCompleteEvents.ANNOTATION_COMPLETED]
        retry_count = 0
        while retry_count < max_retries:
            complete_statuses = self.get_interactions_complete_status(
                app_name=app_name,
                version_name=version_name,
                events_to_check=events_to_check,
                user_interaction_ids=user_interaction_ids,
            )
            if all(status["all_completed"] for status in complete_statuses.values()):
                return self.get_data(
                    app_name=app_name,
                    version_name=version_name,
                    env_type=env_type,
                    return_topics=return_topics,
                    return_annotation_data=return_annotation_data,
                    return_builtin_props=return_builtin_props,
                    return_user_value_properties=return_user_value_properties,
                    return_custom_prompt=return_custom_prompt,
                    return_similarities=return_similarities,
                    user_interaction_ids=user_interaction_ids,
                )
            else:
                for event, status in complete_statuses.items():
                    logger.debug(f"{event} = {status['number_of_completed']}")
                retry_count += 1
                time.sleep(retry_interval_in_seconds)
        return None

    @staticmethod
    def _build_interaction_object(interaction: dict) -> Interaction:
        def _get_first_data(field_data):
            """Extract data from first element of a list field."""
            if not field_data or len(field_data) == 0:
                return None
            return field_data[0]["data"]

        def _get_list_data(field_data):
            """Extract data from all elements of a list field."""
            if not field_data:
                return None
            return [item["data"] for item in field_data]

        return Interaction(
            user_interaction_id=interaction["user_interaction_id"],
            input=_get_first_data(interaction["interaction_texts"].get("input")),
            information_retrieval=_get_list_data(interaction["interaction_texts"].get("information_retrieval")),
            history=_get_list_data(interaction["interaction_texts"].get("history")),
            full_prompt=_get_first_data(interaction["interaction_texts"].get("prompt")),
            output=_get_first_data(interaction["interaction_texts"].get("output")),
            expected_output=_get_first_data(interaction["interaction_texts"].get("expected_output")),
            created_at=interaction["created_at"],
            interaction_datetime=interaction["interaction_datetime"],
            topic=interaction["topic"],
            properties=interaction.get("properties") or {},
            properties_reasons=interaction.get("properties_reasons") or {},
            is_completed=interaction["is_completed"],
            session_id=interaction["session_id"],
            interaction_type=interaction["interaction_type"]["name"],
            metadata=interaction["metadata"],
            tokens=interaction["tokens"],
            input_tokens=interaction.get("input_tokens"),
            output_tokens=interaction.get("output_tokens"),
            model=interaction.get("model"),
            model_provider=interaction.get("model_provider"),
            input_cost=interaction.get("input_cost"),
            output_cost=interaction.get("output_cost"),
            cost=interaction.get("cost"),
            annotation=None if interaction.get("annotation") is None else (
                interaction["annotation"]["value"] and AnnotationType(interaction["annotation"]["value"])),
            annotation_reason=None if interaction.get("annotation") is None else interaction["annotation"]["reason"],
        )

    @HandleExceptions(logger)
    def log_spans_file(
            self,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            json_path: str,
    ):
        """Send spans to the Deepchecks LLM backend.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        env_type : EnvType | str
            Environment type (PROD/EVAL)
        json_path : str
            Path to the JSON file containing spans.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            spans = json.load(f)
            return self.api.send_spans(
                app_name=app_name,
                version_name=version_name,
                env_type=env_type,
                spans=spans,
            )

    @HandleExceptions(logger)
    def log_spans(
            self,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            spans: t.List[Span]
    ):
        """Send spans to the Deepchecks LLM backend.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        env_type : EnvType | str
            Environment type (PROD/EVAL)
        spans : List[Span]
            List of span objects.
        """
        return self.api.send_spans(
            app_name=app_name,
            version_name=version_name,
            env_type=env_type,
            spans=[span.to_span_data() for span in spans],
        )

    @HandleExceptions(logger)
    def get_datasets(self, app_name: str) -> t.List[Dataset]:
        """Get all datasets for an application.

        Parameters
        ----------
        app_name : str
            Application name

        Returns
        -------
        List[Dataset]
            List of datasets
        """
        app_id = self._get_application_id(app_name)
        datasets = self.api.get_datasets(app_id)
        return [Dataset.from_dict(d) for d in datasets]

    @HandleExceptions(logger)
    def get_dataset(self, app_name: str, dataset_name: str) -> t.Optional[Dataset]:
        """Get a specific dataset by name.

        Parameters
        ----------
        app_name : str
            Application name
        dataset_name : str
            Dataset name

        Returns
        -------
        Dataset or None
            The dataset if found, None otherwise
        """
        datasets = self.get_datasets(app_name)
        for dataset in datasets:
            if dataset.dataset_name == dataset_name:
                return dataset
        return None

    @HandleExceptions(logger)
    def create_dataset(self, app_name: str, dataset_name: str,
                       dataset_type: DatasetType = DatasetType.SINGLE_TURN) -> Dataset:
        """Create a new dataset.

        Parameters
        ----------
        app_name : str
            Application name
        dataset_name : str
            Name for the new dataset
        dataset_type : DatasetType, optional
            Type of dataset, defaults to DatasetType.SINGLE_TURN

        Returns
        -------
        Dataset
            The created dataset
        """
        app_id = self._get_application_id(app_name)
        result = self.api.create_dataset(app_id, dataset_name, dataset_type=dataset_type)

        # API returns only ID, fetch the full dataset
        dataset_id = result["id"]
        full_dataset = self.api.get_dataset(app_id, dataset_id)
        return Dataset.from_dict(full_dataset)

    @HandleExceptions(logger)
    def update_dataset(self, app_name: str, dataset_name: str, new_dataset_name: str) -> Dataset:
        """Update a dataset's name.

        Parameters
        ----------
        app_name : str
            Application name
        dataset_name : str
            Current dataset name
        new_dataset_name : str
            New dataset name

        Returns
        -------
        Dataset
            The updated dataset
        """
        app_id = self._get_application_id(app_name)
        dataset = self.get_dataset(app_name, dataset_name)
        if not dataset:
            raise DeepchecksLLMClientError(f"Dataset '{dataset_name}' not found")

        result = self.api.update_dataset(app_id, dataset.id, new_dataset_name)
        return Dataset.from_dict(result)

    @HandleExceptions(logger)
    def get_dataset_samples(
        self,
        app_name: str,
        dataset_name: str,
        offset: int = 0,
        limit: int = 100
    ) -> t.List[DatasetSample]:
        """Get samples from a dataset with pagination.

        Parameters
        ----------
        app_name : str
            Application name
        dataset_name : str
            Dataset name
        offset : int, default=0
            Number of samples to skip
        limit : int, default=100
            Maximum number of samples to return

        Returns
        -------
        List[DatasetSample]
            List of dataset samples
        """
        app_id = self._get_application_id(app_name)
        dataset = self.get_dataset(app_name, dataset_name)
        if not dataset:
            raise DeepchecksLLMClientError(f"Dataset '{dataset_name}' not found")

        result = self.api.get_dataset_samples(app_id, dataset.id, offset, limit)

        return [
            DatasetSample(
                id=s["id"],
                dataset_id=s["dataset_id"],
                input=s["input"],
                output=s.get("output"),
                sample_metadata=s.get("sample_metadata"),
                created_at=pd.to_datetime(s["created_at"]),
                updated_at=pd.to_datetime(s["updated_at"]),
            )
            for s in result["samples"]
        ]

    @HandleExceptions(logger)
    def get_all_dataset_samples(self, app_name: str, dataset_name: str) -> t.List[DatasetSample]:
        """Get all samples from a dataset (handles pagination automatically).

        Parameters
        ----------
        app_name : str
            Application name
        dataset_name : str
            Dataset name

        Returns
        -------
        List[DatasetSample]
            All dataset samples
        """
        return paginate_all(
            lambda offset, limit: self.get_dataset_samples(app_name, dataset_name, offset, limit)
        )

    @HandleExceptions(logger)
    def add_dataset_samples(
        self,
        app_name: str,
        dataset_name: str,
        samples: t.List[t.Dict[str, t.Any]]
    ) -> None:
        """Add samples to a dataset.

        Parameters
        ----------
        app_name : str
            Application name
        dataset_name : str
            Dataset name
        samples : List[dict]
            List of samples to add. Each sample should have:
            - input: dict (required)
            - output: dict (optional)
            - sample_metadata: dict (optional)
        """
        app_id = self._get_application_id(app_name)
        dataset = self.get_dataset(app_name, dataset_name)
        if not dataset:
            raise DeepchecksLLMClientError(f"Dataset '{dataset_name}' not found")

        # Batch samples into groups of 1000 (API limit)
        batch_size = 1000
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            self.api.create_dataset_samples(app_id, dataset.id, batch)

    # Deployment Management Methods

    @HandleExceptions(logger)
    def get_deployments(self, app_name: str) -> t.List[Deployment]:
        """Get all deployments for an application.

        Parameters
        ----------
        app_name : str
            Application name

        Returns
        -------
        List[Deployment]
            List of deployments
        """
        app_id = self._get_application_id(app_name)
        deployments = self.api.get_deployments(app_id)
        return [Deployment.from_dict(d) for d in deployments]

    @HandleExceptions(logger)
    def get_deployment(self, app_name: str, deployment_name: str) -> t.Optional[Deployment]:
        """Get a specific deployment by name.

        Parameters
        ----------
        app_name : str
            Application name
        deployment_name : str
            Deployment name

        Returns
        -------
        Deployment or None
            The deployment if found, None otherwise
        """
        deployments = self.get_deployments(app_name)
        for deployment in deployments:
            if deployment.deployment_name == deployment_name:
                return deployment
        return None

    @HandleExceptions(logger)
    def create_deployment(
        self,
        app_name: str,
        deployment_name: str,
        deployment_url: str,
        timeout: int = 30,
        max_concurrent: int = 5,
        max_retries: int = 2,
        headers: t.Optional[t.List[t.Dict[str, t.Any]]] = None
    ) -> Deployment:
        """Create a new deployment.

        Parameters
        ----------
        app_name : str
            Application name
        deployment_name : str
            Name for the deployment
        deployment_url : str
            URL of the deployment endpoint
        timeout : int, default=30
            Request timeout in seconds (5-300)
        max_concurrent : int, default=5
            Maximum concurrent requests (1-20)
        max_retries : int, default=2
            Maximum number of retries (0-5)
        headers : List[dict], optional
            List of headers to include. Each header should have:
            - name: str (required)
            - value: str (required)

        Returns
        -------
        Deployment
            The created deployment
        """
        app_id = self._get_application_id(app_name)
        result = self.api.create_deployment(
            app_id,
            deployment_name,
            deployment_url,
            timeout,
            max_concurrent,
            max_retries,
            headers,
        )

        # API returns only ID, fetch the full deployment
        deployment_id = result["id"]
        full_deployment = self.api.get_deployment(app_id, deployment_id)
        return Deployment.from_dict(full_deployment)

    @HandleExceptions(logger)
    def update_deployment(
        self,
        app_name: str,
        deployment_name: str,
        new_deployment_name: t.Optional[str] = None,
        deployment_url: t.Optional[str] = None,
        timeout: t.Optional[int] = None,
        max_concurrent: t.Optional[int] = None,
        max_retries: t.Optional[int] = None,
        headers: t.Optional[t.List[t.Dict[str, t.Any]]] = None
    ) -> Deployment:
        """Update a deployment.

        Parameters
        ----------
        app_name : str
            Application name
        deployment_name : str
            Current deployment name
        new_deployment_name : str, optional
            New name for the deployment
        deployment_url : str, optional
            New URL for the deployment
        timeout : int, optional
            New timeout in seconds (5-300)
        max_concurrent : int, optional
            New max concurrent requests (1-20)
        max_retries : int, optional
            New max retries (0-5)
        headers : List[dict], optional
            New headers list

        Returns
        -------
        Deployment
            The updated deployment
        """
        app_id = self._get_application_id(app_name)
        deployment = self.get_deployment(app_name, deployment_name)
        if not deployment:
            raise DeepchecksLLMClientError(f"Deployment '{deployment_name}' not found")

        result = self.api.update_deployment(
            app_id,
            deployment.id,
            new_deployment_name,
            deployment_url,
            timeout,
            max_concurrent,
            max_retries,
            headers,
        )
        return Deployment.from_dict(result)

    async def execute_app(
        self,
        app_name: str,
        dataset_name: str,
        deployment_name: str,
        version_name: t.Optional[str] = None,
        env_type: t.Optional[str] = None,
        additional_headers: t.Optional[t.Dict[str, str]] = None,
        show_progress: bool = True,
        verify_ssl: bool = True,
        progress_callback: t.Optional[t.Callable[[int, int], None]] = None,
        sample_callback: t.Optional[t.Callable] = None,
    ) -> DatasetRunResult:
        """Run a dataset on a deployment with parallel execution.

        This method fetches the dataset samples and deployment configuration,
        then runs all samples against the deployment endpoint using the
        configured parallelism and retry settings.

        Note: This is an async method and must be called with await:
        >>> result = await client.execute_app("my-app", "test-dataset", "prod-deployment")

        Parameters
        ----------
        app_name : str
            Application name
        dataset_name : str
            Dataset name to run
        deployment_name : str
            Deployment name to run against
        version_name : str, optional
            Version name to include in dc_fields
        env_type : str, optional
            Environment type to include in dc_fields
        additional_headers : dict, optional
            Additional headers to include in requests (beyond deployment headers)
        show_progress : bool, default=True
            Whether to show a live progress widget (if in notebook with ipywidgets installed)
        verify_ssl : bool, default=True
            Whether to verify SSL certificates
        progress_callback : callable, optional
            Callback function that receives (completed_count, total_count) after each sample completes
        sample_callback : callable, optional
            Callback function that receives (sample_id, status, keyword args) for each sample status update.
            Keyword args may include: input, output, error, duration, retries

        Returns
        -------
        DatasetRunResult
            Results of running the dataset, including success/failure counts,
            individual sample results, and timing information.

        Examples
        --------
        >>> result = await client.execute_app("my-app", "test-dataset", "prod-deployment")
        >>> print(f"Success rate: {result.success_rate:.1f}%")
        >>> print(f"Failed samples: {result.failed_samples}")
        >>> # Re-run only failed samples
        >>> if result.failed_samples > 0:
        ...     retry_result = await result.rerun_failed()
        ...     print(f"Retry success rate: {retry_result.success_rate:.1f}%")
        """
        return await execute_app_impl(
            client=self,
            app_name=app_name,
            dataset_name=dataset_name,
            deployment_name=deployment_name,
            version_name=version_name,
            env_type=env_type,
            additional_headers=additional_headers,
            show_progress=show_progress,
            verify_ssl=verify_ssl,
            progress_callback=progress_callback,
            sample_callback=sample_callback,
        )
