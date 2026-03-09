import json
import logging
import os
import typing as t
from copy import copy

import httpx
import packaging.version
import pandas
from httpx import URL, USE_CLIENT_DEFAULT, Response

import deepchecks_llm_client
from deepchecks_llm_client.data_types import (
    AnnotationType,
    Application,
    ApplicationType,
    ApplicationVersion,
    ApplicationVersionSchema,
    BuiltInInteractionType,
    DatasetType,
    EnvType,
    InteractionCompleteEvents,
    InteractionUpdate,
    LogInteraction,
    PropertyColumnType,
)
from deepchecks_llm_client.exceptions import DeepchecksLLMClientVersionInNewerError
from deepchecks_llm_client.utils import get_or_create_application_version_id, maybe_raise

__all__ = ["API"]

logger = logging.getLogger(__name__)

TAPI = t.TypeVar("TAPI", bound="API")  # pylint: disable=invalid-name
NULL_STRINGS = {"n/a", "na", "nan", "none"}


class CustomClient(httpx.Client):
    def request(
            self,
            method: str,
            url,
            *,
            content=None,
            data=None,
            files=None,
            json=None,  # pylint: disable=redefined-outer-name
            params=None,
            headers=None,
            cookies=None,
            auth=USE_CLIENT_DEFAULT,
            follow_redirects=USE_CLIENT_DEFAULT,
            timeout=USE_CLIENT_DEFAULT,
            extensions=None,
    ) -> Response:
        json = self._cleanup_json_before_request(json)
        return super().request(
            method,
            url=url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    @classmethod
    def _cleanup_json_before_request(cls, json_data: t.Union[dict, list, None]) -> t.Union[dict, list, None]:
        def _replace_invalid_value(value):
            try:
                if pandas.isnull(value):
                    return None
                if isinstance(value, str) and value.lower().strip() in NULL_STRINGS:
                    return None
            except (TypeError, ValueError):
                pass

            return value

        if json_data is None:
            return None

        if isinstance(json_data, dict):
            for key, value in json_data.items():
                if isinstance(value, dict):
                    json_data[key] = cls._cleanup_json_before_request(value)
                elif isinstance(value, list):
                    json_data[key] = [
                        cls._cleanup_json_before_request(el)
                        if isinstance(el, (dict, list))
                        else el
                        for el in value if _replace_invalid_value(el) is not None
                    ]
                else:
                    json_data[key] = _replace_invalid_value(value)

        elif isinstance(json_data, list):
            json_data = [
                cls._cleanup_json_before_request(el) if isinstance(el, (dict, list))
                else el
                for el in json_data if _replace_invalid_value(el) is not None
            ]

        return json_data


class API:
    """DeepchecksLLMClient API class."""

    session: httpx.Client
    original_host: URL

    @classmethod
    def instantiate(cls: t.Type[TAPI],
                    host: str,
                    token: t.Optional[str] = None,
                    validate_connection: bool = False) -> TAPI:
        headers = (
            {"Authorization": f"Basic {token}", "x-deepchecks-origin": "SDK"}
            if token
            else {"x-deepchecks-origin": "SDK"}
        )
        session = CustomClient(
            base_url=host,
            headers=headers,
            timeout=60,
            verify=os.getenv("VERIFY_SSL", "True").lower() == "true",
        )
        if os.getenv("AWS_PARTNER_APP_AUTH") is not None:
            from deepchecks_llm_client.hadron_auth import SigV4Auth  # pylint: disable=import-outside-toplevel
            session.auth = SigV4Auth()
        return cls(
            session=session,
            validate_connection=validate_connection
        )

    def __init__(self, session: httpx.Client, validate_connection: bool = False):
        self.session = copy(session)
        self.original_host = self.session.base_url
        self.session.base_url = self.session.base_url.join("/api/v1")
        self._async_client = httpx.AsyncClient(
            base_url=self.session.base_url,
            headers=dict(self.session.headers),
            timeout=self.session.timeout,
        )

        try:
            backend_version = packaging.version.parse(self.retrieve_backend_version())
            client_version = packaging.version.parse(self.retrieve_client_version())
            self.session.headers.update({"x-sdk-version": str(client_version)})
        except packaging.version.InvalidVersion as ex:
            raise RuntimeError("Not able to compare backend and client versions, "
                               "backend or client use incorrect or legacy versioning schema.") from ex
        except httpx.ConnectError as ex:
            logger.exception(f"Could not connect to backend {self.original_host}, either the server is down or "
                             f"you are using an incorrect host name")
            if validate_connection:
                raise ex

        else:
            self._verify_version_compatability(backend_version, client_version)

    def _verify_version_compatability(self, backend_version, client_version):
        if client_version.major > backend_version.major \
                or (client_version.major == backend_version.major and client_version.minor > backend_version.minor):
            raise DeepchecksLLMClientVersionInNewerError(
                f"SDK version {client_version} is newer and is not compatible with the backend version {backend_version}, "
                f"please downgrade the SDK by using {self.get_pip_install_msg(backend_version)}"
            )
        if backend_version.major != client_version.major:
            logger.error(
                f"SDK version {client_version} is not compatible with backend version {backend_version}, "
                f"please upgrade SDK to the compatible version using {self.get_pip_install_msg(backend_version)}"
            )
        else:
            versions_diff = backend_version.minor - client_version.minor
            if 0 < versions_diff <= 3:
                logger.warning(
                    f"SDK version is {client_version}, while server version is {backend_version}, client version {client_version} "
                    f"will be deprecated in {4 - versions_diff} releases from now, "
                    f"please update you SDK version to the compatible version using {self.get_pip_install_msg(backend_version)}"
                )
            elif versions_diff > 3:
                logger.error(
                    f"SDK version {client_version} is deprecated (backend version is {backend_version}), please upgrade SDK "
                    f"to the compatible version using {self.get_pip_install_msg(backend_version)}"
                )

    @classmethod
    def get_pip_install_msg(cls, backend_version) -> str:
        return f"pip install -U 'deepchecks-llm-client>={backend_version.major}." \
               f"{backend_version.minor},<{backend_version.major}.{backend_version.minor + 1}'"

    def retrieve_backend_version(self) -> str:
        payload = maybe_raise(self.session.get("backend-version")).json()
        return payload["version"]

    def retrieve_client_version(self) -> str:
        return deepchecks_llm_client.__version__

    def get_application(self, app_name: str) -> t.Dict[str, t.Any]:
        payload = maybe_raise(self.session.get("applications", params={"name": [app_name]})).json()
        return payload[0] if len(payload) > 0 else None

    def get_applications(self) -> t.List[Application]:
        applications = maybe_raise(self.session.get("applications", params={"include_extended_data": True})).json()
        return [
            Application(
                id=app["id"],
                name=app["name"],
                kind=app["kind"],
                created_at=app["created_at"],
                updated_at=app["updated_at"],
                in_progress=app["additional_data"]["in_progress"],
                description=app["description"],
                log_latest_insert_time_epoch=app["additional_data"]["log_latest_insert_time_epoch"],
                n_of_llm_properties=app["additional_data"]["n_of_llm_properties"],
                n_of_interactions=app["additional_data"]["n_of_interactions"],
                notifications_enabled=app["additional_data"]["notifications_enabled"],
                interaction_types=[interaction_type["name"] for interaction_type in app["interaction_types"]],
                versions=[
                    ApplicationVersion(
                        id=app_version["id"],
                        name=app_version["name"],
                        created_at=app_version["created_at"],
                        updated_at=app_version["updated_at"],
                        description=app_version["description"],
                        additional_fields=app_version["additional_fields"],
                    ) for app_version in app["versions"]
                ],
            ) for app in applications
        ]

    def get_versions(self, app_name: str) -> t.List[ApplicationVersion]:
        # Get application first to get its ID
        app = self.get_application(app_name)
        if not app:
            return []
        versions = maybe_raise(self.session.get(f"applications/{app['id']}/versions")).json()

        return [
            ApplicationVersion(
                id=app_version["id"],
                name=app_version["name"],
                created_at=app_version["created_at"],
                updated_at=app_version["updated_at"],
                description=app_version["description"],
                additional_fields=app_version["additional_fields"],
            ) for app_version in versions
        ]

    def create_application_version(
            self,
            application_id: int,
            version_name: str,
            description: t.Optional[str] = None,
            additional_fields: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        return maybe_raise(
            self.session.post(
                f"applications/{application_id}/versions",
                json={
                    "name": version_name,
                    "additional_fields": additional_fields,
                    "description": description,
                },
            )
        ).json()

    def create_application(
            self,
            app_name: str,
            app_type: ApplicationType,
            versions: t.Optional[t.List[ApplicationVersionSchema]] = None,
            description: t.Optional[str] = None,
    ):
        return maybe_raise(
            self.session.post(
                "applications",
                json={
                    "name": app_name,
                    "kind": app_type,
                    "versions": [version.to_json() for version in versions] if versions else [],
                    "description": description,
                },
            )
        ).json()

    def annotate(self,
                 user_interaction_id: str,
                 version_id: int,
                 annotation: t.Union[AnnotationType, str] = None,
                 reason: t.Optional[str] = None) \
            -> t.Optional[httpx.Response]:
        # pylint: disable=redefined-builtin
        return maybe_raise(
            self.session.put(
                "annotations",
                json={
                    "user_interaction_id": user_interaction_id,
                    "application_version_id": version_id,
                    "value": (
                        None if annotation is None else
                        annotation.value if isinstance(annotation, AnnotationType)
                        else str(annotation).lower().strip()
                    ),
                    "reason": reason
                }
            )
        )

    def update_interaction(
            self,
            app_version_id: int,
            user_interaction_id: str,
            interaction_update: InteractionUpdate
    ) -> t.Optional[httpx.Response]:
        # pylint: disable=redefined-builtin
        return maybe_raise(
            self.session.put(
                f"application-versions/{app_version_id}/interactions/{user_interaction_id}",
                json=interaction_update.to_json()
            )
        )

    def delete_interactions(self, user_interaction_ids: t.List[str], app_version_id: int):
        return maybe_raise(
            self.session.request(
                method="DELETE",
                url=f"application-versions/{app_version_id}/interactions",
                json=user_interaction_ids,
            )
        )

    def log_batch(
            self,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            interactions: t.List[LogInteraction],
            ignore_duplicates: bool = False,
    ):
        application_version_id = get_or_create_application_version_id(self, app_name, version_name)

        return maybe_raise(
            self.session.post(
                f"application-versions/{application_version_id}/interactions",
                json={
                    "env_type": env_type.value if isinstance(env_type, EnvType) else env_type.upper(),
                    "interactions": [interaction.to_json() for interaction in interactions],
                    "ignore_duplicates": ignore_duplicates,
                },
            ),
            expected=201,
        )

    def log_interaction(
            self,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            interaction: LogInteraction,
    ) -> t.Optional[httpx.Response]:
        """The log_interaction method is used to log user interactions.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Version name
        env_type : EnvType
            Environment
        interaction : LogInteraction
            Interaction object containing the interaction data
        Returns
        -------
        httpx.Response
            The HTTP response from logging the user interaction

        """
        # pylint: disable=redefined-builtin

        application_version_id = get_or_create_application_version_id(self, app_name, version_name)

        return maybe_raise(
            self.session.post(
                f"application-versions/{application_version_id}/interactions",
                json={
                    "env_type": env_type.value if isinstance(env_type, EnvType) else env_type.upper(),
                    "interactions": [interaction.to_json()]
                }
            ),
            expected=201
        )

    def get_interactions(self, application_version_id: int,
                         limit: int, offset: int,
                         env_type: t.Union[EnvType, str],
                         start_time_epoch: t.Union[int, None],
                         end_time_epoch: t.Union[int, None],
                         user_interaction_ids: t.Union[t.List[str], None] = None,
                         include_incomplete: bool = False,
                         session_ids: t.Union[t.List[str], None] = None,
                         interaction_type: t.Union[str, None] = None,
                         ) -> t.List:
        return maybe_raise(
            self.session.post(
                f"application-versions/{application_version_id}/get-interactions-by-filter",
                json={
                    "filter_schema": {
                        "environment": env_type.value if isinstance(env_type, EnvType) else env_type,
                        "start_time_epoch": start_time_epoch,
                        "end_time_epoch": end_time_epoch,
                        "user_interaction_ids": user_interaction_ids,
                        "session_ids": session_ids,
                        "interaction_type": interaction_type,
                    },
                    "pagination_schema": {
                        "limit": limit,
                        "offset": offset,
                    }
                },
                params={"return_topics": True, "include_incomplete": include_incomplete}
            )
        ).json()

    def get_interactions_csv(
            self,
            application_version_id: int,
            return_topics: bool,
            return_annotation_data: bool,
            return_builtin_props: bool,
            return_user_value_properties: bool,
            return_custom_prompt: bool,
            return_similarities: bool,
            env_type: t.Union[EnvType, str],
            start_time_epoch: t.Union[int, None],
            end_time_epoch: t.Union[int, None],
            user_interaction_ids: t.Union[t.List[str], None] = None,
            include_incomplete: bool = False,
            return_steps: bool = True,
            session_ids: t.Union[t.List[str], None] = None,
            interaction_type: t.Union[str, None] = None,
            return_session_related: bool = False,
            return_only_input_output_texts: bool = False,
    ) -> str:
        return maybe_raise(
            self.session.post(
                f"application-versions/{application_version_id}/interactions-download-all-by-filter",
                json={
                    "environment": env_type.value if isinstance(env_type, EnvType) else env_type,
                    "start_time_epoch": start_time_epoch,
                    "end_time_epoch": end_time_epoch,
                    "user_interaction_ids": user_interaction_ids,
                    "session_ids": session_ids,
                    "interaction_type": interaction_type,
                },
                params={"return_topics": return_topics,
                        "return_builtin_props": return_builtin_props,
                        "return_user_value_properties": return_user_value_properties,
                        "return_custom_prompt": return_custom_prompt,
                        "return_annotation_data": return_annotation_data,
                        "return_similarities_data": return_similarities,
                        "include_incomplete": include_incomplete,
                        "return_steps": return_steps,
                        "return_session_related": return_session_related,
                        "return_only_input_output_texts": return_only_input_output_texts,
                        }
            )
        ).text

    def get_interaction_by_user_interaction_id(self, version_id: int, user_interaction_id: str):
        return maybe_raise(self.session.get(
            f"application-versions/{version_id}/interactions/{user_interaction_id}"
        )).json()

    def update_interaction_type_config(self, application_id: int, interaction_type: str, file):
        if isinstance(file, str):
            with open(file, "rb") as f:
                data = {"file": ("filename", f)}
        else:
            data = {"file": ("filename", file)}
        maybe_raise(self.session.put(f"applications/{application_id}/interaction-type-config",
                                     params={"interaction_type": interaction_type}, files=data))

    def get_interaction_type_config(self, application_id: int, interaction_type: str,
                                    file_save_path: t.Union[str, None] = None) -> str:
        text = maybe_raise(self.session.get(f"applications/{application_id}/interaction-type-config",
                                            params={"interaction_type": interaction_type})).text
        if file_save_path:
            with open(file_save_path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def get_pentest_prompts(
            self,
            app_id: int,
            probes: t.Optional[t.List[str]] = None,
    ) -> str:
        if probes:
            return maybe_raise(self.session.get(f"applications/{app_id}/pentest-prompts", params={"probes": probes})).text
        return maybe_raise(self.session.get(f"applications/{app_id}/pentest-prompts")).text

    def get_user_value_properties_definitions(self, application_id: int, interaction_type: t.Optional[str] = None) -> t.List[dict]:
        return maybe_raise(
            self.session.get(
                f"applications/{application_id}/properties-definitions",
                params={"property_types": ["custom"], "interaction_type": interaction_type}
            )
        ).json()

    def create_user_value_property_definition(
            self,
            application_id: int,
            name: str,
            prop_type: t.Union[PropertyColumnType, str],
            description: str = "",
            interaction_type: t.Optional[str] = None,
    ) -> None:
        return maybe_raise(
            self.session.post(
                f"applications/{application_id}/custom-prop-definitions",
                json=[
                    {
                        "display_name": name,
                        "type": prop_type.value if isinstance(prop_type, PropertyColumnType) else prop_type,
                        "description": description,
                        "interaction_type": interaction_type,
                    }]
            )
        )

    def get_interactions_complete_status(
            self,
            app_version_id: int,
            events_to_check: t.List[InteractionCompleteEvents],
            user_interaction_ids: t.List[str],
    ) -> t.Dict[InteractionCompleteEvents, bool]:
        return maybe_raise(
            self.session.post(
                f"application-versions/{app_version_id}/interactions/complete-status",
                json={"events_to_check": events_to_check, "user_interaction_ids": user_interaction_ids}
            )
        ).json()

    def update_application_version(
            self,
            app_version_id: int,
            name: t.Optional[str] = None,
            description: t.Optional[str] = None,
            additional_fields: t.Optional[dict] = None,
    ):
        return maybe_raise(
            self.session.put(
                f"application-versions/{app_version_id}",
                json={
                    "name": name,
                    "description": description,
                    "additional_fields": additional_fields,
                }
            )
        )

    def create_interaction_type(
            self,
            app_id: int,
            name: str,
            template_name: t.Optional[BuiltInInteractionType] = None
    ):
        return maybe_raise(
            self.session.post(
                f"applications/{app_id}/interaction-types",
                json={
                    "name": name,
                    "type": template_name,
                }
            )
        ).json()

    def get_interaction_types(self, app_id: int):
        return maybe_raise(
            self.session.get(
                f"applications/{app_id}/interaction-types",
            )
        ).json()

    def update_interaction_type(
            self,
            application_id: int,
            interaction_type_id: int,
            name: str,
    ):
        """Update interaction type.

        Args:
            application_id: Application ID
            interaction_type_id: The ID of the interaction type to update
            name: New name for the interaction type
        """
        return maybe_raise(
            self.session.put(
                f"/applications/{application_id}/interaction-types/{interaction_type_id}",
                json={"name": name}
            )
        )

    def delete_interaction_type(
            self,
            application_id: int,
            interaction_type_id: int,
    ):
        """Delete interaction type.

        Args:
            application_id: Application ID
            interaction_type_id: The ID of the interaction type to delete
        """
        return maybe_raise(
            self.session.delete(
                f"/applications/{application_id}/interaction-types/{interaction_type_id}",
            )
        )

    def create_interaction_type_version_data(
            self,
            interaction_type_id: int,
            application_version_id: int,
            model: t.Optional[str] = None,
            prompt: t.Optional[str] = None,
            metadata_params: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        """Create interaction type version data.

        Parameters
        ----------
        interaction_type_id : int
            Interaction type id
        application_version_id : int
            Application version id
        model : str, optional
            Model name
        prompt : str, optional
            Prompt template
        metadata_params : dict, optional
            Additional metadata parameters

        Returns
        -------
        dict
            Created interaction type version data
        """
        # Use the new RESTful endpoint
        return maybe_raise(
            self.session.post(
                f"application-versions/{application_version_id}/interaction-types/{interaction_type_id}/metadata",
                json={
                    "model": model,
                    "prompt": prompt,
                    "metadata_params": metadata_params or {},
                },
            )
        ).json()

    def get_interaction_type_version_data(
        self,
        application_version_id: int,
        interaction_type_id: int,
    ):
        """Get interaction type version data by application version and interaction type.

        Parameters
        ----------
        application_version_id : int
            Application version id
        interaction_type_id : int
            Interaction type id

        Returns
        -------
        dict
            Interaction type version data
        """
        # Use the new RESTful endpoint
        return maybe_raise(
            self.session.get(
                f"application-versions/{application_version_id}/interaction-types/{interaction_type_id}/metadata",
            )
        ).json()

    def send_spans(
            self,
            app_name: str,
            version_name: str,
            env_type: t.Union[EnvType, str],
            spans: list[dict]  # | t.Sequence[ReadableSpan]
    ) -> t.Optional[httpx.Response]:
        """Send OpenTelemetry spans to the Deepchecks LLM backend.

        Parameters
        ----------
        app_name : str
            Application name
        version_name : str
            Name of application version
        env_type : EnvType or str
            Environment type, can be an instance of EnvType or a string
        spans : list[dict] | List[ReadableSpan]
            List of OpenTelemetry spans to send

        Returns
        -------
        httpx.Response
            The HTTP response from sending the spans
        """
        application_version_id = get_or_create_application_version_id(self, app_name, version_name)

        return maybe_raise(
            self.session.post(
                f"application-versions/{application_version_id}/span-interactions",
                json={
                    "env_type": env_type.value if isinstance(env_type, EnvType) else env_type.upper(),
                    # if spans are ReadableSpan convert them to dicts
                    "spans": [span if isinstance(span, dict) else json.loads(span.to_json()) for span in spans],
                },
            ),
            expected=(200, 201),
        )

    # Dataset API methods

    def get_datasets(self, application_id: int) -> t.List[t.Dict[str, t.Any]]:
        """Get all datasets for an application."""
        return maybe_raise(self.session.get(f"applications/{application_id}/datasets")).json()

    def get_dataset(self, application_id: int, dataset_id: int) -> t.Dict[str, t.Any]:
        """Get a specific dataset."""
        return maybe_raise(self.session.get(f"applications/{application_id}/datasets/{dataset_id}")).json()

    def create_dataset(
        self, application_id: int, dataset_name: str, dataset_type: DatasetType = DatasetType.SINGLE_TURN
    ) -> t.Dict[str, t.Any]:
        """Create a new dataset."""
        return maybe_raise(
            self.session.post(
                f"applications/{application_id}/datasets",
                json={"dataset_name": dataset_name, "dataset_type": dataset_type.value}
            ),
            expected=(200, 201)
        ).json()

    def update_dataset(self, application_id: int, dataset_id: int, dataset_name: str) -> t.Dict[str, t.Any]:
        """Update a dataset."""
        return maybe_raise(
            self.session.put(
                f"applications/{application_id}/datasets/{dataset_id}",
                json={"dataset_name": dataset_name}
            )
        ).json()

    def get_dataset_samples(
        self,
        application_id: int,
        dataset_id: int,
        offset: int = 0,
        limit: int = 100
    ) -> t.Dict[str, t.Any]:
        """Get dataset samples with pagination."""
        return maybe_raise(
            self.session.get(
                f"applications/{application_id}/datasets/{dataset_id}/samples",
                params={"offset": offset, "limit": limit}
            )
        ).json()

    def create_dataset_samples(
        self,
        application_id: int,
        dataset_id: int,
        samples: t.List[t.Dict[str, t.Any]]
    ) -> t.Dict[str, t.Any]:
        """Create multiple dataset samples."""
        return maybe_raise(
            self.session.post(
                f"applications/{application_id}/datasets/{dataset_id}/samples",
                json={"samples": samples}
            ),
            expected=(200, 201)
        ).json()

    def simulate_human(
        self,
        application_id: int,
        turns_amount: int,
        input_data: t.List[t.Dict[str, t.Any]],
    ) -> t.Dict[str, t.Any]:
        """Call the simulate-human endpoint to generate a simulated human response."""
        return maybe_raise(
            self.session.post(
                f"applications/{application_id}/simulate-human",
                json={"turns_amount": turns_amount, "input": input_data}
            )
        ).json()

    async def async_simulate_human(
        self,
        application_id: int,
        turns_amount: int,
        input_data: t.List[t.Dict[str, t.Any]],
    ) -> t.Dict[str, t.Any]:
        """Async version of simulate_human for use in async contexts like dataset_runner."""
        response = await self._async_client.post(
            f"applications/{application_id}/simulate-human",
            json={"turns_amount": turns_amount, "input": input_data}
        )
        return maybe_raise(response).json()

    # Deployment API methods

    def get_deployments(self, application_id: int) -> t.List[t.Dict[str, t.Any]]:
        """Get all deployments for an application."""
        return maybe_raise(self.session.get(f"applications/{application_id}/deployments")).json()

    def get_deployment(self, application_id: int, deployment_id: int) -> t.Dict[str, t.Any]:
        """Get a specific deployment."""
        return maybe_raise(self.session.get(f"applications/{application_id}/deployments/{deployment_id}")).json()

    def create_deployment(
        self,
        application_id: int,
        deployment_name: str,
        deployment_url: str,
        timeout: int,
        max_concurrent: int,
        max_retries: int,
        headers: t.Optional[t.List[t.Dict[str, t.Any]]] = None
    ) -> t.Dict[str, t.Any]:
        """Create a new deployment."""
        data = {
            "deployment_name": deployment_name,
            "deployment_url": deployment_url,
            "timeout": timeout,
            "max_concurrent": max_concurrent,
            "max_retries": max_retries,
        }
        if headers is not None:
            data["headers"] = headers

        return maybe_raise(
            self.session.post(
                f"applications/{application_id}/deployments",
                json=data
            ),
            expected=(200, 201)
        ).json()

    def update_deployment(
        self,
        application_id: int,
        deployment_id: int,
        deployment_name: t.Optional[str] = None,
        deployment_url: t.Optional[str] = None,
        timeout: t.Optional[int] = None,
        max_concurrent: t.Optional[int] = None,
        max_retries: t.Optional[int] = None,
        headers: t.Optional[t.List[t.Dict[str, t.Any]]] = None
    ) -> t.Dict[str, t.Any]:
        """Update a deployment."""
        data = {}
        if deployment_name is not None:
            data["deployment_name"] = deployment_name
        if deployment_url is not None:
            data["deployment_url"] = deployment_url
        if timeout is not None:
            data["timeout"] = timeout
        if max_concurrent is not None:
            data["max_concurrent"] = max_concurrent
        if max_retries is not None:
            data["max_retries"] = max_retries
        if headers is not None:
            data["headers"] = headers

        return maybe_raise(
            self.session.put(
                f"applications/{application_id}/deployments/{deployment_id}",
                json=data
            )
        ).json()
