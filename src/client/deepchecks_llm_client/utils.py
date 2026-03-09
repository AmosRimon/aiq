from __future__ import annotations

import ast
import enum
import functools
import json
import logging
import typing as t
from datetime import datetime
from json import JSONDecodeError
from types import GeneratorType

import httpx
from typing_extensions import ParamSpec, TypeVar

from deepchecks_llm_client.exceptions import (
    BaseDeepchecksLLMAPIError,
    DeepchecksLLMBadRequestError,
    DeepchecksLLMClientError,
    DeepchecksLLMServerError,
    UsageLimitExceeded,
)

if t.TYPE_CHECKING:
    from deepchecks_llm_client.api import API

P = ParamSpec("P")
T = TypeVar("T")

MAX_TOPIC_LENGTH = 80


logger = logging.getLogger(__name__)


def _format_error_message_with_details(response_json: dict) -> str:
    """Format error message with additional error details if available.

    Parameters
    ==========
    response_json : dict
        The JSON response containing error information

    Returns
    =======
    str
        Formatted error message with additional details
    """

    if "additional_information" in response_json and "errors" in response_json["additional_information"]:
        error_message = "Error details:\n"
        errors = response_json["additional_information"]["errors"]

        for i, error in enumerate(errors):
            error_data = ast.literal_eval(error)
            lines = []
            if "loc" in error_data:
                location = " -> ".join(str(x) for x in error_data["loc"])
                lines.append(f"- Field: {location}")
            if "type" in error_data:
                lines.append(f"  Type: {error_data['type']}")
            if "msg" in error_data:
                lines.append(f"  Message: {error_data['msg']}")
            if "input" in error_data:
                lines.append(f"  Invalid Input: {error_data['input']}")
            error_message += "\n".join(lines)
            if i < len(errors) - 1:
                error_message += "\n\n"

        return error_message
    else:
        return response_json["error_message"]


class ResultStatusEnum(str, enum.Enum):
    SUCCESS = "success"
    ERROR = "error"


def maybe_raise(
        response: httpx.Response,
        expected: t.Union[int, t.Tuple[int, int]] = (200, 299),
        msg: t.Optional[str] = None
) -> httpx.Response:
    """Verify response status and raise an HTTPError if got unexpected status code.

    Parameters
    ==========
    response : Response
        http response instance
    expected : Union[int, Tuple[int, int]] , default (200, 299)
        HTTP status code that is expected to receive
    msg : Optional[str] , default None
        error message to show in case of unexpected status code,
        next template parameters available:
        - status (HTTP status code)
        - reason (HTTP reason message)
        - url (request url)
        - body (response payload if available)
        - error (default error message that will include all previous parameters)

    Returns
    =======
    Response
    """
    status = response.status_code
    url = response.url
    reason = response.content

    error_template = "Error: {status}: {body}"
    client_error_template = "{status} Client Error: {body}"

    server_error_template = (
        "{status} Server Internal Error: {body}"
        "Please, reach Deepchecks support team for more information about this problem.\n"
    )

    def select_template(status):
        if 400 <= status <= 499:
            return client_error_template
        elif 500 <= status <= 599:
            return server_error_template
        else:
            return error_template

    def select_exception(status):
        if 400 <= status <= 499:
            if status == 402:
                return UsageLimitExceeded
            return DeepchecksLLMBadRequestError
        elif 500 <= status <= 599:
            return DeepchecksLLMServerError
        else:
            return BaseDeepchecksLLMAPIError

    def process_body():
        try:
            response_json = response.json()
            if "error_message" in response_json:
                return _format_error_message_with_details(response_json)
            return json.dumps(response.json(), indent=3)
        except JSONDecodeError:
            return

    if isinstance(expected, int) and status != expected:
        body = process_body()
        error = select_template(status).format(status=status, reason=reason, url=url, body=body)
        exception_class = select_exception(status)
        raise exception_class(
            error if msg is None else msg.format(
                status=status,
                body=body,
                error=error
            ),
            request=response.request,
            response=response
        )

    if isinstance(expected, (tuple, list)) and not expected[0] <= status <= expected[1]:
        body = process_body()
        error = select_template(status).format(status=status, reason=reason, url=url, body=body)
        exception_class = select_exception(status)
        raise exception_class(
            error if msg is None else msg.format(
                status=status,
                reason=reason,
                url=url,
                body=body,
                error=error
            ),
            request=response.request,
            response=response

        )
    return response


def null_log_filter(record):  # pylint: disable=unused-argument
    return False


class HandleExceptions:
    def __init__(self, logger_: logging.Logger, return_self: bool = False):
        self.logger = logger_
        self.return_self = return_self

    def __call__(self, original_func: t.Callable[P, T]) -> t.Callable[P, T]:
        return self.wraps(original_func)

    def wraps(self, original_func: t.Callable[..., t.Any]):
        @functools.wraps(original_func)
        def wrapper(dc_client_instance, *args, **kwargs):
            try:
                result = original_func(dc_client_instance, *args, **kwargs)
            except Exception as ex:  # pylint: disable=broad-exception-caught
                self.logger.error("Deepchecks SDK encountered a problem in %s: %s", original_func.__name__, str(ex))
                if not dc_client_instance.silent_mode:
                    raise ex
                if self.return_self:
                    return dc_client_instance
                return None
            if self.return_self:
                return dc_client_instance
            return result

        return wrapper


class HandleGeneratorExceptions(HandleExceptions):
    def wraps(self, original_func: t.Callable[..., GeneratorType]):
        @functools.wraps(original_func)
        def wrapper(dc_client_instance, *args, **kwargs):
            try:
                gen = original_func(dc_client_instance, *args, **kwargs)
                yield from gen
            except Exception as ex:  # pylint: disable=broad-exception-caught
                self.logger.error("Deepchecks SDK encountered a problem in %s: %s", original_func.__name__, str(ex))
                if not dc_client_instance.silent_mode:
                    raise ex

        return wrapper


def get_timestamp(time: t.Union[datetime, int, None]):
    return int(time.timestamp()) if isinstance(time, datetime) else time


E = t.TypeVar("E", bound=enum.Enum)


def convert_to_enum(value: t.Any, enum_type: t.Type[E]) -> E:
    if isinstance(value, enum_type):
        return value

    by_value = {it.value: it for it in enum_type}
    by_name = {it.name: it for it in enum_type}

    if value in by_name:
        return by_name[value]
    if value in by_value:
        return by_value[value]

    raise TypeError(f'Cannot convert given value "{value}" into "{enum_type.__name__}" enum instance')


def safe_list_eval(s: t.Union[t.List[str], str]) -> t.List[str]:
    if isinstance(s, list):
        return s
    try:
        res = ast.literal_eval(s)
        if isinstance(res, list):
            return [str(i) for i in res]
    except (SyntaxError, ValueError):
        pass
    return [s]


def check_topic(topic: t.Union[str, None]):
    if topic and len(topic) > MAX_TOPIC_LENGTH:
        logger.warning(f"Topic {topic} is too long and will be truncated to {MAX_TOPIC_LENGTH} during interaction upload")


@functools.lru_cache(maxsize=128)
def get_application(api_instance: API, app_name: str) -> dict:
    """Get application by name with caching.

    This method is cached to avoid repeated API calls for the same application.

    .. note::
        The cache persists for the lifetime of the process. If applications are
        created, renamed, or deleted during execution, the cache will not reflect
        these changes until the process restarts or the cache is cleared using
        ``get_application.cache_clear()``.

    Parameters
    ==========
    api_instance : API
        The API instance to use for making requests
    app_name : str
        The application name

    Returns
    =======
    dict
        The application data

    Raises
    ======
    DeepchecksLLMClientError
        If the application is not found
    """
    app = api_instance.get_application(app_name)
    if not app:
        raise DeepchecksLLMClientError(f"Application '{app_name}' not found")
    return app


@functools.lru_cache(maxsize=128)
def get_application_version_id(api_instance: API, app_name: str, version_name: str) -> int:
    """Get application_version_id from app_name and version_name.

    This method is cached to avoid repeated API calls for the same app/version combination.
    If the version doesn't exist, a DeepchecksLLMClientError will be raised.

    .. note::
        The cache persists for the lifetime of the process. If applications or versions
        are created, renamed, or deleted during execution, the cache will not reflect
        these changes until the process restarts or the cache is cleared using
        ``get_application_version_id.cache_clear()``.

    Parameters
    ==========
    api_instance : API
        The API instance to use for making requests
    app_name : str
        The application name
    version_name : str
        The version name

    Returns
    =======
    int
        The application version ID

    Raises
    ======
    DeepchecksLLMClientError
        If the application or version is not found
    """
    app = get_application(api_instance, app_name)

    version = next((v for v in app.get("versions", []) if v.get("name") == version_name), None)
    if version is None:
        raise DeepchecksLLMClientError(
            f"Could not find version '{version_name}' in application '{app_name}'"
        )

    return version["id"]


@functools.lru_cache(maxsize=128)
def get_or_create_application_version_id(api_instance: API, app_name: str, version_name: str) -> int:
    """Get or create application_version_id from app_name and version_name.

    This method is cached to avoid repeated API calls for the same app/version combination.
    If the version doesn't exist, it will be created automatically.

    .. note::
        The cache persists for the lifetime of the process. If applications or versions
        are created, renamed, or deleted during execution, the cache will not reflect
        these changes until the process restarts or the cache is cleared using
        ``get_application_version_id.cache_clear()``.

    Parameters
    ==========
    api_instance : API
        The API instance to use for making requests
    app_name : str
        The application name
    version_name : str
        The version name

    Returns
    =======
    int
        The application version ID

    Raises
    ======
    DeepchecksLLMClientError
        If the application is not found
    """
    app = get_application(api_instance, app_name)

    version = next((v for v in app.get("versions", []) if v.get("name") == version_name), None)
    if version is None:
        # Create the version if it doesn't exist
        version = api_instance.create_application_version(
            application_id=app["id"],
            version_name=version_name,
        )

        # Clear caches so subsequent calls get the new version
        get_application.cache_clear()
        get_application_version_id.cache_clear()

    return version["id"]


def paginate_all(
    fetch_fn: t.Callable[[int, int], t.List[t.Any]],
    limit: int = 100
) -> t.List[t.Any]:
    """Generic pagination helper that fetches all items.

    Parameters
    ==========
    fetch_fn : Callable[[int, int], List[Any]]
        Function that takes (offset, limit) and returns a list of items
    limit : int, default 100
        Number of items to fetch per page

    Returns
    =======
    List[Any]
        All items from all pages
    """
    all_items = []
    offset = 0

    while True:
        items = fetch_fn(offset, limit)
        if not items:
            break
        all_items.extend(items)
        offset += limit

    return all_items
