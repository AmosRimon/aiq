from httpx import HTTPStatusError


class DeepchecksLLMClientError(Exception):
    pass


class DeepchecksLLMClientVersionInNewerError(Exception):
    pass


class BaseDeepchecksLLMAPIError(HTTPStatusError):
    """general API exception"""


class DeepchecksLLMBadRequestError(BaseDeepchecksLLMAPIError):
    """Raised when API return 4xx code"""
    pass


class DeepchecksLLMServerError(BaseDeepchecksLLMAPIError):
    """Raised when API return 5xx code"""
    pass

class UsageLimitExceeded(BaseDeepchecksLLMAPIError):
    """Raised when API return 402 code"""
    pass
