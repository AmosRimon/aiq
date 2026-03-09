"""Module for running datasets on deployments with parallel execution."""
import asyncio
import time
import typing as t
import uuid

import httpx

from deepchecks_llm_client.data_types import Dataset, DatasetRunResult, DatasetRunSampleResult, DatasetSample, DatasetType, Deployment

if t.TYPE_CHECKING:
    from deepchecks_llm_client.api import API


class DatasetRunner:
    """Runner for executing dataset samples against a deployment with parallel processing."""

    def __init__(
        self,
        deployment: Deployment,
        app_name: str,
        api: 'API',
        version_name: t.Optional[str] = None,
        env_type: t.Optional[str] = None,
        additional_headers: t.Optional[t.Dict[str, str]] = None,
        verify_ssl: bool = True,
    ):
        """Initialize the dataset runner.

        Parameters
        ----------
        deployment : Deployment
            The deployment configuration to use for running the dataset
        app_name : str
            Application name for dc_fields
        api : API
            Backend API client, used for multi-turn execution
        version_name : str, optional
            Version name for dc_fields
        env_type : str, optional
            Environment type for dc_fields
        additional_headers : dict, optional
            Additional headers to include in requests (beyond those in deployment config)
        verify_ssl : bool, default=True
            Whether to verify SSL certificates for HTTPS requests
        """
        self.deployment = deployment
        self.app_name = app_name
        self.version_name = version_name
        self.env_type = env_type
        self.additional_headers = additional_headers or {}
        self.verify_ssl = verify_ssl
        self.api = api

    def _build_headers(self) -> t.Dict[str, str]:
        """Build the complete headers dictionary from deployment and additional headers."""
        headers = {h.name: h.value for h in self.deployment.headers}
        headers.update(self.additional_headers)
        return headers

    def _build_dc_fields(self) -> t.Dict[str, t.Any]:
        """Build dc_fields dict for request bodies."""
        dc_fields: t.Dict[str, t.Any] = {"app_name": self.app_name}
        if self.version_name:
            dc_fields["version_name"] = self.version_name
        if self.env_type:
            dc_fields["env_type"] = self.env_type
        return dc_fields

    async def _call_deployment(
        self,
        client: httpx.AsyncClient,
        content: t.Any,
        session_id: t.Optional[str] = None,
    ) -> t.Dict[str, t.Any]:
        """Send a single request to the deployment URL and return the parsed response."""
        dc_fields = self._build_dc_fields()
        if session_id:
            dc_fields["session_id"] = session_id
        request_body = {
            "dc_fields": dc_fields,
            "content": content,
        }
        response = await client.post(
            self.deployment.deployment_url,
            json=request_body,
            timeout=self.deployment.timeout,
        )
        response.raise_for_status()
        try:
            return response.json()
        except (ValueError, TypeError):
            return {"text": response.text}

    async def _run_sample(
        self,
        sample: DatasetSample,
        semaphore: asyncio.Semaphore,
        work_fn: t.Callable[[], t.Awaitable[t.Tuple[t.Any, int]]],
        live_widget_update: t.Optional[t.Callable] = None,
    ) -> DatasetRunSampleResult:
        """Run a sample with shared semaphore, timing, error handling, and widget updates.

        Parameters
        ----------
        sample : DatasetSample
            The dataset sample being run
        semaphore : asyncio.Semaphore
            Semaphore to limit concurrent requests
        work_fn : async callable
            Async function that performs the actual work and returns (response_data, retries).
            May raise httpx exceptions on failure.
        live_widget_update : callable, optional
            Callback for live widget updates
        """
        async with semaphore:
            if live_widget_update:
                live_widget_update(sample.id, status='in_progress', input=sample.input)

            start_time = time.time()

            try:
                response_data, retries = await work_fn()
                duration = time.time() - start_time

                result = DatasetRunSampleResult(
                    sample_id=sample.id,
                    sample_input=sample.input,
                    sample_output=sample.output,
                    sample_metadata=sample.sample_metadata,
                    success=True,
                    deployment_response=response_data,
                    duration_seconds=duration,
                    retries=retries,
                    status_code=200,
                )

                if live_widget_update:
                    live_widget_update(
                        sample.id,
                        status='completed',
                        input=sample.input,
                        output=response_data,
                        duration=duration,
                        retries=retries,
                        status_code=200,
                        sample_metadata=sample.sample_metadata,
                    )

                return result

            except httpx.HTTPStatusError as e:
                duration = time.time() - start_time
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                return self._handle_error(
                    sample, live_widget_update, error_msg, "HTTPStatusError",
                    duration, status_code=e.response.status_code,
                    retries=self.deployment.max_retries,
                )

            except httpx.TimeoutException:
                duration = time.time() - start_time
                error_msg = f"Request timed out after {self.deployment.timeout}s"
                return self._handle_error(
                    sample, live_widget_update, error_msg, "TimeoutException", duration,
                    retries=self.deployment.max_retries,
                )

            except Exception as e:  # pylint: disable=broad-exception-caught
                duration = time.time() - start_time
                return self._handle_error(
                    sample, live_widget_update, str(e), type(e).__name__, duration,
                    retries=self.deployment.max_retries,
                )

    def _handle_error(
        self,
        sample: DatasetSample,
        live_widget_update: t.Optional[t.Callable],
        error_msg: str,
        error_type: str,
        duration: float,
        status_code: t.Optional[int] = None,
        retries: int = 0,
    ) -> DatasetRunSampleResult:
        """Create error result and notify widget."""
        result = DatasetRunSampleResult(
            sample_id=sample.id,
            sample_input=sample.input,
            sample_output=sample.output,
            sample_metadata=sample.sample_metadata,
            success=False,
            error_message=error_msg,
            error_type=error_type,
            duration_seconds=duration,
            retries=retries,
            status_code=status_code,
        )
        if live_widget_update:
            live_widget_update(
                sample.id,
                status='failed',
                input=sample.input,
                error=error_msg,
                duration=duration,
                retries=retries,
                status_code=status_code,
                sample_metadata=sample.sample_metadata,
            )
        return result

    async def _single_turn_work(
        self,
        client: httpx.AsyncClient,
        sample: DatasetSample,
    ) -> t.Tuple[t.Any, int]:
        """Execute a single-turn sample with retries. Returns (response_data, retries)."""
        last_exception = None
        for attempt in range(self.deployment.max_retries + 1):
            try:
                response_data = await self._call_deployment(client, sample.input)
                return response_data, attempt
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_exception = e
                if attempt == self.deployment.max_retries:
                    raise last_exception from e  # pylint: disable=raising-bad-type

    async def _multi_turn_work(
        self,
        client: httpx.AsyncClient,
        sample: DatasetSample,
    ) -> t.Tuple[t.Any, int]:
        """Execute a multi-turn conversation. Returns (conversation_list, 0)."""
        session_id = str(uuid.uuid4())
        conversation: t.List[t.Dict[str, t.Any]] = []

        # Turn 0: send the original sample input to the deployment
        turn_0_response = await self._call_deployment(client, sample.input, session_id=session_id)
        conversation.append({"role": "user", "content": sample.input})
        conversation.append({"role": "assistant", "content": turn_0_response})

        # Subsequent turns — backend controls when to stop via is_finished
        turn = 0
        while True:
            turn += 1
            # Call simulate-human to generate next human input
            simulated = await self.api.async_simulate_human(
                application_id=self.deployment.application_id,
                turns_amount=turn,
                input_data=conversation,
            )

            if simulated.get("is_finished", False):
                break

            output = simulated.get("output")
            if not output or "message" not in output:
                raise ValueError(f"Invalid simulate_human response: missing 'output.message' in {simulated}")
            human_input = output["message"]

            # Send simulated human input to deployment
            ai_response = await self._call_deployment(client, human_input, session_id=session_id)
            conversation.append({"role": "user", "content": human_input})
            conversation.append({"role": "assistant", "content": ai_response})

        return conversation, 0

    async def execute_app(
        self,
        dataset: Dataset,
        samples: t.List[DatasetSample],
        progress_callback: t.Optional[t.Callable[[int, int], None]] = None,
        live_widget_update: t.Optional[t.Callable] = None,
    ) -> DatasetRunResult:
        """Run the dataset asynchronously with parallel processing.

        Parameters
        ----------
        dataset : Dataset
            The dataset being run
        samples : list of DatasetSample
            The samples to run
        progress_callback : callable, optional
            Callback function to report progress, receives (completed, total) as arguments
        live_widget_update : callable, optional
            Callback for live widget updates

        Returns
        -------
        DatasetRunResult
            The aggregated results of running all samples
        """
        start_time = time.time()

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.deployment.max_concurrent)

        # Build headers
        headers = self._build_headers()

        is_multi_turn = dataset.dataset_type == DatasetType.MULTI_TURN

        # Create async HTTP client
        async with httpx.AsyncClient(
            headers=headers,
            verify=self.verify_ssl,
            follow_redirects=True,
        ) as client:
            def make_work_fn(s):  # pylint: disable=cell-var-from-loop
                if is_multi_turn:
                    return lambda: self._multi_turn_work(client, s)  # pylint: disable=non-awaited-async
                return lambda: self._single_turn_work(client, s)  # pylint: disable=non-awaited-async

            # Create coroutines for all samples (awaited in as_completed loop below)
            coroutines = [
                self._run_sample(sample, semaphore, make_work_fn(sample), live_widget_update)  # pylint: disable=non-awaited-async
                for sample in samples
            ]

            # Run with progress tracking
            results: list[DatasetRunSampleResult] = []
            for coro in asyncio.as_completed(coroutines):
                result = await coro
                results.append(result)

                if progress_callback:
                    progress_callback(len(results), len(samples))

        # Calculate statistics
        duration = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return DatasetRunResult(
            dataset_name=dataset.dataset_name,
            deployment_name=self.deployment.deployment_name,
            total_samples=len(samples),
            successful_samples=successful,
            failed_samples=failed,
            duration_seconds=duration,
            results=results,
            app_name=self.app_name,
            version_name=self.version_name,
            env_type=self.env_type,
            additional_headers=self.additional_headers,
            verify_ssl=self.verify_ssl,
            _deployment=self.deployment,
            _api=self.api,
        )
