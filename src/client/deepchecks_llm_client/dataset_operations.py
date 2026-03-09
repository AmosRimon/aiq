"""High-level dataset operations including running datasets on deployments."""
import typing as t
from datetime import datetime

from deepchecks_llm_client.data_types import Dataset, DatasetRunResult, DatasetSample
from deepchecks_llm_client.dataset_runner import DatasetRunner
from deepchecks_llm_client.exceptions import DeepchecksLLMClientError
from deepchecks_llm_client.notebook_widgets import create_interactive_results_widget, create_live_progress_widget

if t.TYPE_CHECKING:
    from deepchecks_llm_client.client import DeepchecksLLMClient


def _setup_progress_widget(
    samples: t.List[DatasetSample],
    show_progress: bool,
    context_message: str
) -> t.Optional[t.Dict[str, t.Any]]:
    """Setup live progress widget or print fallback message.

    Parameters
    ----------
    samples : list of DatasetSample
        Samples to track progress for
    show_progress : bool
        Whether to attempt showing progress
    context_message : str
        Message to print if widget is not available

    Returns
    -------
    dict or None
        Widget dictionary with 'update' callback, or None if not available
    """
    live_widget = None
    if show_progress:
        live_widget = create_live_progress_widget(samples)
        if not live_widget:
            print(context_message)
    return live_widget


def _print_text_summary(result: DatasetRunResult, is_retry: bool) -> None:
    """Print text summary of dataset run results.

    Parameters
    ----------
    result : DatasetRunResult
        Results to display
    is_retry : bool
        Whether this was a retry operation
    """
    if is_retry:
        print("\nRetry completed:")
    else:
        print("\nDataset run completed:")

    print(f"  Dataset: {result.dataset_name}")
    print(f"  Deployment: {result.deployment_name}")

    if is_retry:
        print(f"  Total samples retried: {result.total_samples}")
        print(f"  Successful: {result.successful_samples}")
        print(f"  Still failed: {result.failed_samples}")
    else:
        print(f"  Total samples: {result.total_samples}")
        print(f"  Successful: {result.successful_samples}")
        print(f"  Failed: {result.failed_samples}")

    print(f"  Success rate: {result.success_rate:.1f}%")
    print(f"  Duration: {result.duration_seconds:.2f}s")

    if result.failed_samples > 0:
        if is_retry:
            print("\nTo retry again, use: await result.rerun_failed()")
        else:
            print("\nTo re-run only failed samples, use: await result.rerun_failed()")


def _display_results(
    result: DatasetRunResult,
    show_progress: bool,
    live_widget: t.Optional[t.Dict[str, t.Any]],
    is_retry: bool = False
) -> None:
    """Display results using interactive widget or text summary.

    Parameters
    ----------
    result : DatasetRunResult
        Results to display
    show_progress : bool
        Whether to show interactive results
    live_widget : dict or None
        The live widget that was used during execution, if any
    is_retry : bool, default=False
        Whether this was a retry operation
    """
    if show_progress and not live_widget:
        widget = create_interactive_results_widget(result)
        if not widget:
            _print_text_summary(result, is_retry)


async def execute_app(
    client: 'DeepchecksLLMClient',
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
    >>> result = await execute_app(client, "my-app", "test-dataset", "prod-deployment")

    Parameters
    ----------
    client : DeepchecksLLMClient
        The client instance to use for API calls
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
    >>> result = await execute_app(client, "my-app", "test-dataset", "prod-deployment")
    >>> print(f"Success rate: {result.success_rate:.1f}%")
    >>> print(f"Failed samples: {result.failed_samples}")
    >>> # Re-run only failed samples
    >>> if result.failed_samples > 0:
    ...     for failed in result.failed_results:
    ...         print(f"Sample {failed.sample_id} failed: {failed.error_message}")
    """
    # Get dataset and deployment
    dataset = client.get_dataset(app_name, dataset_name)
    if not dataset:
        raise DeepchecksLLMClientError(f"Dataset '{dataset_name}' not found")

    deployment = client.get_deployment(app_name, deployment_name)
    if not deployment:
        raise DeepchecksLLMClientError(f"Deployment '{deployment_name}' not found")

    # Get all samples
    samples = client.get_all_dataset_samples(app_name, dataset_name)
    if not samples:
        raise DeepchecksLLMClientError(f"Dataset '{dataset_name}' has no samples")

    # Create runner
    runner = DatasetRunner(
        deployment=deployment,
        app_name=app_name,
        version_name=version_name,
        env_type=env_type,
        additional_headers=additional_headers,
        verify_ssl=verify_ssl,
        api=client.api,
    )

    # Setup live widget if in notebook
    live_widget = _setup_progress_widget(
        samples,
        show_progress,
        f"\nRunning dataset '{dataset_name}' on deployment '{deployment_name}'...\n"
        f"Total samples: {len(samples)}"
    )

    # Run the dataset asynchronously
    result = await runner.execute_app(
        dataset,
        samples,
        progress_callback=progress_callback,
        live_widget_update=sample_callback if sample_callback else (live_widget['update'] if live_widget else None),
    )

    # Show interactive results widget if available (only if we didn't show live widget)
    _display_results(result, show_progress, live_widget, is_retry=False)

    return result


async def rerun_failed_samples(
    previous_result: DatasetRunResult,
    show_progress: bool = True,
) -> DatasetRunResult:
    """Re-run only the failed samples from a previous dataset run.

    Parameters
    ----------
    previous_result : DatasetRunResult
        The result from a previous dataset run containing failed samples
    show_progress : bool, default=True
        Whether to show a live progress widget (if in notebook with ipywidgets installed)

    Returns
    -------
    DatasetRunResult
        Results of re-running the failed samples

    Examples
    --------
    >>> result = await execute_app(client, "my-app", "test-dataset", "prod-deployment")
    >>> if result.failed_samples > 0:
    ...     retry_result = await result.rerun_failed()
    ...     print(f"Retry success rate: {retry_result.success_rate:.1f}%")
    """
    # Check if there are any failed samples
    if previous_result.failed_samples == 0:
        raise DeepchecksLLMClientError("No failed samples to re-run in the provided result")

    # Check if we have the required context
    if not previous_result._deployment:   # pylint: disable=protected-access
        raise DeepchecksLLMClientError("Cannot rerun - deployment config not available in result. This may be from an older SDK version.")

    deployment = previous_result._deployment  # pylint: disable=protected-access

    # Convert failed results back to DatasetSample objects
    failed_samples = []
    for failed_result in previous_result.failed_results:
        sample = DatasetSample(
            id=failed_result.sample_id,
            dataset_id=0,  # Not needed for re-running
            input=failed_result.sample_input,
            output=failed_result.sample_output,
            sample_metadata=failed_result.sample_metadata,
            created_at=None,  # Not needed for re-running
            updated_at=None,  # Not needed for re-running
        )
        failed_samples.append(sample)

    # Create a minimal Dataset object
    dataset = Dataset(
        id=0,  # Not used in runner
        application_id=0,  # Not used in runner
        dataset_name=previous_result.dataset_name,
        samples_count=len(failed_samples),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Create runner with same config from previous result
    runner = DatasetRunner(
        deployment=deployment,
        app_name=previous_result.app_name,
        api=previous_result._api,  # pylint: disable=protected-access
        version_name=previous_result.version_name,
        env_type=previous_result.env_type,
        additional_headers=previous_result.additional_headers,
        verify_ssl=previous_result.verify_ssl,
    )

    # Setup live widget if in notebook
    live_widget = _setup_progress_widget(
        failed_samples,
        show_progress,
        f"\nRe-running {len(failed_samples)} failed samples from "
        f"'{previous_result.dataset_name}' on deployment '{previous_result.deployment_name}'..."
    )

    # Run the failed samples
    result = await runner.execute_app(
        dataset,
        failed_samples,
        progress_callback=None,
        live_widget_update=live_widget['update'] if live_widget else None,
    )

    # Show interactive results widget if available
    _display_results(result, show_progress, live_widget, is_retry=True)

    return result
