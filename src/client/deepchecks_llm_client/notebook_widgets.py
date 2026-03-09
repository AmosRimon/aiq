"""Optional notebook widgets for progress visualization.

This module provides widgets for Jupyter notebooks when ipywidgets is installed.
It's completely optional and the SDK works without it.
"""
import json
import typing as t

# Optional imports for notebook widgets
try:
    from IPython.display import display
    from ipywidgets import HTML, Accordion, Button, HBox, IntProgress, VBox
    WIDGETS_AVAILABLE = True
except ImportError:
    display = None
    HTML = Button = HBox = Accordion = VBox = IntProgress = None
    WIDGETS_AVAILABLE = False

# HTML style constants
_STYLE_PRE_CODE = "background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto;"
_STYLE_SUCCESS_BG = "#d4edda"
_STYLE_FAILED_BG = "#f8d7da"
_STYLE_PENDING_BG = "#f0f0f0"
_STYLE_IN_PROGRESS_BG = "#fff3cd"
_STYLE_INFO_BG = "#e7f3ff"
_STYLE_ERROR_BG = "#ffe6e6"
_STYLE_TABLE_CELL = "padding: 8px; border-bottom: 1px solid #ddd;"
_STYLE_TABLE_CELL_LAST = "padding: 8px;"


def _notebook_widget_get_status_icon(status: str) -> str:
    """Get icon for a given status in notebook widgets."""
    return {
        'pending': '⏸',
        'in_progress': '⏳',
        'completed': '✓',
        'failed': '✗',
    }.get(status, '?')


def _notebook_widget_get_status_display(status: str) -> tuple[str, str, str]:
    """Get display properties for a status: (icon, color, text).

    Parameters
    ----------
    status : str
        Status string: 'pending', 'in_progress', 'completed', or 'failed'

    Returns
    -------
    tuple[str, str, str]
        A tuple of (icon, background_color, status_text)
    """
    if status == 'pending':
        return '⏸', _STYLE_PENDING_BG, 'PENDING'
    if status == 'in_progress':
        return '⏳', _STYLE_IN_PROGRESS_BG, 'IN PROGRESS'
    if status == 'completed':
        return '✓', _STYLE_SUCCESS_BG, 'SUCCESS'
    # failed
    return '✗', _STYLE_FAILED_BG, 'FAILED'


def _notebook_widget_format_json(data: t.Any) -> str:
    """Format data as JSON in HTML for notebook widgets."""
    if data is None:
        return "<em>None</em>"
    try:
        return f"<pre style='{_STYLE_PRE_CODE}'>{json.dumps(data, indent=2)}</pre>"
    except (TypeError, ValueError):
        return f"<pre style='{_STYLE_PRE_CODE}'>{str(data)}</pre>"


def _notebook_widget_table_row(label: str, value: t.Any, bg_color: str = 'inherit', is_last: bool = False) -> str:
    """Generate a table row for summary tables.

    Parameters
    ----------
    label : str
        The label text
    value : Any
        The value to display
    bg_color : str, default='inherit'
        Background color for the row
    is_last : bool, default=False
        Whether this is the last row (no bottom border)

    Returns
    -------
    str
        HTML string for the table row
    """
    cell_style = _STYLE_TABLE_CELL_LAST if is_last else _STYLE_TABLE_CELL
    return f"""
                <tr style="background-color: {bg_color};">
                    <td style="{cell_style}"><strong>{label}</strong></td>
                    <td style="{cell_style}">{value}</td>
                </tr>"""


def _create_pagination_controls(total_pages: int):
    """Create pagination control widgets (Previous/Next buttons and page info).

    Parameters
    ----------
    total_pages : int
        Total number of pages

    Returns
    -------
    dict
        Dictionary containing 'prev_button', 'next_button', 'page_info', and 'hbox'
    """
    if not WIDGETS_AVAILABLE:
        return None

    prev_button = Button(description='← Previous', disabled=True)
    next_button = Button(description='Next →', disabled=total_pages <= 1)
    page_info = HTML(value=f"<div style='text-align: center; padding: 10px;'>Page 1 of {total_pages}</div>")
    pagination_hbox = HBox([prev_button, page_info, next_button])

    return {
        'prev_button': prev_button,
        'next_button': next_button,
        'page_info': page_info,
        'hbox': pagination_hbox,
    }


def _update_pagination_controls(controls: dict, current_page: int, total_pages: int):
    """Update pagination control states.

    Parameters
    ----------
    controls : dict
        Dictionary containing pagination controls from _create_pagination_controls()
    current_page : int
        Current page index (0-based)
    total_pages : int
        Total number of pages
    """
    controls['prev_button'].disabled = current_page == 0
    controls['next_button'].disabled = current_page >= total_pages - 1
    controls['page_info'].value = f"<div style='text-align: center; padding: 10px;'>Page {current_page + 1} of {total_pages}</div>"


def _notebook_widget_generate_sample_html(
    status: str,
    sample_input: t.Any = None,
    output: t.Any = None,
    error: str = None,
    duration: float = 0.0,
    retries: int = 0,
    sample_metadata: t.Any = None,
    status_code: t.Optional[int] = None,
) -> str:
    """Generate HTML for a sample's details.

    Parameters
    ----------
    status : str
        Status: 'pending', 'in_progress', 'completed', or 'failed'
    sample_input : Any, optional
        The sample input data
    output : Any, optional
        The deployment response (for completed samples)
    error : str, optional
        Error message (for failed samples)
    duration : float, default=0.0
        Duration in seconds
    retries : int, default=0
        Number of retries
    sample_metadata : Any, optional
        Additional metadata
    status_code : int, optional
        HTTP status code from the deployment response

    Returns
    -------
    str
        HTML string for the sample
    """
    icon, status_color, status_text = _notebook_widget_get_status_display(status)

    html = f"""
    <div style="padding: 10px;">
        <div style="background-color: {status_color}; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <h4 style="margin: 0;">{icon} Status: {status_text}</h4>
        </div>
    """

    if sample_input is not None:
        html += f"""
        <div style="margin-bottom: 15px;">
            <h4 style="margin: 5px 0;">📥 Input:</h4>
            {_notebook_widget_format_json(sample_input)}
        </div>
        """

    if status == 'completed' and output is not None:
        html += f"""
        <div style="margin-bottom: 15px;">
            <h4 style="margin: 5px 0;">📤 Response:</h4>
            {_notebook_widget_format_json(output)}
        </div>
        <div style="background-color: {_STYLE_INFO_BG}; padding: 10px; border-radius: 5px;">
            <strong>⏱️ Duration:</strong> {duration:.3f}s<br>
            <strong>🔄 Retries:</strong> {retries}
        """
        if status_code is not None:
            html += f"<br><strong>📡 Status Code:</strong> <span style='color: #4caf50; font-weight: 600;'>{status_code}</span>"
        html += """
        </div>
        """
    elif status == 'failed' and error is not None:
        html += f"""
        <div style="background-color: {_STYLE_ERROR_BG}; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <h4 style="margin: 5px 0; color: #d32f2f;">❌ Error Details:</h4>
            <strong>Message:</strong> {error}
        </div>
        <div style="background-color: {_STYLE_IN_PROGRESS_BG}; padding: 10px; border-radius: 5px;">
            <strong>⏱️ Duration:</strong> {duration:.3f}s<br>
            <strong>🔄 Retries Attempted:</strong> {retries}
        """
        if status_code is not None:
            html += f"<br><strong>📡 Status Code:</strong> <span style='color: #f44336; font-weight: 600;'>{status_code}</span>"
        html += """
        </div>
        """
    elif status == 'pending':
        html += f"""
        <div style="background-color: {_STYLE_PENDING_BG}; padding: 10px; border-radius: 5px;">
            <em>Waiting to start...</em>
        </div>
        """
    elif status == 'in_progress':
        html += f"""
        <div style="background-color: {_STYLE_IN_PROGRESS_BG}; padding: 10px; border-radius: 5px;">
            <em>Processing...</em>
        </div>
        """

    if sample_metadata:
        html += f"""
        <div style="margin-top: 15px;">
            <h4 style="margin: 5px 0;">🏷️ Metadata:</h4>
            {_notebook_widget_format_json(sample_metadata)}
        </div>
        """

    html += "</div>"
    return html


def create_interactive_results_widget(result: t.Any, samples_per_page: int = 10):
    """Create an interactive results viewer with expandable sample details.

    Parameters
    ----------
    result : DatasetRunResult
        The result object to display
    samples_per_page : int, default=10
        Number of samples to show per page in the accordion

    Returns
    -------
    Widget or None
        Interactive widget if ipywidgets is available, None otherwise
    """
    if not WIDGETS_AVAILABLE:
        return None

    total_samples = result.total_samples
    total_pages = (total_samples + samples_per_page - 1) // samples_per_page

    # Create summary section
    failed_bg = _STYLE_FAILED_BG if result.failed_samples > 0 else 'inherit'
    summary_html = f"""
    <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 15px;">
        <h3 style="margin-top: 0;">📊 Dataset Run Results</h3>
        <table style="width: 100%; border-collapse: collapse;">{
            _notebook_widget_table_row('Dataset:', result.dataset_name)}{
            _notebook_widget_table_row('Deployment:', result.deployment_name)}{
            _notebook_widget_table_row('Total Samples:', result.total_samples)}{
            _notebook_widget_table_row('✓ Successful:', f'<strong>{result.successful_samples}</strong>', _STYLE_SUCCESS_BG)}{
            _notebook_widget_table_row('✗ Failed:', f'<strong>{result.failed_samples}</strong>', failed_bg)}{
            _notebook_widget_table_row('Success Rate:', f'{result.success_rate:.1f}%')}{
            _notebook_widget_table_row('Duration:', f'{result.duration_seconds:.2f}s', is_last=True)}
        </table>
    </div>
    """

    # State for pagination
    state = {
        'current_page': 0,
        'samples_per_page': samples_per_page,
        'total_pages': total_pages,
    }

    # Create widgets
    summary_widget = HTML(value=summary_html)
    accordion = Accordion(children=[])
    pagination = _create_pagination_controls(total_pages)

    def _update_accordion():
        """Update accordion to show current page of samples."""
        start_idx = state['current_page'] * state['samples_per_page']
        end_idx = min(start_idx + state['samples_per_page'], total_samples)
        page_samples = result.results[start_idx:end_idx]

        # Create accordion items
        accordion_items = []
        for sample in page_samples:
            status = 'completed' if sample.success else 'failed'
            error = f"Type: {sample.error_type}<br>Message: {sample.error_message}" if not sample.success else None

            sample_html = _notebook_widget_generate_sample_html(
                status=status,
                sample_input=sample.sample_input,
                output=sample.deployment_response if sample.success else None,
                error=error,
                duration=sample.duration_seconds,
                retries=sample.retries,
                sample_metadata=sample.sample_metadata,
                status_code=sample.status_code,
            )
            accordion_items.append(HTML(value=sample_html))

        accordion.children = accordion_items

        # Set titles
        for i, sample in enumerate(page_samples):
            status = "✓" if sample.success else "✗"
            accordion.set_title(i, f"{status} Sample {start_idx + i + 1} (ID: {sample.sample_id})")

        # Update pagination controls
        _update_pagination_controls(pagination, state['current_page'], state['total_pages'])

        # Initially all closed
        accordion.selected_index = None

    def _on_prev_click(_):
        if state['current_page'] > 0:
            state['current_page'] -= 1
            _update_accordion()

    def _on_next_click(_):
        if state['current_page'] < state['total_pages'] - 1:
            state['current_page'] += 1
            _update_accordion()

    pagination['prev_button'].on_click(_on_prev_click)
    pagination['next_button'].on_click(_on_next_click)

    # Initial render
    _update_accordion()

    # Add filter tip
    tip_html = """
    <div style="margin: 15px 0; padding: 10px; background-color: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 3px;">
        <strong>💡 Pro Tips:</strong><br>
        • Re-run failed samples: <code>await result.rerun_failed()</code><br>
        • Access only failed samples: <code>result.failed_results</code><br>
        • Access only successful samples: <code>result.successful_results</code><br>
        • Get sample by ID: <code>next(r for r in result.results if r.sample_id == 123)</code><br>
        • Click on each sample below to expand/collapse details
    </div>
    """
    tip_widget = HTML(value=tip_html)

    # Add header for accordion
    accordion_header = HTML(value="""
    <h3 style="margin: 20px 0 10px 0;">📋 Sample Results (click to expand)</h3>
    """)

    container = VBox([summary_widget, tip_widget, accordion_header, accordion, pagination['hbox']])
    display(container)
    return container



def create_live_progress_widget(samples: t.List, samples_per_page: int = 10):
    """Create a live progress tracker that updates as samples complete.

    Parameters
    ----------
    samples : list
        List of DatasetSample objects to track
    samples_per_page : int, default=10
        Number of samples to show per page in the accordion

    Returns
    -------
    dict or None
        Dictionary with 'container' (the widget), 'update' (function to update status),
        and 'state' (internal state) if ipywidgets is available, None otherwise
    """
    if not WIDGETS_AVAILABLE:
        return None

    total_samples = len(samples)

    # Initialize state for all samples using actual sample IDs
    state = {
        'samples': [
            {
                'sample_id': sample.id,
                'status': 'pending',  # pending, in_progress, completed, failed
                'input': None,
                'output': None,
                'error': None,
                'duration': 0.0,
                'retries': 0,
                'status_code': None,
                'sample_metadata': None,
            }
            for sample in samples
        ],
        'sample_id_to_idx': {sample.id: idx for idx, sample in enumerate(samples)},
        'current_page': 0,
        'samples_per_page': samples_per_page,
        'total_pages': (total_samples + samples_per_page - 1) // samples_per_page,
    }

    # Create widgets
    progress = IntProgress(
        value=0,
        min=0,
        max=total_samples,
        description='0/0',
        bar_style='info',
        orientation='horizontal',
        style={'description_width': 'initial'}
    )

    summary_html = HTML(value=_generate_summary_html(state, total_samples))
    accordion = Accordion(children=[])
    pagination = _create_pagination_controls(state['total_pages'])

    # Helper functions
    def _generate_sample_html(sample):
        return _notebook_widget_generate_sample_html(
            status=sample['status'],
            sample_input=sample['input'],
            output=sample['output'],
            error=sample['error'],
            duration=sample['duration'],
            retries=sample['retries'],
            sample_metadata=sample.get('sample_metadata'),
            status_code=sample.get('status_code'),
        )

    def _update_accordion():
        """Update accordion to show current page of samples."""
        start_idx = state['current_page'] * state['samples_per_page']
        end_idx = min(start_idx + state['samples_per_page'], total_samples)
        page_samples = state['samples'][start_idx:end_idx]

        # Create accordion items
        accordion_items = [HTML(value=_generate_sample_html(sample)) for sample in page_samples]
        accordion.children = accordion_items

        # Set titles
        for i, sample in enumerate(page_samples):
            status_icon = _notebook_widget_get_status_icon(sample['status'])
            accordion.set_title(i, f"{status_icon} Sample {sample['sample_id']}")

        # Update pagination controls
        _update_pagination_controls(pagination, state['current_page'], state['total_pages'])

        # Close all accordions
        accordion.selected_index = None

    def _on_prev_click(_):
        if state['current_page'] > 0:
            state['current_page'] -= 1
            _update_accordion()

    def _on_next_click(_):
        if state['current_page'] < state['total_pages'] - 1:
            state['current_page'] += 1
            _update_accordion()

    pagination['prev_button'].on_click(_on_prev_click)
    pagination['next_button'].on_click(_on_next_click)

    def update(sample_id, status='pending', **kwargs):
        """Update a sample's status and details.

        Parameters
        ----------
        sample_id : int
            The sample ID from the dataset
        status : str
            New status: 'pending', 'in_progress', 'completed', 'failed'
        **kwargs
            Additional fields: input, output, error, duration, retries, status_code, sample_metadata
        """
        if sample_id not in state['sample_id_to_idx']:
            return

        idx = state['sample_id_to_idx'][sample_id]
        sample = state['samples'][idx]
        sample['status'] = status

        # Update optional fields
        for key in ['input', 'output', 'error', 'duration', 'retries', 'status_code', 'sample_metadata']:
            if key in kwargs:
                sample[key] = kwargs[key]

        # Update progress bar
        completed = sum(1 for s in state['samples'] if s['status'] in ('completed', 'failed'))
        progress.value = completed
        progress.description = f"{completed}/{total_samples}"

        if completed == total_samples:
            progress.bar_style = 'success'

        # Update summary
        summary_html.value = _generate_summary_html(state, total_samples)

        # If sample is on current page, update accordion
        sample_page = idx // state['samples_per_page']
        if sample_page == state['current_page']:
            _update_accordion()

    # Initial render
    _update_accordion()

    # Create container
    accordion_header = HTML(value="<h3 style='margin: 20px 0 10px 0;'>📋 Sample Results (click to expand)</h3>")
    container = VBox([summary_html, progress, accordion_header, accordion, pagination['hbox']])

    display(container)

    return {
        'container': container,
        'update': update,
        'state': state,
    }


def _generate_summary_html(state, total_samples):
    """Generate summary HTML for live progress widget."""
    pending = sum(1 for s in state['samples'] if s['status'] == 'pending')
    in_progress = sum(1 for s in state['samples'] if s['status'] == 'in_progress')
    completed = sum(1 for s in state['samples'] if s['status'] == 'completed')
    failed = sum(1 for s in state['samples'] if s['status'] == 'failed')

    failed_bg = _STYLE_FAILED_BG if failed > 0 else 'inherit'
    return f"""
    <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 15px;">
        <h3 style="margin-top: 0;">📊 Live Progress</h3>
        <table style="width: 100%; border-collapse: collapse;">{
            _notebook_widget_table_row('Total Samples:', total_samples)}{
            _notebook_widget_table_row('⏸ Pending:', pending, _STYLE_PENDING_BG)}{
            _notebook_widget_table_row('⏳ In Progress:', in_progress, _STYLE_IN_PROGRESS_BG)}{
            _notebook_widget_table_row('✓ Completed:', completed, _STYLE_SUCCESS_BG)}{
            _notebook_widget_table_row('✗ Failed:', failed, failed_bg, is_last=True)}
        </table>
    </div>
    """
