# Client SDK Development Guide

## Overview

This is the Python client SDK for the Deepchecks LLM backend. Users install this package to interact with the backend API.

## Critical: Backward Compatibility

**IMPORTANT**: The backend must support SDK versions up to **3 versions older** than the current version.

### When Making Changes

When modifying the SDK or backend API:

1. **Backend API Changes**: Ensure backward compatibility with older SDK versions
   - Add new fields as optional, never remove existing fields
   - New endpoints are fine; changing/removing endpoints breaks compatibility
   - When changing request/response schemas, maintain support for old formats
   - If deprecating/changing an endpoint that exists in the SDK, note that you need to test it works with older SDK versions

2. **SDK Changes**: Can freely add new features that use new backend endpoints
   - The backend is always newer than or equal to the SDK version users have
   - You can assume the backend supports your SDK changes
   - If changing how the SDK calls existing endpoints, note that the backend must support the old calling pattern for 3+ versions

### Examples

**Good** - Backward compatible:
```python
# Backend adds optional field (old SDK can ignore it)
class Response:
    existing_field: str
    new_optional_field: str | None = None  # ✓ OK
```

**Bad** - Breaking change:
```python
# Backend removes field (old SDK will break)
class Response:
    # removed_field: str  # ✗ BREAKS old SDK versions
    existing_field: str
```

**Good** - New endpoint:
```python
# New endpoint doesn't break old SDK
@router.post("/v1/new-feature")  # ✓ OK - old SDK won't call it
```

**Bad** - Changed endpoint:
```python
# Changing existing endpoint breaks old SDK
@router.post("/v1/existing-endpoint")
def handler(new_required_param: str):  # ✗ BREAKS - new required param
    ...
```

## Project Structure

- [deepchecks_llm_client/](deepchecks_llm_client/) - Main package
  - [client.py](deepchecks_llm_client/client.py) - Core client implementation
  - [api.py](deepchecks_llm_client/api.py) - API wrapper
  - [data_types.py](deepchecks_llm_client/data_types.py) - Data models (keep in sync with backend)
  - [sdk/](deepchecks_llm_client/sdk/) - SDK helpers
  - [examples/](deepchecks_llm_client/examples/) - Usage examples

## Backend Integration

The client SDK communicates with the backend at [../backend](../backend).

When making changes:
- Check [../backend/deepchecks_llm_eval/api/](../backend/deepchecks_llm_eval/api/) for API endpoints
- Ensure data types in [data_types.py](deepchecks_llm_client/data_types.py) match backend [orm/data_types.py](../backend/deepchecks_llm_eval/orm/data_types.py)
- Backend documentation: [../backend/CLAUDE.md](../backend/CLAUDE.md)
