"""Admission and proxy behavior without peer-engine holds."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from core.inference.attached import arbiter
from routes import inference


def test_local_admission_failure_is_retryable(monkeypatch):
    monkeypatch.setattr(arbiter, "free_for_local", AsyncMock(return_value = arbiter.ArbiterResult(error = "oMLX is still loading")))
    with pytest.raises(HTTPException) as error:
        asyncio.run(inference._admit_attached_local_load())
    assert error.value.status_code == 503
    assert error.value.detail == "oMLX is still loading"
    assert error.value.headers == {"Retry-After": "15"}


def test_local_admission_success_needs_no_hold(monkeypatch):
    result = arbiter.ArbiterResult(acted = True, actions = ("Unloaded oMLX",))
    unload = AsyncMock(return_value = result)
    monkeypatch.setattr(arbiter, "free_for_local", unload)
    assert asyncio.run(inference._admit_attached_local_load()) == result
    unload.assert_awaited_once_with("local_load")


def test_omlx_provider_admission_refusal_is_retryable(monkeypatch):
    monkeypatch.setattr(arbiter, "before_omlx_use", AsyncMock(return_value = arbiter.ArbiterResult(error = "Studio is training")))
    with pytest.raises(HTTPException) as error:
        asyncio.run(inference._admit_attached_omlx_provider("omlx", None))
    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "15"}


def test_provider_admission_ignores_other_destinations(monkeypatch):
    gate = AsyncMock(return_value = arbiter.ArbiterResult(error = "never"))
    monkeypatch.setattr(arbiter, "before_omlx_use", gate)
    asyncio.run(inference._admit_attached_omlx_provider("openai", "https://api.openai.com/v1"))
    gate.assert_not_awaited()
