"""Regression tests for the reasoning-preserving DeepSeek integration."""

from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_deepseek import ChatDeepSeek

from backend.app.config import Settings
from backend.app.services.llm import (
    MissingDeepSeekApiKeyError,
    ReasoningPreservingChatDeepSeek,
    build_chat_model,
)

DUMMY_API_KEY = "temporary-test-not-a-credential"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"


def _settings(**overrides: Any) -> Settings:
    """Build isolated settings without reading the local .env file."""
    values: dict[str, Any] = {
        "deepseek_api_key": DUMMY_API_KEY,
        "deepseek_model": MODEL,
        "deepseek_base_url": BASE_URL,
        "deepseek_reasoning_effort": "low",
        **overrides,
    }
    return Settings(_env_file=None, **values)


def _model() -> ReasoningPreservingChatDeepSeek:
    return ReasoningPreservingChatDeepSeek(
        model=MODEL,
        api_key=DUMMY_API_KEY,
        base_url=BASE_URL,
    )


def _base_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=MODEL,
        api_key=DUMMY_API_KEY,
        base_url=BASE_URL,
    )


def _tool_call(call_id: str) -> dict[str, Any]:
    return {
        "name": "lookup_status",
        "args": {"reference": call_id},
        "id": call_id,
        "type": "tool_call",
    }


def _assistant_with_reasoning(
    reasoning_content: str,
    *tool_calls: dict[str, Any],
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=list(tool_calls),
        additional_kwargs={"reasoning_content": reasoning_content},
    )


def _outbound_messages(model: ChatDeepSeek, messages: list[Any]) -> list[dict]:
    return model._get_request_payload(messages)["messages"]


def test_reasoning_content_and_tool_call_are_preserved() -> None:
    call = _tool_call("call_1")
    messages = [
        HumanMessage(content="Check the status."),
        _assistant_with_reasoning("reasoning one", call),
        ToolMessage(content="approved", tool_call_id="call_1"),
    ]

    model = _model()
    payload = model._get_request_payload(messages)
    outbound = payload["messages"]
    base_payload = _base_model()._get_request_payload(messages)

    assert outbound[1]["reasoning_content"] == "reasoning one"
    assert outbound[1]["tool_calls"][0]["id"] == "call_1"
    assert outbound[1]["tool_calls"][0]["function"]["name"] == "lookup_status"
    assert outbound[2]["tool_call_id"] == "call_1"

    # All parent-generated top-level fields remain intact.
    assert set(payload) == set(base_payload)
    assert payload["model"] == base_payload["model"]
    assert payload["stream"] == base_payload["stream"]


def test_assistant_without_reasoning_is_unchanged() -> None:
    messages = [
        HumanMessage(content="Check the status."),
        AIMessage(content="", tool_calls=[_tool_call("call_1")]),
        ToolMessage(content="approved", tool_call_id="call_1"),
    ]

    base_messages = _base_model()._get_request_payload(messages)["messages"]
    outbound = _outbound_messages(_model(), messages)

    assert outbound == base_messages
    assert "reasoning_content" not in outbound[1]


def test_multiple_assistant_messages_keep_their_own_reasoning() -> None:
    messages = [
        HumanMessage(content="Start."),
        _assistant_with_reasoning("first reasoning", _tool_call("call_1")),
        ToolMessage(content="one", tool_call_id="call_1"),
        _assistant_with_reasoning("second reasoning", _tool_call("call_2")),
        ToolMessage(content="two", tool_call_id="call_2"),
    ]

    outbound = _outbound_messages(_model(), messages)
    assistants = [m for m in outbound if m["role"] == "assistant"]

    assert [m["reasoning_content"] for m in assistants] == [
        "first reasoning",
        "second reasoning",
    ]
    assert [m["tool_calls"][0]["id"] for m in assistants] == ["call_1", "call_2"]


def test_tool_call_ids_and_tool_results_stay_associated() -> None:
    assistant = _assistant_with_reasoning(
        "parallel reasoning",
        _tool_call("call_1"),
        _tool_call("call_2"),
    )
    messages = [
        HumanMessage(content="Check both."),
        assistant,
        ToolMessage(content="one", tool_call_id="call_1"),
        ToolMessage(content="two", tool_call_id="call_2"),
    ]

    outbound = _outbound_messages(_model(), messages)
    outbound_assistant = next(m for m in outbound if m["role"] == "assistant")
    outbound_tools = [m for m in outbound if m["role"] == "tool"]

    assert [call["id"] for call in outbound_assistant["tool_calls"]] == [
        "call_1",
        "call_2",
    ]
    assert [(m["tool_call_id"], m["content"]) for m in outbound_tools] == [
        ("call_1", "one"),
        ("call_2", "two"),
    ]


def test_non_assistant_messages_are_unaffected() -> None:
    call = _tool_call("call_1")
    messages = [
        SystemMessage(content="system instructions"),
        HumanMessage(content="human request"),
        _assistant_with_reasoning("assistant reasoning", call),
        ToolMessage(content="tool output", tool_call_id="call_1"),
    ]

    base_messages = _base_model()._get_request_payload(messages)["messages"]
    outbound = _outbound_messages(_model(), messages)

    assert outbound[0] == base_messages[0]
    assert outbound[1] == base_messages[1]
    assert outbound[3] == base_messages[3]
    assert outbound[0] == {"content": "system instructions", "role": "system"}
    assert outbound[1] == {"content": "human request", "role": "user"}
    assert outbound[3] == {
        "content": "tool output",
        "role": "tool",
        "tool_call_id": "call_1",
    }
    for message in (outbound[0], outbound[1], outbound[3]):
        assert "reasoning_content" not in message


def test_message_count_mismatch_fails_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_parent(
        self: ChatDeepSeek,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        return {"messages": [{"role": "user", "content": "only one"}]}

    monkeypatch.setattr(ChatDeepSeek, "_get_request_payload", fake_parent)

    with pytest.raises(RuntimeError, match="Cannot align ChatDeepSeek messages"):
        _model()._get_request_payload(
            [HumanMessage(content="one"), HumanMessage(content="two")]
        )


def test_role_mismatch_fails_without_attaching_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_assistant = {"role": "user", "content": ""}

    def fake_parent(
        self: ChatDeepSeek,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "start"},
                wrong_assistant,
            ]
        }

    monkeypatch.setattr(ChatDeepSeek, "_get_request_payload", fake_parent)

    with pytest.raises(RuntimeError, match="assistant messages"):
        _model()._get_request_payload(
            [
                HumanMessage(content="start"),
                _assistant_with_reasoning("must not leak", _tool_call("call_1")),
            ]
        )

    assert "reasoning_content" not in wrong_assistant


def test_tool_call_mismatch_fails_without_attaching_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "wrong_call",
                "type": "function",
                "function": {"name": "lookup_status", "arguments": "{}"},
            }
        ],
    }

    def fake_parent(
        self: ChatDeepSeek,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "start"},
                wrong_assistant,
            ]
        }

    monkeypatch.setattr(ChatDeepSeek, "_get_request_payload", fake_parent)

    with pytest.raises(RuntimeError, match="tool calls"):
        _model()._get_request_payload(
            [
                HumanMessage(content="start"),
                _assistant_with_reasoning("must not leak", _tool_call("call_1")),
            ]
        )

    assert "reasoning_content" not in wrong_assistant


def test_invalid_reasoning_content_type_fails_safely() -> None:
    assistant = AIMessage(
        content="",
        tool_calls=[_tool_call("call_1")],
        additional_kwargs={"reasoning_content": 123},
    )
    messages = [
        HumanMessage(content="start"),
        assistant,
        ToolMessage(content="result", tool_call_id="call_1"),
    ]

    with pytest.raises(RuntimeError, match="Unexpected reasoning_content type"):
        _model()._get_request_payload(messages)


def test_factory_uses_configured_values() -> None:
    settings = _settings(
        deepseek_model="custom-deepseek-model",
        deepseek_base_url="https://example.invalid/v1",
        deepseek_reasoning_effort="high",
    )

    model = build_chat_model(settings)

    assert model.model_name == "custom-deepseek-model"
    assert model.api_base == "https://example.invalid/v1"
    assert model.reasoning_effort == "high"
    assert model.extra_body == {"thinking": {"type": "enabled"}}
    assert model.api_key is not None
    assert model.api_key.get_secret_value() == DUMMY_API_KEY


def test_factory_returns_custom_subclass() -> None:
    model = build_chat_model(_settings())

    assert isinstance(model, ReasoningPreservingChatDeepSeek)
    assert isinstance(model, ChatDeepSeek)
    assert model.reasoning_effort == "low"


def test_factory_missing_key_raises_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = _settings(deepseek_api_key=None)

    with pytest.raises(MissingDeepSeekApiKeyError) as exc_info:
        build_chat_model(settings)

    message = str(exc_info.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "not configured" in message.lower()
    assert DUMMY_API_KEY not in message
