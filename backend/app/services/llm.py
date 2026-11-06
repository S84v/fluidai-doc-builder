"""DeepSeek model integration and construction."""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from backend.app.config import Settings, get_settings


class MissingDeepSeekApiKeyError(RuntimeError):
    """Raised when a DeepSeek model is requested without an API key."""


class ReasoningPreservingChatDeepSeek(ChatDeepSeek):
    """ChatDeepSeek that returns prior reasoning content to the API.

    DeepSeek requires ``reasoning_content`` from earlier assistant turns to be
    included when a request carries tools. ``langchain-deepseek`` 1.1.1 stores
    the value in ``AIMessage.additional_kwargs`` on the way in, but omits it
    when serializing messages for the next request.
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Restore reasoning content onto the matching outbound messages.

        The parent serializer preserves message order, so reasoning is restored
        positionally. The lengths, roles and tool-call IDs are checked first;
        mismatches raise instead of risking reasoning being attached to the
        wrong assistant turn.
        """
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        outbound_messages = payload.get("messages")
        if outbound_messages is None:
            raise RuntimeError(
                "ReasoningPreservingChatDeepSeek requires a Chat Completions payload."
            )
        if len(outbound_messages) != len(messages):
            raise RuntimeError(
                "Cannot align ChatDeepSeek messages with the request payload; "
                "refusing to restore reasoning_content."
            )

        for message, outbound in zip(messages, outbound_messages, strict=True):
            if not isinstance(message, AIMessage):
                continue
            if outbound.get("role") != "assistant":
                raise RuntimeError(
                    "Cannot align ChatDeepSeek assistant messages with the "
                    "request payload; refusing to restore reasoning_content."
                )

            reasoning_content = message.additional_kwargs.get("reasoning_content")
            if reasoning_content is None:
                continue
            if not isinstance(reasoning_content, str):
                raise RuntimeError(
                    "Unexpected reasoning_content type on an assistant message."
                )

            source_tool_call_ids = [call.get("id") for call in message.tool_calls]
            outbound_tool_call_ids = [
                call.get("id") for call in outbound.get("tool_calls") or []
            ]
            if source_tool_call_ids != outbound_tool_call_ids:
                raise RuntimeError(
                    "Cannot align ChatDeepSeek assistant tool calls with the "
                    "request payload; refusing to restore reasoning_content."
                )

            outbound["reasoning_content"] = reasoning_content

        return payload


def build_chat_model(
    settings: Settings | None = None,
) -> ReasoningPreservingChatDeepSeek:
    """Construct the application's DeepSeek chat model on demand."""
    resolved = settings or get_settings()
    if not resolved.deepseek_api_key:
        raise MissingDeepSeekApiKeyError(
            "DEEPSEEK_API_KEY is not configured. Set it in the environment or "
            "in the local .env file."
        )

    return ReasoningPreservingChatDeepSeek(
        model=resolved.deepseek_model,
        api_key=resolved.deepseek_api_key,
        base_url=resolved.deepseek_base_url,
        reasoning_effort=resolved.deepseek_reasoning_effort,
        extra_body={"thinking": {"type": "enabled"}},
    )
