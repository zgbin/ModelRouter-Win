"""
Anthropic ↔ OpenAI protocol converter for ModelRouter.

Faithful Python port of the Android Kotlin ProtocolConverter object.
All functions are module-level (no class needed).
"""

import json
import re
import time

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns for clean_tool_call_tags
# ---------------------------------------------------------------------------
_TC_RESIDUAL_RE = re.compile(
    r"(</(?:function|tool|param|parameter)>){2,}", re.IGNORECASE
)
_TC_TAG_RE = re.compile(
    r"</?(?:function|tool|param|parameter)[^>]*/?>", re.IGNORECASE
)
_TC_INCOMPLETE_TAG_RE = re.compile(
    r"</?(?:function|tool|param|parameter)\b[^<]*$", re.IGNORECASE | re.MULTILINE
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def anthropic_to_openai_request(anthropic_body: dict) -> dict:
    """Convert an Anthropic Messages API request body to OpenAI Chat Completions format."""

    openai_body: dict = {}

    # --- model ---
    openai_body["model"] = anthropic_body.get("model", "unknown")

    # --- max_tokens ---
    max_tokens = (
        anthropic_body.get("max_tokens")
        or anthropic_body.get("max_completion_tokens")
        or 4096
    )
    openai_body["max_tokens"] = int(max_tokens)

    # --- stream ---
    openai_body["stream"] = anthropic_body.get("stream", False)

    # --- temperature (clamp to [0, 1]) ---
    temperature = anthropic_body.get("temperature")
    if temperature is not None:
        openai_body["temperature"] = max(0.0, min(1.0, float(temperature)))

    # --- top_p ---
    top_p = anthropic_body.get("top_p")
    if top_p is not None:
        openai_body["top_p"] = float(top_p)

    # --- stop_sequences → stop ---
    stop_sequences = anthropic_body.get("stop_sequences")
    if stop_sequences and len(stop_sequences) > 0:
        openai_body["stop"] = stop_sequences

    # --- metadata.user_id → user ---
    metadata = anthropic_body.get("metadata")
    if isinstance(metadata, dict):
        user_id = metadata.get("user_id")
        if user_id:
            openai_body["user"] = user_id

    # --- system → system message ---
    system_text = _extract_system_text(anthropic_body.get("system"))

    messages: list[dict] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})

    # --- convert messages ---
    for msg in anthropic_body.get("messages", []):
        _convert_anthropic_message(msg, messages)
    openai_body["messages"] = messages

    # --- tools ---
    _convert_anthropic_tools(anthropic_body, openai_body)

    # --- tool_choice ---
    _convert_anthropic_tool_choice(anthropic_body, openai_body)

    # --- thinking → chat_template_kwargs.enable_thinking ---
    thinking = anthropic_body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type in ("enabled", "adaptive"):
            openai_body["chat_template_kwargs"] = {"enable_thinking": True}

    return openai_body


def openai_to_anthropic_response(openai_response: dict, request_model: str) -> dict:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format."""

    choices = openai_response.get("choices", [])
    choice = choices[0] if len(choices) > 0 else None
    message = choice.get("message") if isinstance(choice, dict) else None
    finish_reason = (choice or {}).get("finish_reason", "stop")

    output_content = ""
    tool_calls: list[dict] = []
    reasoning_content = None

    if isinstance(message, dict):
        output_content = message.get("content") or ""
        rc = message.get("reasoning_content")
        if isinstance(rc, str):
            reasoning_content = rc
        tcs = message.get("tool_calls")
        if isinstance(tcs, list):
            tool_calls = [tc for tc in tcs if isinstance(tc, dict)]

    # Build Anthropic content array
    response_content: list[dict] = []

    if reasoning_content:
        response_content.append({"type": "thinking", "thinking": reasoning_content})

    cleaned = clean_tool_call_tags(output_content)
    if cleaned:
        response_content.append({"type": "text", "text": cleaned})

    for tc in tool_calls:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
        try:
            parsed_input = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            parsed_input = {}
        response_content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", "") if isinstance(func, dict) else "",
            "input": parsed_input,
        })

    # Usage
    usage = openai_response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    prompt_tokens_details = usage.get("prompt_tokens_details")
    cache_creation = (
        prompt_tokens_details.get("cached_tokens", 0)
        if isinstance(prompt_tokens_details, dict)
        else 0
    )

    anthropic_usage: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_creation > 0:
        anthropic_usage["cache_creation_input_tokens"] = 0
        anthropic_usage["cache_read_input_tokens"] = cache_creation

    response = {
        "id": openai_response.get("id", f"msg_{int(time.time() * 1000)}"),
        "type": "message",
        "role": "assistant",
        "content": response_content,
        "model": openai_response.get("model", request_model),
        "stop_reason": map_finish_reason_to_stop_reason(finish_reason, len(tool_calls) > 0),
        "stop_sequence": None,
        "usage": anthropic_usage,
    }

    return response


def map_finish_reason_to_stop_reason(finish_reason: str, has_tool_calls: bool) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "content_filter":
        return "end_turn"
    return "end_turn"


def map_stop_reason_to_finish_reason(stop_reason: str | None) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    return "stop"


def clean_tool_call_tags(text: str) -> str:
    """Remove residual function/tool/param XML-like tags from model output."""
    if not text:
        return ""
    cleaned = _TC_RESIDUAL_RE.sub("", text)
    cleaned = _TC_TAG_RE.sub("", cleaned)
    cleaned = _TC_INCOMPLETE_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def extract_content_string(raw_content) -> str:
    """Extract text from various Anthropic content formats (string, array of objects)."""
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "thinking":
                    parts.append(item.get("thinking", ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return json.dumps(raw_content, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_system_text(system) -> str:
    """Extract system text from Anthropic system field (string / array / object)."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for item in system:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _convert_anthropic_message(msg_obj: dict, messages: list[dict]) -> None:
    """Convert a single Anthropic message and append it to the OpenAI messages list."""
    role = msg_obj.get("role", "user")
    content = msg_obj.get("content")

    if isinstance(content, list):
        text_parts: list[str] = []
        tool_results: list[dict] = []
        tool_uses: list[dict] = []
        image_parts: list[dict] = []

        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "text":
                text_parts.append(item.get("text", ""))

            elif item_type == "image":
                source = item.get("source")
                if not isinstance(source, dict):
                    continue
                src_type = source.get("type", "")
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                url = source.get("url", "")
                if src_type == "base64" and data:
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    })
                elif src_type == "url" and url:
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                    })

            elif item_type == "tool_result":
                tool_use_id = item.get("tool_use_id", "")
                is_error = item.get("is_error", False)
                raw_content = item.get("content")
                content_str = extract_content_string(raw_content)
                tool_results.append({
                    "tool_call_id": tool_use_id,
                    "content": f"[Tool Error] {content_str}" if (is_error and content_str) else content_str,
                })

            elif item_type == "tool_use":
                tool_uses.append({
                    "id": item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": json.dumps(item.get("input", {}), ensure_ascii=False),
                    },
                })

            elif item_type == "thinking":
                # Skip thinking blocks — not sent to OpenAI
                pass

        # Assemble message(s) based on content types and role
        if role == "assistant" and tool_uses:
            messages.append({
                "role": role,
                "content": "\n".join(text_parts),
                "tool_calls": tool_uses,
            })
        elif role == "user" and tool_results:
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"],
                })
            if text_parts or image_parts:
                _add_user_content_message(messages, text_parts, image_parts)
        elif role == "user" and image_parts:
            _add_user_content_message(messages, text_parts, image_parts)
        else:
            text = "\n".join(text_parts)
            if text:
                messages.append({"role": role, "content": text})
    else:
        # Simple string content
        content_str = content if isinstance(content, str) else ""
        messages.append({"role": role, "content": content_str})


def _add_user_content_message(messages: list[dict], text_parts: list[str], image_parts: list[dict]) -> None:
    """Add a user message with mixed text/image content."""
    if image_parts:
        user_content: list[dict] = []
        for tp in text_parts:
            user_content.append({"type": "text", "text": tp})
        for ip in image_parts:
            user_content.append(ip)
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": "\n".join(text_parts)})


def _convert_anthropic_tools(anthropic_body: dict, openai_body: dict) -> None:
    """Convert Anthropic tools to OpenAI function tools."""
    tools = anthropic_body.get("tools")
    if not tools:
        return
    openai_tools: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        input_schema = tool.get("input_schema") or tool.get("parameters") or {}
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": input_schema,
            },
        })
    if openai_tools:
        openai_body["tools"] = openai_tools


def _convert_anthropic_tool_choice(anthropic_body: dict, openai_body: dict) -> None:
    """Convert Anthropic tool_choice to OpenAI tool_choice."""
    tool_choice = anthropic_body.get("tool_choice")
    if tool_choice is None:
        return

    if isinstance(tool_choice, str):
        openai_body["tool_choice"] = "required" if tool_choice == "any" else tool_choice
    elif isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        if tc_type == "auto":
            openai_body["tool_choice"] = "auto"
        elif tc_type == "any":
            openai_body["tool_choice"] = "required"
        elif tc_type == "none":
            openai_body["tool_choice"] = "none"
        elif tc_type == "tool":
            openai_body["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
        else:
            openai_body["tool_choice"] = tc_type or "auto"
