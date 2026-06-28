"""
HTTP 代理服务器 - ModelRouter Windows 应用

从 Android Kotlin ModelRouterServer 移植而来，使用 aiohttp 实现异步 HTTP 服务。
负责接收所有 API 请求，智能路由到上游 AI 提供商，支持模型切换、故障转移和密钥轮换。
"""

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Set

import aiohttp
from aiohttp import web

from core.config_manager import config_manager
from core.models import SpeedTestResult
from core.protocol_converter import (
    anthropic_to_openai_request,
    clean_tool_call_tags,
    map_finish_reason_to_stop_reason,
    openai_to_anthropic_response,
)
from core.provider_manager import provider_manager
from core.router_state import router_state
from core.speed_tester import SpeedTester
from core.stats_manager import stats_manager

logger = logging.getLogger(__name__)

# OpenAI 标准请求参数白名单
OPENAI_STANDARD_PARAMS: Set[str] = {
    "model", "messages", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "n", "stream", "stream_options",
    "stop", "presence_penalty", "frequency_penalty",
    "logit_bias", "logprobs", "top_logprobs",
    "user", "tools", "tool_choice", "parallel_tool_calls",
    "response_format", "seed", "service_tier",
    "chat_template_kwargs", "repetition_penalty",
}

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class _RouterServer:
    """ModelRouter HTTP 代理服务器"""

    def __init__(self, port: int = 8190):
        self.port = port
        self.session: Optional[aiohttp.ClientSession] = None
        self._speed_tester = SpeedTester()
        self._runner: Optional[aiohttp.web.AppRunner] = None

    # ================================================================
    # 服务器生命周期
    # ================================================================

    def create_app(self) -> web.Application:
        """创建 aiohttp 应用并注册所有路由"""
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat_completion)
        app.router.add_post("/v1/messages", self.handle_anthropic_messages)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/api/status", self.handle_status)
        app.router.add_post("/api/speed_test", self.handle_speed_test)
        app.router.add_post("/api/lock", self.handle_lock)
        app.router.add_post("/api/unlock", self.handle_unlock)
        app.router.add_get("/api/config", self.handle_get_config)
        app.router.add_get("/api/stats", self.handle_get_stats)
        app.router.add_get("/api/dashboard", self.handle_dashboard_api)
        app.router.add_post("/api/reload", self.handle_reload)
        app.router.add_post("/api/lock_model", self.handle_lock_model_api)
        app.router.add_post("/api/unlock_model", self.handle_unlock_model_api)
        app.router.add_get("/api/lock_status", self.handle_lock_status_api)
        app.router.add_get("/health", self.handle_health)
        return app

    async def start(self) -> None:
        """启动服务器"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300, connect=30)
        )
        app = self.create_app()
        self._runner = aiohttp.web.AppRunner(app)
        await self._runner.setup()
        site = aiohttp.web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        logger.info("RouterServer started on port %d", self.port)

    async def stop(self) -> None:
        """停止服务器"""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self.session is not None:
            await self.session.close()
            self.session = None
        logger.info("RouterServer stopped")

    # ================================================================
    # 核心：Chat Completions
    # ================================================================

    async def handle_chat_completion(self, request: web.Request) -> web.StreamResponse:
        """POST /v1/chat/completions"""
        try:
            json_body = await request.json()
            is_stream = json_body.get("stream", False)
            group_name = json_body.get("group")

            if is_stream:
                return await self._handle_stream_completion_impl(
                    request, json_body, group_name
                )
            else:
                return await self._handle_non_stream_completion(json_body, group_name)
        except Exception as e:
            logger.exception("Error handling chat completion")
            return self._json_error("api_error", str(e))

    async def _handle_non_stream_completion(
        self, json_body: dict, group_name: Optional[str]
    ) -> web.Response:
        """非流式 Chat Completions 处理逻辑"""
        group = group_name or config_manager.get_group_by_port(self.port)
        tried_model_ids: Set[str] = set()

        for model_retry in range(3):
            model_id: Optional[str] = None
            model_acquired = False
            try:
                model_id = (
                    self._get_model_to_use(json_body, group)
                    if model_retry == 0
                    else config_manager.select_fastest_model(group)
                )
                if model_id is None:
                    return self._json_error("api_error", "No model available")
                if model_id in tried_model_ids:
                    return self._json_error("api_error", "No alternative model available")
                tried_model_ids.add(model_id)

                router_state.acquire_model(model_id)
                model_acquired = True
                json_body["model"] = model_id
                json_body.pop("group", None)

                provider_id = self._get_provider_id_for_model(model_id)
                api_key = provider_manager.get_next_key(provider_id)
                if not api_key:
                    return self._json_error(
                        "auth_error",
                        f"No API key available for provider {provider_id}",
                    )

                sanitized_body = self._sanitize_request_body(json_body)
                response_str = await self._forward_to_provider(
                    sanitized_body, api_key, provider_id
                )

                # 检查 early_rate_limit
                is_early_rate_limit = False
                try:
                    parsed = json.loads(response_str)
                    error = parsed.get("error")
                    if isinstance(error, dict):
                        error_type = error.get("type", "")
                        if error_type == "early_rate_limit":
                            is_early_rate_limit = True
                    if not is_early_rate_limit:
                        if "error" in parsed:
                            stats_manager.record_call(model_id, False)
                            error_msg = (
                                parsed.get("error", {}).get("message", "unknown")
                                if isinstance(parsed.get("error"), dict)
                                else "unknown"
                            )
                            error_type = (
                                parsed.get("error", {}).get("type", "")
                                if isinstance(parsed.get("error"), dict)
                                else ""
                            )
                            router_state.update_model_error(
                                model_id, f"上游错误: {error_msg}"
                            )
                            if router_state.get_locked_model(group) == model_id:
                                router_state.unlock_group(group)
                            logger.warning(
                                "Non-stream upstream error on model %s (%s), switching model",
                                model_id,
                                error_type,
                            )
                            continue
                        if "choices" not in parsed:
                            stats_manager.record_call(model_id, False)
                            return self._json_error(
                                "api_error",
                                "Invalid response format from upstream: missing choices",
                            )
                except Exception:
                    pass

                if is_early_rate_limit:
                    router_state.update_model_error(model_id, "429 限流")
                    if router_state.get_locked_model(group) == model_id:
                        router_state.unlock_group(group)
                    logger.warning(
                        "Early 429 on model %s, switching model (retry %d)",
                        model_id,
                        model_retry + 1,
                    )
                    continue

                stats_manager.record_call(model_id, True)
                return web.Response(
                    status=200,
                    content_type="application/json",
                    text=response_str,
                )
            except Exception as e:
                logger.warning(
                    "Non-stream completion error on model %s, switching to next model",
                    model_id,
                )
                if model_id is not None:
                    try:
                        stats_manager.record_call(model_id, False)
                    except Exception:
                        pass
                    router_state.update_model_error(model_id, f"异常: {e}")
                    if router_state.get_locked_model(group) == model_id:
                        router_state.unlock_group(group)
                continue
            finally:
                if model_acquired and model_id is not None:
                    router_state.release_model(model_id)

        return self._json_error("api_error", "No model available after early rate limit")

    # ================================================================
    # 核心：Anthropic Messages
    # ================================================================

    async def handle_anthropic_messages(
        self, request: web.Request
    ) -> web.StreamResponse:
        """POST /v1/messages"""
        try:
            json_body = await request.json()
            is_stream = json_body.get("stream", False)
            model = json_body.get("model", "unknown")

            chat_body = anthropic_to_openai_request(json_body)

            if is_stream:
                return await self._handle_anthropic_stream_impl(
                    request, chat_body, model
                )
            else:
                return await self._handle_anthropic_non_stream(chat_body, model)
        except Exception as e:
            logger.exception("Anthropic error")
            return self._anthropic_error("api_error", str(e))

    async def _handle_anthropic_non_stream(
        self, chat_body: dict, model: str
    ) -> web.Response:
        """Anthropic 非流式处理"""
        group = config_manager.get_group_by_port(self.port)
        tried_model_ids: Set[str] = set()

        for model_retry in range(3):
            model_id: Optional[str] = None
            model_acquired = False
            try:
                model_id = (
                    self._get_model_to_use(chat_body, None)
                    if model_retry == 0
                    else config_manager.select_fastest_model(group)
                )
                if model_id is None:
                    return self._anthropic_error("api_error", "No model available")
                if model_id in tried_model_ids:
                    return self._anthropic_error(
                        "api_error", "No alternative model available"
                    )
                tried_model_ids.add(model_id)

                router_state.acquire_model(model_id)
                model_acquired = True
                chat_body["model"] = model_id
                provider_id = self._get_provider_id_for_model(model_id)
                api_key = provider_manager.get_next_key(provider_id)
                if not api_key:
                    return self._anthropic_error(
                        "authentication_error",
                        f"No API key available for provider {provider_id}",
                    )

                sanitized_chat_body = self._sanitize_request_body(chat_body)
                chat_response = await self._forward_to_provider(
                    sanitized_chat_body, api_key, provider_id
                )

                # 检查 early_rate_limit
                is_early_rate_limit = False
                try:
                    check_parsed = json.loads(chat_response)
                    check_error = check_parsed.get("error")
                    if isinstance(check_error, dict):
                        error_type = check_error.get("type", "")
                        if error_type == "early_rate_limit":
                            is_early_rate_limit = True
                except Exception:
                    pass

                if is_early_rate_limit:
                    router_state.update_model_error(model_id, "429 限流")
                    if router_state.get_locked_model(group) == model_id:
                        router_state.unlock_group(group)
                    logger.warning(
                        "Anthropic non-stream early 429 on model %s, switching model",
                        model_id,
                    )
                    continue

                response_obj = None
                try:
                    response_obj = json.loads(chat_response)
                except Exception as e:
                    logger.exception("Failed to parse upstream response as JSON")
                    return self._anthropic_error(
                        "api_error", "Invalid response from upstream: not valid JSON"
                    )

                if "error" in response_obj:
                    stats_manager.record_call(model_id, False)
                    error_msg = (
                        response_obj.get("error", {}).get("message", "unknown")
                        if isinstance(response_obj.get("error"), dict)
                        else "unknown"
                    )
                    router_state.update_model_error(model_id, f"上游错误: {error_msg}")
                    if router_state.get_locked_model(group) == model_id:
                        router_state.unlock_group(group)
                    logger.warning(
                        "Anthropic non-stream upstream error on model %s, switching model",
                        model_id,
                    )
                    continue

                choices = response_obj.get("choices")
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    stats_manager.record_call(model_id, False)
                    return self._anthropic_error(
                        "api_error", "No choices in upstream response"
                    )

                anthropic_response = openai_to_anthropic_response(
                    response_obj, model
                )
                stats_manager.record_call(model_id, True)
                return web.Response(
                    status=200,
                    content_type="application/json",
                    text=json.dumps(anthropic_response, ensure_ascii=False),
                )
            except Exception as e:
                logger.warning(
                    "Anthropic non-stream error on model %s, switching to next model",
                    model_id,
                )
                if model_id is not None:
                    try:
                        stats_manager.record_call(model_id, False)
                    except Exception:
                        pass
                    router_state.update_model_error(model_id, f"异常: {e}")
                    if router_state.get_locked_model(group) == model_id:
                        router_state.unlock_group(group)
                continue
            finally:
                if model_acquired and model_id is not None:
                    router_state.release_model(model_id)

        return self._anthropic_error(
            "api_error", "No model available after early rate limit"
        )

    # ----------------------------------------------------------------
    # Anthropic 流式代理（需要原始 request 来 prepare StreamResponse）
    # ----------------------------------------------------------------

    async def _handle_anthropic_stream_impl(
        self,
        request: web.Request,
        chat_body: dict,
        model: str,
    ) -> web.StreamResponse:
        """Anthropic 流式代理实现 - 完整版本，接收原始 request"""
        group = config_manager.get_group_by_port(self.port)
        tried_model_ids: Set[str] = set()

        for model_retry in range(3):
            model_id = (
                self._get_model_to_use(chat_body, None)
                if model_retry == 0
                else config_manager.select_fastest_model(group)
            )
            if model_id is None:
                return self._anthropic_error("api_error", "No model available")
            if model_id in tried_model_ids:
                return self._anthropic_error(
                    "api_error", "No alternative model available"
                )
            tried_model_ids.add(model_id)

            model_acquired = False
            switch_model = False
            try:
                router_state.acquire_model(model_id)
                model_acquired = True
                chat_body["model"] = model_id
                chat_body["stream"] = True

                provider_id = self._get_provider_id_for_model(model_id)
                api_key = provider_manager.get_next_key(provider_id)
                if not api_key:
                    return self._anthropic_error(
                        "authentication_error",
                        f"No API key available for provider {provider_id}",
                    )

                base_url = (
                    provider_manager.get_provider(provider_id).base_url
                    if provider_manager.get_provider(provider_id)
                    else DEFAULT_BASE_URL
                )

                current_api_key = api_key
                attempt = 0
                max_retries = 2
                total_sleep_ms = 0
                max_total_sleep_ms = 5000

                while True:
                    attempt += 1
                    body_str = json.dumps(self._sanitize_request_body(chat_body))
                    url = f"{base_url.rstrip('/')}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {current_api_key}",
                        "Content-Type": "application/json",
                    }

                    try:
                        timeout_sec = config_manager.get_model_timeout(model_id)
                        stream_timeout = aiohttp.ClientTimeout(
                            total=300, connect=30, sock_read=timeout_sec
                        )
                        upstream_resp = await self.session.post(
                            url,
                            data=body_str,
                            headers=headers,
                            timeout=stream_timeout,
                        )

                        if upstream_resp.status != 200:
                            error_body = await upstream_resp.text()

                            if upstream_resp.status == 429:
                                if provider_manager.is_early429(
                                    provider_id, current_api_key
                                ):
                                    router_state.update_model_error(
                                        model_id, "429 限流"
                                    )
                                    if (
                                        router_state.get_locked_model(group)
                                        == model_id
                                    ):
                                        router_state.unlock_group(group)
                                    logger.warning(
                                        "Anthropic stream early 429 on model %s, switching model",
                                        model_id,
                                    )
                                    switch_model = True
                                    break

                                if attempt <= max_retries:
                                    new_key = provider_manager.peek_next_key(
                                        provider_id, current_api_key
                                    )
                                    if new_key and new_key != current_api_key:
                                        current_api_key = new_key
                                        logger.warning(
                                            "Anthropic stream 429, retry %d with different key",
                                            attempt,
                                        )
                                        continue
                                    delay = 500 * attempt
                                    if total_sleep_ms + delay > max_total_sleep_ms:
                                        logger.warning(
                                            "Anthropic stream 429, total sleep limit reached, giving up"
                                        )
                                        break
                                    total_sleep_ms += delay
                                    logger.warning(
                                        "Anthropic stream 429, retry %d after %dms (same key)",
                                        attempt,
                                        delay,
                                    )
                                    await asyncio.sleep(delay / 1000.0)
                                    continue

                            if upstream_resp.status == 400:
                                logger.warning(
                                    "Anthropic stream 400 from %s: %s",
                                    provider_id,
                                    error_body[:300],
                                )
                            else:
                                logger.warning(
                                    "Anthropic stream upstream error HTTP %d: %s",
                                    upstream_resp.status,
                                    error_body[:200],
                                )
                            stats_manager.record_call(model_id, False)
                            router_state.update_model_error(
                                model_id, f"HTTP {upstream_resp.status}"
                            )
                            if router_state.get_locked_model(group) == model_id:
                                router_state.unlock_group(group)
                            switch_model = True
                            break

                        # 成功 - 建立 Anthropic SSE 流式响应
                        stats_manager.record_call(model_id, True)
                        model_acquired = False  # 由流结束时释放

                        message_id = f"msg_{int(time.time() * 1000)}"

                        response = web.StreamResponse(
                            status=200,
                            headers={
                                "Content-Type": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            },
                        )
                        await response.prepare(request)

                        # 写入 Anthropic 流式事件
                        try:
                            await self._write_anthropic_stream_events(
                                message_id, model, upstream_resp.content, response
                            )
                        except Exception:
                            logger.exception(
                                "Anthropic stream pipe error for model %s",
                                model_id,
                            )
                        finally:
                            router_state.release_model(model_id)
                            try:
                                await response.write_eof()
                            except Exception:
                                pass

                        return response

                    except asyncio.TimeoutError:
                        logger.warning(
                            "Anthropic stream timeout on model %s, switching to next model",
                            model_id,
                        )
                        try:
                            stats_manager.record_call(model_id, False)
                        except Exception:
                            pass
                        router_state.update_model_error(model_id, "超时")
                        if router_state.get_locked_model(group) == model_id:
                            router_state.unlock_group(group)
                        switch_model = True
                        break
                    except (aiohttp.ClientConnectorError, aiohttp.ClientError):
                        logger.warning(
                            "Anthropic stream connection error on model %s, switching to next model",
                            model_id,
                        )
                        try:
                            stats_manager.record_call(model_id, False)
                        except Exception:
                            pass
                        router_state.update_model_error(model_id, "连接失败")
                        if router_state.get_locked_model(group) == model_id:
                            router_state.unlock_group(group)
                        switch_model = True
                        break
            except Exception as e:
                logger.exception("Anthropic stream outer error")
                return self._anthropic_error("api_error", str(e))
            finally:
                if model_acquired:
                    router_state.release_model(model_id)

            if switch_model:
                continue

        return self._anthropic_error(
            "api_error", "No model available after early rate limit"
        )

    # ----------------------------------------------------------------
    # Anthropic SSE 转换 - 最核心最复杂的部分
    # ----------------------------------------------------------------

    async def _write_anthropic_stream_events(
        self,
        message_id: str,
        model: str,
        input_stream: aiohttp.StreamReader,
        response: web.StreamResponse,
    ) -> None:
        """将 OpenAI SSE 流转换为 Anthropic SSE 流

        这是整个服务器最复杂的方法，忠实移植自 Android 的 writeAnthropicStreamEvents。
        """
        # message_start 事件
        msg_start_data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        await self._write_sse(response, "message_start", msg_start_data)
        await self._write_sse(response, "ping", {"type": "ping"})

        output_tokens = 0
        input_tokens = 0

        # tool_call 状态跟踪
        class ToolCallState:
            def __init__(self, tc_id: str, tc_name: str):
                self.id = tc_id
                self.name = tc_name

        tool_call_blocks: Dict[int, ToolCallState] = {}
        tool_call_index_map: Dict[int, int] = {}
        current_tool_index = 0
        has_tool_calls = False

        text_block_started = False
        thinking_block_started = False
        last_finish_reason: Optional[str] = None
        content_block_index = 0
        text_block_index = -1
        thinking_block_index = -1

        try:
            buffer = b""
            async for raw_chunk in input_stream:
                buffer += raw_chunk
                # 按行处理
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()

                    if not line.startswith("data: "):
                        continue
                    json_str = line[6:].strip()
                    if json_str == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(json_str)
                    except Exception:
                        logger.warning("Stream chunk parse error: %s", json_str[:100])
                        continue

                    # 提取 usage
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        pt = usage.get("prompt_tokens")
                        if isinstance(pt, int):
                            input_tokens = pt

                    # 提取 choices
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or len(choices) == 0:
                        continue

                    choice_obj = choices[0]
                    if not isinstance(choice_obj, dict):
                        continue

                    finish_reason = choice_obj.get("finish_reason")
                    if finish_reason is not None and finish_reason != "null":
                        last_finish_reason = finish_reason

                    delta = choice_obj.get("delta")
                    if not isinstance(delta, dict):
                        continue

                    reasoning_content = delta.get("reasoning_content", "") or ""
                    content = delta.get("content", "") or ""
                    tool_calls_delta = delta.get("tool_calls")

                    if reasoning_content or content or tool_calls_delta:
                        logger.debug(
                            "Stream delta: reasoning=%s, content=%s, tools=%s",
                            reasoning_content[:30],
                            content[:30],
                            tool_calls_delta is not None,
                        )

                    # --- reasoning_content -> thinking block ---
                    if reasoning_content:
                        if not thinking_block_started:
                            if text_block_started:
                                await self._write_sse(
                                    response,
                                    "content_block_stop",
                                    {"type": "content_block_stop", "index": text_block_index},
                                )
                                text_block_started = False
                            thinking_block_index = content_block_index
                            content_block_index += 1
                            thinking_block_started = True
                            await self._write_sse(
                                response,
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": thinking_block_index,
                                    "content_block": {
                                        "type": "thinking",
                                        "thinking": "",
                                    },
                                },
                            )
                        output_tokens += 1
                        await self._write_sse(
                            response,
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": thinking_block_index,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": reasoning_content,
                                },
                            },
                        )

                    # --- content -> text block ---
                    if content:
                        if thinking_block_started:
                            await self._write_sse(
                                response,
                                "content_block_stop",
                                {"type": "content_block_stop", "index": thinking_block_index},
                            )
                            thinking_block_started = False
                        cleaned = clean_tool_call_tags(content)
                        if cleaned:
                            if not text_block_started:
                                text_block_index = content_block_index
                                content_block_index += 1
                                text_block_started = True
                                await self._write_sse(
                                    response,
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": text_block_index,
                                        "content_block": {
                                            "type": "text",
                                            "text": "",
                                        },
                                    },
                                )
                            output_tokens += 1
                            await self._write_sse(
                                response,
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": cleaned,
                                    },
                                },
                            )

                    # --- tool_calls -> tool_use blocks ---
                    if isinstance(tool_calls_delta, list):
                        for tc_delta in tool_calls_delta:
                            if not isinstance(tc_delta, dict):
                                continue
                            has_tool_calls = True
                            tc_index = tc_delta.get("index", 0)
                            tc_id = tc_delta.get("id", "")
                            func = tc_delta.get("function")
                            tc_name = (
                                func.get("name", "") if isinstance(func, dict) else ""
                            )
                            tc_args = (
                                func.get("arguments", "")
                                if isinstance(func, dict)
                                else ""
                            )

                            if tc_index not in tool_call_blocks:
                                if text_block_started:
                                    await self._write_sse(
                                        response,
                                        "content_block_stop",
                                        {
                                            "type": "content_block_stop",
                                            "index": text_block_index,
                                        },
                                    )
                                    text_block_started = False

                                block_index = content_block_index
                                content_block_index += 1
                                tool_call_blocks[tc_index] = ToolCallState(
                                    tc_id, tc_name
                                )
                                tool_call_index_map[tc_index] = block_index
                                current_tool_index = block_index

                                await self._write_sse(
                                    response,
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tc_id,
                                            "name": tc_name,
                                            "input": {},
                                        },
                                    },
                                )

                                await self._write_sse(
                                    response,
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": block_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": "",
                                        },
                                    },
                                )

                            # 更新 tool call 状态
                            if tc_id and not tool_call_blocks[tc_index].id:
                                tool_call_blocks[tc_index].id = tc_id
                            if tc_name and not tool_call_blocks[tc_index].name:
                                tool_call_blocks[tc_index].name = tc_name

                            if tc_args:
                                block_index = tool_call_index_map.get(
                                    tc_index, current_tool_index
                                )
                                await self._write_sse(
                                    response,
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": block_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": tc_args,
                                        },
                                    },
                                )

            # 处理 buffer 中剩余的数据
            if buffer.strip():
                line = buffer.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    json_str = line[6:].strip()
                    if json_str != "[DONE]":
                        try:
                            chunk = json.loads(json_str)
                            usage = chunk.get("usage")
                            if isinstance(usage, dict):
                                pt = usage.get("prompt_tokens")
                                if isinstance(pt, int):
                                    input_tokens = pt
                        except Exception:
                            pass

            # 关闭未关闭的块
            if thinking_block_started:
                await self._write_sse(
                    response,
                    "content_block_stop",
                    {"type": "content_block_stop", "index": thinking_block_index},
                )
            if text_block_started:
                await self._write_sse(
                    response,
                    "content_block_stop",
                    {"type": "content_block_stop", "index": text_block_index},
                )
            for idx in tool_call_index_map.values():
                await self._write_sse(
                    response,
                    "content_block_stop",
                    {"type": "content_block_stop", "index": idx},
                )

            # message_delta 和 message_stop
            stop_reason = map_finish_reason_to_stop_reason(
                last_finish_reason or "stop", has_tool_calls
            )
            await self._write_sse(
                response,
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": output_tokens},
                },
            )
            await self._write_sse(
                response, "message_stop", {"type": "message_stop"}
            )

        except Exception as e:
            logger.exception("Stream write error")
            try:
                err_msg = str(e.message if hasattr(e, "message") else e).replace(
                    "\\", "\\\\"
                ).replace('"', '\\"')
                await self._write_sse(
                    response,
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "message": err_msg,
                            "type": "stream_error",
                        },
                    },
                )
            except Exception:
                pass

    # ================================================================
    # OpenAI 流式代理（使用原始 request 建立 StreamResponse）
    # ================================================================

    async def _handle_stream_completion_impl(
        self,
        request: web.Request,
        json_body: dict,
        group_name: Optional[str],
    ) -> web.StreamResponse:
        """流式 Chat Completions 代理实现 - 完整版本"""
        group = group_name or config_manager.get_group_by_port(self.port)
        tried_model_ids: Set[str] = set()

        for model_retry in range(3):
            model_id = (
                self._get_model_to_use(json_body, group)
                if model_retry == 0
                else config_manager.select_fastest_model(group)
            )
            if model_id is None:
                return self._json_error("api_error", "No model available")
            if model_id in tried_model_ids:
                return self._json_error("api_error", "No alternative model available")
            tried_model_ids.add(model_id)

            model_acquired = False
            switch_model = False
            try:
                router_state.acquire_model(model_id)
                model_acquired = True
                json_body["model"] = model_id
                json_body["stream"] = True
                json_body.pop("group", None)

                provider_id = self._get_provider_id_for_model(model_id)
                api_key = provider_manager.get_next_key(provider_id)
                if not api_key:
                    return self._json_error(
                        "auth_error",
                        f"No API key available for provider {provider_id}",
                    )

                base_url = (
                    provider_manager.get_provider(provider_id).base_url
                    if provider_manager.get_provider(provider_id)
                    else DEFAULT_BASE_URL
                )

                current_api_key = api_key
                attempt = 0
                max_retries = 2
                total_sleep_ms = 0
                max_total_sleep_ms = 5000

                while True:
                    attempt += 1
                    body_str = json.dumps(self._sanitize_request_body(json_body))
                    url = f"{base_url.rstrip('/')}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {current_api_key}",
                        "Content-Type": "application/json",
                    }

                    try:
                        timeout_sec = config_manager.get_model_timeout(model_id)
                        stream_timeout = aiohttp.ClientTimeout(
                            total=300, connect=30, sock_read=timeout_sec
                        )
                        upstream_resp = await self.session.post(
                            url,
                            data=body_str,
                            headers=headers,
                            timeout=stream_timeout,
                        )

                        if upstream_resp.status != 200:
                            error_body = await upstream_resp.text()

                            if upstream_resp.status == 429:
                                if provider_manager.is_early429(
                                    provider_id, current_api_key
                                ):
                                    router_state.update_model_error(
                                        model_id, "429 限流"
                                    )
                                    if (
                                        router_state.get_locked_model(group)
                                        == model_id
                                    ):
                                        router_state.unlock_group(group)
                                    logger.warning(
                                        "Stream early 429 on model %s, switching model",
                                        model_id,
                                    )
                                    switch_model = True
                                    break

                                if attempt <= max_retries:
                                    new_key = provider_manager.peek_next_key(
                                        provider_id, current_api_key
                                    )
                                    if new_key and new_key != current_api_key:
                                        current_api_key = new_key
                                        logger.warning(
                                            "Stream 429, retry %d with different key",
                                            attempt,
                                        )
                                        continue
                                    delay = 500 * attempt
                                    if total_sleep_ms + delay > max_total_sleep_ms:
                                        logger.warning(
                                            "Stream 429, total sleep limit reached, giving up"
                                        )
                                        break
                                    total_sleep_ms += delay
                                    logger.warning(
                                        "Stream 429, retry %d after %dms (same key)",
                                        attempt,
                                        delay,
                                    )
                                    await asyncio.sleep(delay / 1000.0)
                                    continue

                            if upstream_resp.status == 400:
                                logger.warning(
                                    "Stream 400 error from %s: %s",
                                    provider_id,
                                    error_body[:300],
                                )
                            else:
                                logger.warning(
                                    "Stream upstream error HTTP %d: %s",
                                    upstream_resp.status,
                                    error_body[:200],
                                )
                            stats_manager.record_call(model_id, False)
                            router_state.update_model_error(
                                model_id, f"HTTP {upstream_resp.status}"
                            )
                            if router_state.get_locked_model(group) == model_id:
                                router_state.unlock_group(group)
                            switch_model = True
                            break

                        # 成功 - 建立流式响应
                        stats_manager.record_call(model_id, True)
                        model_acquired = False  # 由流结束时释放

                        response = web.StreamResponse(
                            status=200,
                            headers={
                                "Content-Type": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            },
                        )
                        await response.prepare(request)

                        # 记录 TTFT 并代理流数据
                        first_chunk_received = False
                        start_time = time.time()

                        try:
                            async for raw_line in upstream_resp.content:
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = (time.time() - start_time) * 1000
                                    router_state.update_speed_test_result(
                                        model_id, int(ttft)
                                    )
                                await response.write(raw_line)
                        except Exception:
                            logger.exception(
                                "Stream write error for model %s", model_id
                            )
                        finally:
                            router_state.release_model(model_id)
                            try:
                                await response.write_eof()
                            except Exception:
                                pass

                        return response

                    except asyncio.TimeoutError:
                        logger.warning(
                            "Stream timeout on model %s, switching to next model",
                            model_id,
                        )
                        try:
                            stats_manager.record_call(model_id, False)
                        except Exception:
                            pass
                        router_state.update_model_error(model_id, "超时")
                        if router_state.get_locked_model(group) == model_id:
                            router_state.unlock_group(group)
                        switch_model = True
                        break
                    except (aiohttp.ClientConnectorError, aiohttp.ClientError):
                        logger.warning(
                            "Stream connection error on model %s, switching to next model",
                            model_id,
                        )
                        try:
                            stats_manager.record_call(model_id, False)
                        except Exception:
                            pass
                        router_state.update_model_error(model_id, "连接失败")
                        if router_state.get_locked_model(group) == model_id:
                            router_state.unlock_group(group)
                        switch_model = True
                        break
            except Exception as e:
                logger.exception("Stream outer error")
                return self._json_error("api_error", str(e))
            finally:
                if model_acquired:
                    router_state.release_model(model_id)

            if switch_model:
                continue

        return self._json_error("api_error", "No model available after early rate limit")

    # ================================================================
    # _forward_to_provider：转发请求到上游
    # ================================================================

    async def _forward_to_provider(
        self, body: dict, api_key: str, provider_id: str
    ) -> str:
        """将请求转发到上游提供者，处理 429 和超时重试"""
        provider = provider_manager.get_provider(provider_id)
        base_url = provider.base_url if provider else DEFAULT_BASE_URL

        if not api_key:
            return json.dumps(
                {"error": {"message": "No API key available", "type": "auth_error"}}
            )

        max_retries = 3
        attempt = 0
        current_api_key = api_key
        total_sleep_ms = 0
        max_total_sleep_ms = 8000

        while True:
            attempt += 1
            url = f"{base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {current_api_key}",
                "Content-Type": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=120, connect=30)

            try:
                async with self.session.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    response_body = await resp.text()

                    if resp.status == 429:
                        if provider_manager.is_early429(provider_id, current_api_key):
                            logger.warning(
                                "Early 429 (key count < limit), model-level rate limit"
                            )
                            return json.dumps(
                                {
                                    "error": {
                                        "message": "Model rate limited (early 429)",
                                        "type": "early_rate_limit",
                                    }
                                }
                            )
                        if attempt <= max_retries:
                            new_key = provider_manager.peek_next_key(
                                provider_id, current_api_key
                            )
                            if new_key and new_key != current_api_key:
                                current_api_key = new_key
                                logger.warning(
                                    "Rate limited (429), retry %d/%d with different key",
                                    attempt,
                                    max_retries,
                                )
                            else:
                                base_delay = 500 * (1 << (attempt - 1))
                                jitter = int(random.random() * base_delay * 0.5)
                                delay = base_delay + jitter
                                if total_sleep_ms + delay > max_total_sleep_ms:
                                    logger.warning(
                                        "Rate limited (429), total sleep limit reached, giving up"
                                    )
                                    return json.dumps(
                                        {
                                            "error": {
                                                "message": "Rate limited after retries",
                                                "type": "rate_limit_error",
                                            }
                                        }
                                    )
                                total_sleep_ms += delay
                                logger.warning(
                                    "Rate limited (429), retry %d/%d after %dms (same key)",
                                    attempt,
                                    max_retries,
                                    delay,
                                )
                                await asyncio.sleep(delay / 1000.0)
                            continue
                        return json.dumps(
                            {
                                "error": {
                                    "message": f"Rate limited after {max_retries} retries",
                                    "type": "rate_limit_error",
                                }
                            }
                        )

                    if not response_body:
                        return json.dumps(
                            {
                                "error": {
                                    "message": f"Empty response from upstream (HTTP {resp.status})",
                                    "type": "upstream_error",
                                }
                            }
                        )

                    if resp.status != 200:
                        if resp.status == 400:
                            logger.warning(
                                "Upstream 400 error from %s: %s",
                                provider_id,
                                response_body[:300],
                            )
                            try:
                                req_body = body if isinstance(body, dict) else {}
                                sent_params = [
                                    k for k in req_body.keys() if k != "messages"
                                ]
                                logger.warning(
                                    "Request params sent to %s: %s",
                                    provider_id,
                                    sent_params,
                                )
                            except Exception:
                                pass
                        else:
                            logger.warning(
                                "Upstream error HTTP %d: %s",
                                resp.status,
                                response_body[:200],
                            )
                        try:
                            json.loads(response_body)
                            return response_body
                        except Exception:
                            pass
                        return json.dumps(
                            {
                                "error": {
                                    "message": f"Upstream error HTTP {resp.status}: {response_body[:100]}",
                                    "type": "upstream_error",
                                }
                            }
                        )

                    return response_body

            except asyncio.TimeoutError:
                if attempt <= max_retries:
                    base_delay = 500 * (1 << (attempt - 1))
                    jitter = int(random.random() * base_delay * 0.5)
                    delay = base_delay + jitter
                    if total_sleep_ms + delay > max_total_sleep_ms:
                        logger.warning(
                            "Timeout retry, total sleep limit reached, giving up"
                        )
                        return json.dumps(
                            {
                                "error": {
                                    "message": "Connection timeout after retries",
                                    "type": "timeout_error",
                                }
                            }
                        )
                    total_sleep_ms += delay
                    logger.warning(
                        "Timeout on attempt %d/%d, retrying after %dms",
                        attempt,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay / 1000.0)
                    continue
                logger.exception("Forward to provider %s failed (timeout)", provider_id)
                return json.dumps(
                    {
                        "error": {
                            "message": "Connection timeout after retries",
                            "type": "timeout_error",
                        }
                    }
                )
            except Exception as e:
                logger.exception("Forward to provider %s failed", provider_id)
                return json.dumps(
                    {
                        "error": {
                            "message": f"Connection failed: {e}",
                            "type": "connection_error",
                        }
                    }
                )

    # ================================================================
    # 简单 API 处理器
    # ================================================================

    async def handle_models(self, request: web.Request) -> web.Response:
        """GET /v1/models - 聚合所有启用提供者的模型列表"""
        try:
            enabled_providers = provider_manager.get_enabled_providers()
            if not enabled_providers:
                return self._json_error("upstream_error", "No enabled providers")

            all_models: List[Dict[str, Any]] = []
            fetch_errors = 0

            async def fetch_provider_models(provider):
                nonlocal fetch_errors
                api_key = provider.api_keys[0] if provider.api_keys else ""
                if not api_key:
                    fetch_errors += 1
                    return

                url = f"{provider.base_url.rstrip('/')}/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                timeout = aiohttp.ClientTimeout(total=10, connect=5)

                try:
                    async with self.session.get(
                        url, headers=headers, timeout=timeout
                    ) as resp:
                        body = await resp.text()
                        if body:
                            try:
                                data = json.loads(body)
                                model_list = data.get("data", [])
                                for item in model_list:
                                    if isinstance(item, dict):
                                        model_id = item.get("id")
                                        if model_id:
                                            all_models.append(
                                                {
                                                    "id": model_id,
                                                    "object": "model",
                                                    "owned_by": item.get(
                                                        "owned_by", provider.name
                                                    ),
                                                    "provider": provider.id,
                                                    "provider_name": provider.name,
                                                }
                                            )
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(
                        "Failed to fetch models from provider %s: %s",
                        provider.id,
                        str(e),
                    )
                    fetch_errors += 1

            # 并发获取所有 provider 的模型
            await asyncio.gather(
                *[fetch_provider_models(p) for p in enabled_providers],
                return_exceptions=True,
            )

            if not all_models:
                return self._json_error(
                    "upstream_error", "No models available from any provider"
                )

            result = {"object": "list", "data": all_models}
            return self._json_ok(result)
        except Exception as e:
            logger.exception("Error fetching models")
            return self._json_error(
                "connection_error", f"Failed to fetch models: {e}"
            )

    async def handle_status(self, request: web.Request) -> web.Response:
        """GET /api/status"""
        group = config_manager.get_group_by_port(self.port)
        locked_model = router_state.get_locked_model(group)
        return self._json_ok(
            {
                "status": "running",
                "locked": locked_model is not None,
                "locked_model": locked_model,
                "group": group,
                "port": self.port,
            }
        )

    async def handle_speed_test(self, request: web.Request) -> web.Response:
        """POST /api/speed_test"""
        try:
            json_body = await request.json()
            model_id = json_body.get("model")
            if not model_id:
                return self._json_error("bad_request", "Model ID required")

            # 异步启动测速
            provider_id = self._get_provider_id_for_model(model_id)
            result = await self._speed_tester.test_model(model_id, provider_id)
            if result.success:
                router_state.update_speed_test_result(
                    model_id, int(result.response_time)
                )
            else:
                router_state.update_model_error(
                    model_id, result.error or "失败"
                )

            return self._json_ok({"status": "completed", "model": model_id})
        except Exception as e:
            return self._json_error("api_error", str(e))

    async def handle_lock(self, request: web.Request) -> web.Response:
        """POST /api/lock"""
        try:
            json_body = await request.json()
            model_id = json_body.get("model")
            if not model_id:
                return self._json_error("bad_request", "Model ID required")
            group_name = json_body.get("group") or config_manager.get_group_by_port(
                self.port
            )

            router_state.lock_model(group_name, model_id)
            return self._json_ok(
                {"locked": True, "model": model_id, "group": group_name}
            )
        except Exception as e:
            return self._json_error("api_error", str(e))

    async def handle_unlock(self, request: web.Request) -> web.Response:
        """POST /api/unlock"""
        group_name = config_manager.get_group_by_port(self.port)
        router_state.unlock_group(group_name)
        return self._json_ok({"locked": False, "group": group_name})

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health"""
        return self._json_ok(
            {"status": "healthy", "timestamp": int(time.time() * 1000)}
        )

    async def handle_get_config(self, request: web.Request) -> web.Response:
        """GET /api/config"""
        return self._json_ok(config_manager.get_config())

    async def handle_get_stats(self, request: web.Request) -> web.Response:
        """GET /api/stats"""
        return self._json_ok(stats_manager.get_stats())

    async def handle_dashboard_api(self, request: web.Request) -> web.Response:
        """GET /api/dashboard - 聚合所有运行时状态"""
        groups = config_manager.get_all_groups()
        locked_models = router_state.get_locked_models()
        speed_results = router_state.get_speed_test_results()
        availability = router_state.get_model_availability()
        model_stats = stats_manager.get_model_stats()

        groups_data = []
        for g in groups:
            locked_model = locked_models.get(g.name)
            current_model = locked_model or config_manager.select_fastest_model(
                g.name
            )

            models_data = []
            for m in g.models:
                if not m.enabled:
                    continue
                rt = speed_results.get(m.id)
                is_available = availability.get(m.id, True)
                provider_info = provider_manager.get_provider(m.provider_id)
                models_data.append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "provider": m.provider_id,
                        "provider_name": (
                            provider_info.name if provider_info else m.provider_id
                        ),
                        "status": {
                            "is_healthy": is_available,
                            "avg_response_time": rt,
                            "total_requests": model_stats.get(m.id, 0),
                        },
                    }
                )

            groups_data.append(
                {
                    "name": g.name,
                    "port": g.port,
                    "enabled": g.enabled,
                    "models": models_data,
                    "current_model": current_model or "",
                    "locked_model": locked_model or "",
                }
            )

        group_stats_map = {}
        for g in groups:
            stats = {}
            for m in g.models:
                if m.enabled:
                    stats[m.id] = model_stats.get(m.id, 0)
            group_stats_map[g.name] = stats

        lock_status_list = [
            {"group": group, "model_id": model_id, "locked": True}
            for group, model_id in locked_models.items()
        ]

        providers_data = []
        for p in provider_manager.get_all_providers():
            providers_data.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "base_url": p.base_url,
                    "rate_limit_type": p.effective_rate_limit_type.name,
                    "rate_limit_value": p.rate_limit_value,
                    "api_key_count": len(p.api_keys),
                    "model_count": len(p.models),
                    "enabled": p.enabled,
                    "request_counts": provider_manager.get_request_counts(p.id),
                }
            )

        group_current_model = {}
        for g in groups:
            group_current_model[g.name] = (
                locked_models.get(g.name)
                or config_manager.select_fastest_model(g.name)
                or ""
            )

        return self._json_ok(
            {
                "groups": groups_data,
                "api_call_stats": {
                    "total_calls": stats_manager.get_total_calls(),
                    "total_errors": stats_manager.get_total_errors(),
                    "group_stats": group_stats_map,
                },
                "lock_status": {
                    "locked_models": locked_models,
                    "locks": lock_status_list,
                },
                "group_current_model": group_current_model,
                "providers": providers_data,
            }
        )

    async def handle_reload(self, request: web.Request) -> web.Response:
        """POST /api/reload - 重载所有管理器"""
        config_manager.reload()
        provider_manager.reload()
        router_state.unlock_all()
        return self._json_ok({"success": True, "message": "配置重载成功"})

    async def handle_lock_model_api(self, request: web.Request) -> web.Response:
        """POST /api/lock_model"""
        try:
            json_body = await request.json()
            group = json_body.get(
                "group"
            ) or config_manager.get_group_by_port(self.port)
            model_id = json_body.get("model_id", "")

            if not model_id:
                return self._json_error("bad_request", "model_id required")

            router_state.lock_model(group, model_id)
            return self._json_ok(
                {
                    "success": True,
                    "locked": {"locked": True, "group": group, "model_id": model_id},
                }
            )
        except Exception as e:
            return self._json_error("api_error", str(e))

    async def handle_unlock_model_api(self, request: web.Request) -> web.Response:
        """POST /api/unlock_model"""
        group = config_manager.get_group_by_port(self.port)
        old = router_state.get_locked_model(group)
        router_state.unlock_group(group)
        return self._json_ok({"success": True, "unlocked": old, "group": group})

    async def handle_lock_status_api(self, request: web.Request) -> web.Response:
        """GET /api/lock_status"""
        group = config_manager.get_group_by_port(self.port)
        locked_model = router_state.get_locked_model(group)
        return self._json_ok(
            {
                "locked": locked_model is not None,
                "model_id": locked_model or "",
                "group": group,
            }
        )

    # ================================================================
    # 辅助方法
    # ================================================================

    def _get_model_to_use(
        self, json_body: dict, group_name: Optional[str]
    ) -> Optional[str]:
        """检查锁定模型，否则选择最快模型"""
        group = group_name or config_manager.get_group_by_port(self.port)
        locked_model = router_state.get_locked_model(group)
        if locked_model is not None:
            return locked_model
        return config_manager.select_fastest_model(group)

    def _get_provider_id_for_model(self, model_id: str) -> str:
        """查找模型所属的 provider_id"""
        groups = config_manager.get_all_groups()
        for group in groups:
            for m in group.models:
                if m.id == model_id:
                    return m.provider_id
        return provider_manager.get_provider_id_for_model(model_id)

    def _sanitize_request_body(self, body: dict) -> dict:
        """保留 OpenAI 标准参数，移除非标准参数"""
        keys_to_remove = [k for k in body.keys() if k not in OPENAI_STANDARD_PARAMS]
        if not keys_to_remove:
            return body
        cleaned = dict(body)
        for key in keys_to_remove:
            logger.debug("Sanitizing non-standard param: %s", key)
            cleaned.pop(key, None)
        return cleaned

    # ================================================================
    # 响应辅助
    # ================================================================

    @staticmethod
    def _json_ok(data: Any) -> web.Response:
        """返回 JSON 成功响应"""
        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps(data, ensure_ascii=False),
        )

    @staticmethod
    def _json_error(error_type: str, message: str) -> web.Response:
        """返回 JSON 错误响应（OpenAI 格式）"""
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps(
                {"error": {"message": message, "type": error_type}},
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def _anthropic_error(error_type: str, message: str) -> web.Response:
        """返回 Anthropic 格式错误响应

        格式: {"type": "error", "error": {"type": "...", "message": "..."}}
        HTTP 状态码根据错误类型映射
        """
        status_map = {
            "authentication_error": 401,
            "permission_error": 403,
            "not_found_error": 404,
            "rate_limit_error": 500,
            "invalid_request_error": 400,
            "overloaded_error": 503,
            "timeout_error": 503,
        }
        status = status_map.get(error_type, 500)
        return web.Response(
            status=status,
            content_type="application/json",
            text=json.dumps(
                {
                    "type": "error",
                    "error": {"type": error_type, "message": message},
                },
                ensure_ascii=False,
            ),
        )

    @staticmethod
    async def _write_sse(
        response: web.StreamResponse, event: str, data: Any
    ) -> None:
        """写入 SSE 事件到流式响应"""
        data_str = json.dumps(data, ensure_ascii=False)
        sse_msg = f"event: {event}\ndata: {data_str}\n\n"
        await response.write(sse_msg.encode("utf-8"))
