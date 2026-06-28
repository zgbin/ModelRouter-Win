"""模型测速器 - ModelRouter Windows 应用

从 Android Kotlin SpeedTester 移植而来，使用 aiohttp 异步 HTTP 客户端。
通过发送流式请求并测量首字节时间（TTFT）来测试模型响应速度。
"""

import asyncio
import logging
import time

import aiohttp

from core.models import SpeedTestResult
from core.provider_manager import provider_manager

logger = logging.getLogger(__name__)


class SpeedTester:
    """模型测速器，测量模型首字节响应时间（TTFT）"""

    TIMEOUT_MS = 120_000

    async def test_model(self, model_id: str, provider_id: str = "nvidia") -> SpeedTestResult:
        """测试指定模型的响应速度

        Args:
            model_id: 模型 ID
            provider_id: 供应方 ID，默认 "nvidia"

        Returns:
            SpeedTestResult 测速结果
        """
        provider = provider_manager.get_provider(provider_id)
        if provider is None:
            return SpeedTestResult(
                model_id=model_id,
                response_time=0.0,
                success=False,
                error="供应方未找到",
            )

        base_url = provider.base_url
        api_key = provider.api_keys[0] if provider.api_keys else ""

        if not api_key:
            return SpeedTestResult(
                model_id=model_id,
                response_time=0.0,
                success=False,
                error="无可用 API Key",
            )

        url = f"{base_url.rstrip('/')}/chat/completions"
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=120, connect=30)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        error_label = {
                            429: "429 限流",
                            404: "404 未找到",
                            401: "401 认证失败",
                            403: "403 禁止访问",
                            500: "500 服务器错误",
                            502: "502 网关错误",
                            503: "503 服务不可用",
                        }.get(resp.status, f"HTTP {resp.status}")
                        ttft_ms = (time.time() - start_time) * 1000
                        await resp.read()
                        return SpeedTestResult(
                            model_id=model_id,
                            response_time=ttft_ms,
                            success=False,
                            error=error_label,
                        )

                    # 读取流式响应的第一个块，测量 TTFT
                    try:
                        chunk = await resp.content.read(512)
                        ttft_ms = (time.time() - start_time) * 1000
                    finally:
                        resp.release()

                    if not chunk:
                        return SpeedTestResult(
                            model_id=model_id,
                            response_time=ttft_ms,
                            success=False,
                            error="空流",
                        )

                    if ttft_ms <= self.TIMEOUT_MS:
                        return SpeedTestResult(
                            model_id=model_id,
                            response_time=ttft_ms,
                            success=True,
                        )
                    else:
                        return SpeedTestResult(
                            model_id=model_id,
                            response_time=ttft_ms,
                            success=False,
                            error="超时",
                        )

        except aiohttp.ClientConnectorError:
            ttft_ms = (time.time() - start_time) * 1000
            return SpeedTestResult(
                model_id=model_id,
                response_time=ttft_ms,
                success=False,
                error="连接失败",
            )
        except aiohttp.ClientConnectorDNSError:
            ttft_ms = (time.time() - start_time) * 1000
            return SpeedTestResult(
                model_id=model_id,
                response_time=ttft_ms,
                success=False,
                error="DNS解析失败",
            )
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
            ttft_ms = (time.time() - start_time) * 1000
            return SpeedTestResult(
                model_id=model_id,
                response_time=ttft_ms,
                success=False,
                error="超时",
            )
        except Exception:
            ttft_ms = (time.time() - start_time) * 1000
            logger.exception("Speed test failed for %s", model_id)
            return SpeedTestResult(
                model_id=model_id,
                response_time=ttft_ms,
                success=False,
                error="未知错误",
            )

