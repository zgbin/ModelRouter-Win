"""ModelRouter Windows 应用 - Provider/API Key 管理器

负责管理 API 提供商信息、密钥轮换、请求计数等。
从 Android Kotlin ProviderManager 移植，使用 JSON 文件替代 SharedPreferences。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional

from core.models import KeySwitchStrategy, ProviderInfo, ProviderModel, RateLimitType

# ============================================================
# 常量
# ============================================================

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PROVIDER_ID = "nvidia"
DEFAULT_SPEED_TEST_KEY = "nvapi-YOUR_KEY_HERE"
DEFAULT_WORK_KEY_1 = "nvapi-YOUR_KEY_HERE"
DEFAULT_WORK_KEY_2 = "nvapi-YOUR_KEY_HERE"
DEFAULT_WORK_KEY_3 = "nvapi-YOUR_KEY_HERE"

FIVE_HOURS_MS = 5 * 60 * 60 * 1000

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "providers.json")


# ============================================================
# 默认 Provider 创建
# ============================================================

def _create_default_providers() -> list[ProviderInfo]:
    return [
        ProviderInfo(
            id="nvidia",
            name="NVIDIA NIM",
            base_url=DEFAULT_BASE_URL,
            api_keys=[DEFAULT_WORK_KEY_1],
            rate_limit_type=RateLimitType.PER_MINUTE,
            rate_limit_value=40,
            switch_threshold=35,
            key_switch_strategy=KeySwitchStrategy.THRESHOLD,
            models=[
                ProviderModel("qwen/qwen3-next-80b-a3b-instruct", "Qwen3 Next 80B"),
                ProviderModel("nvidia/llama-3.3-nemotron-super-49b-v1", "Nemotron Super 49B"),
                ProviderModel("minimaxai/minimax-m2.5", "MiniMax M2.5"),
                ProviderModel("stepfun-ai/step-3.5-flash", "Step 3.5 Flash"),
                ProviderModel("meta/llama-3.1-70b-instruct", "Llama 3.1 70B"),
                ProviderModel("qwen/qwen3-coder-480b-a35b-instruct", "Qwen3 Coder 480B"),
                ProviderModel("meta/llama-3.1-405b-instruct", "Llama 3.1 405B"),
            ],
            enabled=True,
            is_default=True,
        ),
        ProviderInfo(
            id="agnes",
            name="Agnes AI",
            base_url="https://apihub.agnes-ai.com/v1",
            api_keys=[""],
            rate_limit_type=RateLimitType.UNLIMITED,
            rate_limit_value=0,
            switch_threshold=0,
            key_switch_strategy=KeySwitchStrategy.THRESHOLD,
            models=[
                ProviderModel("agnes-2.0-flash", "Agnes 2.0 Flash"),
                ProviderModel("agnes-1.5-flash", "Agnes 1.5 Flash"),
            ],
            enabled=True,
            is_default=True,
        ),
    ]


# ============================================================
# _ProviderManager 核心类
# ============================================================

class _ProviderManager:
    """Provider 管理器，线程安全，使用 JSON 文件持久化。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._providers: list[ProviderInfo] = []
        # provider_id -> { key -> count }
        self._provider_request_counts: Dict[str, Dict[str, int]] = {}
        # provider_id -> 上次重置时的分钟编号
        self._provider_last_reset_minute: Dict[str, int] = {}
        # provider_id -> 上次重置时的5小时编号
        self._provider_last_reset_5hour: Dict[str, int] = {}
        # provider_id -> 上次重置时的天编号
        self._provider_last_reset_day: Dict[str, int] = {}
        # provider_id -> 当前密钥索引
        self._provider_key_index: Dict[str, int] = {}

        self._load_providers()

    # ---- 持久化 ----

    def _load_providers(self) -> None:
        """从 config/providers.json 加载，或创建默认配置并保存。"""
        self._providers.clear()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    self._providers = [ProviderInfo.from_dict(item) for item in data]
                    return
            except Exception:
                pass
        # 加载失败或文件不存在，使用默认值
        self._providers = _create_default_providers()
        self._save_providers()

    def _save_providers(self) -> None:
        """将 providers 保存到 JSON 文件。"""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        data = [p.to_dict() for p in self._providers]
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_providers(self) -> None:
        """公开的保存接口。"""
        with self._lock:
            self._save_providers()

    # ---- 查询 ----

    def get_all_providers(self) -> list[ProviderInfo]:
        return list(self._providers)

    def get_provider(self, provider_id: str) -> Optional[ProviderInfo]:
        for p in self._providers:
            if p.id == provider_id:
                return p
        return None

    def get_enabled_providers(self) -> list[ProviderInfo]:
        return [p for p in self._providers if p.enabled]

    # ---- 增删改 ----

    def add_provider(self, provider: ProviderInfo) -> bool:
        with self._lock:
            if any(p.id == provider.id for p in self._providers):
                return False
            self._providers.append(provider)
            self._save_providers()
            return True

    def update_provider(self, provider: ProviderInfo) -> bool:
        with self._lock:
            for i, p in enumerate(self._providers):
                if p.id == provider.id:
                    self._providers[i] = provider
                    self._save_providers()
                    return True
            return False

    def remove_provider(self, provider_id: str) -> bool:
        with self._lock:
            provider = self.get_provider(provider_id)
            if provider is None:
                return False
            if provider.is_default:
                return False
            self._providers = [p for p in self._providers if p.id != provider_id]
            # 清理计数状态
            self._provider_request_counts.pop(provider_id, None)
            self._provider_last_reset_minute.pop(provider_id, None)
            self._provider_last_reset_5hour.pop(provider_id, None)
            self._provider_last_reset_day.pop(provider_id, None)
            self._provider_key_index.pop(provider_id, None)
            self._save_providers()
            return True

    # ---- API Key 管理 ----

    def add_api_key(self, provider_id: str, key: str) -> bool:
        with self._lock:
            for i, p in enumerate(self._providers):
                if p.id == provider_id:
                    if key in p.api_keys:
                        return False
                    self._providers[i] = ProviderInfo(
                        id=p.id, name=p.name, base_url=p.base_url,
                        api_keys=p.api_keys + [key],
                        rate_limit_type=p.rate_limit_type,
                        rate_limit_value=p.rate_limit_value,
                        switch_threshold=p.switch_threshold,
                        key_switch_strategy=p.key_switch_strategy,
                        models=p.models, enabled=p.enabled, is_default=p.is_default,
                    )
                    self._save_providers()
                    return True
            return False

    def remove_api_key(self, provider_id: str, key: str) -> bool:
        with self._lock:
            for i, p in enumerate(self._providers):
                if p.id == provider_id:
                    if len(p.api_keys) <= 1:
                        return False
                    new_keys = [k for k in p.api_keys if k != key]
                    if len(new_keys) == len(p.api_keys):
                        return False  # key 不存在
                    self._providers[i] = ProviderInfo(
                        id=p.id, name=p.name, base_url=p.base_url,
                        api_keys=new_keys,
                        rate_limit_type=p.rate_limit_type,
                        rate_limit_value=p.rate_limit_value,
                        switch_threshold=p.switch_threshold,
                        key_switch_strategy=p.key_switch_strategy,
                        models=p.models, enabled=p.enabled, is_default=p.is_default,
                    )
                    self._save_providers()
                    return True
            return False

    # ---- 配置更新 ----

    def update_rate_limit(self, provider_id: str, rate_limit_type: RateLimitType,
                          rate_limit_value: int, switch_threshold: int) -> bool:
        with self._lock:
            for i, p in enumerate(self._providers):
                if p.id == provider_id:
                    self._providers[i] = ProviderInfo(
                        id=p.id, name=p.name, base_url=p.base_url,
                        api_keys=p.api_keys,
                        rate_limit_type=rate_limit_type,
                        rate_limit_value=rate_limit_value,
                        switch_threshold=switch_threshold,
                        key_switch_strategy=p.key_switch_strategy,
                        models=p.models, enabled=p.enabled, is_default=p.is_default,
                    )
                    self._save_providers()
                    return True
            return False

    def update_key_switch_strategy(self, provider_id: str, strategy: KeySwitchStrategy) -> bool:
        with self._lock:
            for i, p in enumerate(self._providers):
                if p.id == provider_id:
                    self._providers[i] = ProviderInfo(
                        id=p.id, name=p.name, base_url=p.base_url,
                        api_keys=p.api_keys,
                        rate_limit_type=p.rate_limit_type,
                        rate_limit_value=p.rate_limit_value,
                        switch_threshold=p.switch_threshold,
                        key_switch_strategy=strategy,
                        models=p.models, enabled=p.enabled, is_default=p.is_default,
                    )
                    self._save_providers()
                    return True
            return False

    def update_models(self, provider_id: str, models: list[ProviderModel]) -> bool:
        with self._lock:
            for i, p in enumerate(self._providers):
                if p.id == provider_id:
                    self._providers[i] = ProviderInfo(
                        id=p.id, name=p.name, base_url=p.base_url,
                        api_keys=p.api_keys,
                        rate_limit_type=p.rate_limit_type,
                        rate_limit_value=p.rate_limit_value,
                        switch_threshold=p.switch_threshold,
                        key_switch_strategy=p.key_switch_strategy,
                        models=models, enabled=p.enabled, is_default=p.is_default,
                    )
                    self._save_providers()
                    return True
            return False

    def update_base_url(self, provider_id: str, base_url: str) -> bool:
        with self._lock:
            for i, p in enumerate(self._providers):
                if p.id == provider_id:
                    self._providers[i] = ProviderInfo(
                        id=p.id, name=p.name, base_url=base_url,
                        api_keys=p.api_keys,
                        rate_limit_type=p.rate_limit_type,
                        rate_limit_value=p.rate_limit_value,
                        switch_threshold=p.switch_threshold,
                        key_switch_strategy=p.key_switch_strategy,
                        models=p.models, enabled=p.enabled, is_default=p.is_default,
                    )
                    self._save_providers()
                    return True
            return False

    # ---- 计数器重置 ----

    def _reset_counters_if_needed(self, provider_id: str, rate_limit_type: RateLimitType) -> None:
        """根据 rate_limit_type 检查是否需要重置请求计数器。"""
        # 注意：此方法需在 _lock 内调用
        counts = self._provider_request_counts.setdefault(provider_id, {})
        now_ms = int(time.time() * 1000)

        if rate_limit_type == RateLimitType.PER_MINUTE:
            current_minute = now_ms // 60000
            last_reset = self._provider_last_reset_minute.setdefault(provider_id, current_minute)
            if current_minute != last_reset:
                self._provider_last_reset_minute[provider_id] = current_minute
                counts.clear()

        elif rate_limit_type == RateLimitType.PER_5_HOURS:
            current_5hour = now_ms // FIVE_HOURS_MS
            last_reset = self._provider_last_reset_5hour.setdefault(provider_id, current_5hour)
            if current_5hour != last_reset:
                self._provider_last_reset_5hour[provider_id] = current_5hour
                counts.clear()

        elif rate_limit_type == RateLimitType.PER_DAY:
            current_day = now_ms // 86_400_000
            last_reset = self._provider_last_reset_day.setdefault(provider_id, current_day)
            if current_day != last_reset:
                self._provider_last_reset_day[provider_id] = current_day
                counts.clear()

        # UNLIMITED: 不需要重置

    def reset_counters_if_needed(self, provider_id: str) -> None:
        """公开接口：检查并重置计数器。"""
        with self._lock:
            provider = self.get_provider(provider_id)
            if provider is None:
                return
            self._reset_counters_if_needed(provider_id, provider.effective_rate_limit_type)

    # ---- 密钥轮换 ----

    def get_next_key(self, provider_id: str) -> str:
        """获取下一个可用的 API Key，实现轮换逻辑。"""
        with self._lock:
            provider = self.get_provider(provider_id)
            if provider is None:
                return ""
            keys = provider.api_keys
            if not keys:
                return ""
            if len(keys) == 1:
                return keys[0]

            strategy = provider.effective_key_switch_strategy

            # EVERY_REQUEST 策略：轮询
            if strategy == KeySwitchStrategy.EVERY_REQUEST:
                idx = self._provider_key_index.get(provider_id, 0)
                current_idx = idx % len(keys)
                self._provider_key_index[provider_id] = idx + 1
                return keys[current_idx]

            # THRESHOLD 策略
            self._reset_counters_if_needed(provider_id, provider.effective_rate_limit_type)

            counts = self._provider_request_counts.setdefault(provider_id, {})
            idx = self._provider_key_index.get(provider_id, 0)
            threshold = provider.switch_threshold

            current_idx = idx % len(keys)
            current_key = keys[current_idx]
            current_count = counts.get(current_key, 0)

            if provider.effective_rate_limit_type == RateLimitType.UNLIMITED or current_count < threshold:
                counts[current_key] = current_count + 1
                return current_key

            # 当前 key 已达阈值，尝试寻找未达阈值的 key
            for attempt in range(len(keys)):
                next_idx = (current_idx + 1 + attempt) % len(keys)
                next_key = keys[next_idx]
                next_count = counts.get(next_key, 0)
                if next_count < threshold:
                    self._provider_key_index[provider_id] = next_idx
                    counts[next_key] = next_count + 1
                    return next_key

            # 所有 key 都达到阈值，回退到下一个 key
            fallback_idx = (current_idx + 1) % len(keys)
            self._provider_key_index[provider_id] = fallback_idx
            fallback_key = keys[fallback_idx]
            counts[fallback_key] = counts.get(fallback_key, 0) + 1
            return fallback_key

    def peek_next_key(self, provider_id: str, exclude_key: str) -> str:
        """查找一个与 exclude_key 不同的、未达阈值的 key。"""
        with self._lock:
            provider = self.get_provider(provider_id)
            if provider is None:
                return ""
            keys = provider.api_keys
            if not keys:
                return ""

            self._reset_counters_if_needed(provider_id, provider.effective_rate_limit_type)

            counts = self._provider_request_counts.setdefault(provider_id, {})
            threshold = provider.switch_threshold

            for idx, key in enumerate(keys):
                if key == exclude_key:
                    continue
                count = counts.get(key, 0)
                if provider.effective_rate_limit_type == RateLimitType.UNLIMITED or count < threshold:
                    self._provider_key_index[provider_id] = idx
                    counts[key] = count + 1
                    return key

            # 所有其他 key 都达到阈值，选第一个不同的 key
            alt_key = ""
            for key in keys:
                if key != exclude_key:
                    alt_key = key
                    break
            if alt_key:
                alt_idx = keys.index(alt_key) % len(keys)
                self._provider_key_index[provider_id] = alt_idx
                counts[alt_key] = counts.get(alt_key, 0) + 1
            return alt_key

    def is_early429(self, provider_id: str, current_key: str) -> bool:
        """判断是否为"提前429"（密钥级限流 vs 模型级限流）。

        返回 True 表示当前 key 的请求计数小于 rate_limit_value，
        意味着是模型级别的限流，而非 key 级别。
        """
        with self._lock:
            provider = self.get_provider(provider_id)
            if provider is None:
                return True
            self._reset_counters_if_needed(provider_id, provider.effective_rate_limit_type)
            counts = self._provider_request_counts.get(provider_id)
            if counts is None:
                return True
            count = counts.get(current_key, 0)
            return count < provider.rate_limit_value

    # ---- 状态查询 ----

    def get_request_counts(self, provider_id: str) -> Dict[str, int]:
        provider = self.get_provider(provider_id)
        if provider is None:
            return {}
        with self._lock:
            self._reset_counters_if_needed(provider_id, provider.effective_rate_limit_type)
            counts = self._provider_request_counts.get(provider_id, {})
            return dict(counts)

    def get_provider_status(self, provider_id: str) -> Optional[dict]:
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
        with self._lock:
            self._reset_counters_if_needed(provider_id, provider.effective_rate_limit_type)
            counts = self._provider_request_counts.get(provider_id, {})
            return {
                "id": provider.id,
                "name": provider.name,
                "base_url": provider.base_url,
                "rate_limit_type": provider.effective_rate_limit_type.name,
                "rate_limit_value": provider.rate_limit_value,
                "switch_threshold": provider.switch_threshold,
                "api_key_count": len(provider.api_keys),
                "model_count": len(provider.models),
                "enabled": provider.enabled,
                "request_counts": dict(counts),
            }

    def get_provider_id_for_model(self, model_id: str) -> str:
        """查找拥有指定 model_id 的 provider。"""
        for provider in self._providers:
            if any(m.id == model_id for m in provider.models):
                return provider.id
        return DEFAULT_PROVIDER_ID

    def get_speed_test_key(self, provider_id: str) -> str:
        """返回 provider 的第一个 key，用于速度测试。"""
        provider = self.get_provider(provider_id)
        if provider is None or not provider.api_keys:
            return ""
        return provider.api_keys[0]

    def reload(self) -> None:
        """从文件重新加载 providers。"""
        with self._lock:
            self._load_providers()


# ============================================================
# 模块级单例
# ============================================================

provider_manager = _ProviderManager()
