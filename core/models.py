"""ModelRouter Windows 应用 - 数据模型定义"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# 枚举类型
# ============================================================

class RateLimitType(Enum):
    UNLIMITED = "UNLIMITED"
    PER_MINUTE = "PER_MINUTE"
    PER_5_HOURS = "PER_5_HOURS"
    PER_DAY = "PER_DAY"


class KeySwitchStrategy(Enum):
    THRESHOLD = "THRESHOLD"
    EVERY_REQUEST = "EVERY_REQUEST"


# ============================================================
# 辅助：枚举反序列化
# ============================================================

def _enum_from_str(enum_cls: type, value: Any, default: Enum) -> Enum:
    """从字符串/枚举值安全还原枚举，支持大小写不敏感，失败返回 default。"""
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    # 先精确匹配
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        pass
    # 按名称匹配
    try:
        return enum_cls[value]
    except KeyError:
        pass
    # 大小写不敏感匹配
    upper = str(value).upper()
    for member in enum_cls:
        if member.value.upper() == upper or member.name.upper() == upper:
            return member
    return default


# ============================================================
# 数据模型
# ============================================================

@dataclass
class ProviderModel:
    id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProviderModel:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
        )


@dataclass
class ProviderInfo:
    id: str
    name: str
    base_url: str
    api_keys: List[str] = field(default_factory=list)
    rate_limit_type: Optional[RateLimitType] = None
    rate_limit_value: int = 40
    switch_threshold: int = 35
    key_switch_strategy: Optional[KeySwitchStrategy] = None
    models: List[ProviderModel] = field(default_factory=list)
    enabled: bool = True
    is_default: bool = False

    # ---- 带默认值的属性 ----
    @property
    def effective_rate_limit_type(self) -> RateLimitType:
        """返回 rate_limit_type，若为 None 则默认 PER_MINUTE。"""
        if self.rate_limit_type is None:
            return RateLimitType.PER_MINUTE
        return self.rate_limit_type

    @property
    def effective_key_switch_strategy(self) -> KeySwitchStrategy:
        """返回 key_switch_strategy，若为 None 则默认 THRESHOLD。"""
        if self.key_switch_strategy is None:
            return KeySwitchStrategy.THRESHOLD
        return self.key_switch_strategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "api_keys": list(self.api_keys),
            "rate_limit_type": self.effective_rate_limit_type.value,
            "rate_limit_value": self.rate_limit_value,
            "switch_threshold": self.switch_threshold,
            "key_switch_strategy": self.effective_key_switch_strategy.value,
            "models": [m.to_dict() for m in self.models],
            "enabled": self.enabled,
            "is_default": self.is_default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProviderInfo:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            base_url=data.get("base_url", ""),
            api_keys=list(data.get("api_keys", [])),
            rate_limit_type=_enum_from_str(
                RateLimitType, data.get("rate_limit_type"), RateLimitType.PER_MINUTE
            ),
            rate_limit_value=data.get("rate_limit_value", 40),
            switch_threshold=data.get("switch_threshold", 35),
            key_switch_strategy=_enum_from_str(
                KeySwitchStrategy, data.get("key_switch_strategy"), KeySwitchStrategy.THRESHOLD
            ),
            models=[ProviderModel.from_dict(m) for m in data.get("models", [])],
            enabled=data.get("enabled", True),
            is_default=data.get("is_default", False),
        )


@dataclass
class ConfigModelItem:
    id: str
    name: str
    priority: int = 1
    provider: str = "nvidia"
    provider_id: str = "nvidia"
    timeout: int = 60
    endpoint: str = ""
    api_key: str = ""
    enabled: bool = True
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "provider": self.provider,
            "provider_id": self.provider_id,
            "timeout": self.timeout,
            "endpoint": self.endpoint,
            "api_key": self.api_key,
            "enabled": self.enabled,
            "is_fallback": self.is_fallback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConfigModelItem:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            priority=data.get("priority", 1),
            provider=data.get("provider", "nvidia"),
            provider_id=data.get("provider_id", "nvidia"),
            timeout=data.get("timeout", 60),
            endpoint=data.get("endpoint", ""),
            api_key=data.get("api_key", ""),
            enabled=data.get("enabled", True),
            is_fallback=data.get("is_fallback", False),
        )


@dataclass
class GroupItem:
    name: str
    description: str = ""
    port: int = 8190
    models: List[ConfigModelItem] = field(default_factory=list)
    enabled: bool = True
    is_backup: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "port": self.port,
            "models": [m.to_dict() for m in self.models],
            "enabled": self.enabled,
            "is_backup": self.is_backup,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroupItem:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            port=data.get("port", 8190),
            models=[ConfigModelItem.from_dict(m) for m in data.get("models", [])],
            enabled=data.get("enabled", True),
            is_backup=data.get("is_backup", False),
        )


@dataclass
class SpeedTestResult:
    model_id: str
    response_time: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "response_time": self.response_time,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpeedTestResult:
        return cls(
            model_id=data.get("model_id", ""),
            response_time=data.get("response_time", 0.0),
            success=data.get("success", False),
            error=data.get("error"),
            timestamp=data.get("timestamp", time.time()),
        )
