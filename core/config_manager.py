"""
配置管理器 - ModelRouter Windows 应用
从 Android Kotlin ConfigManager 移植而来，使用 JSON 文件存储替代 SharedPreferences。
"""

import json
import os
import threading
from typing import Any, Dict, List, Optional

from core.models import ConfigModelItem, GroupItem
from core.router_state import router_state


# 配置文件路径（相对于应用根目录）
_CONFIG_DIR = "config"
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "groups.json")


def _get_default_groups() -> List[GroupItem]:
    """返回默认分组配置（与 Android 版一致）"""
    return [
        GroupItem(
            name="综合对话组",
            description="通用对话和问答(全部支持tools)",
            port=8190,
            models=[
                ConfigModelItem(id="qwen/qwen3-next-80b-a3b-instruct", name="Qwen3 Next 80B (最快)", priority=1, provider="nvidia", timeout=30),
                ConfigModelItem(id="nvidia/llama-3.3-nemotron-super-49b-v1", name="Nemotron Super 49B", priority=2, provider="nvidia", timeout=30),
                ConfigModelItem(id="minimaxai/minimax-m2.5", name="MiniMax M2.5", priority=3, provider="nvidia", timeout=60),
                ConfigModelItem(id="stepfun-ai/step-3.5-flash", name="Step 3.5 Flash", priority=4, provider="nvidia", timeout=30),
                ConfigModelItem(id="meta/llama-3.1-70b-instruct", name="Llama 3.1 70B Instruct", priority=5, provider="nvidia", timeout=60),
            ],
        ),
        GroupItem(
            name="代码组",
            description="代码生成、审查、调试(全部支持tools)",
            port=8191,
            enabled=False,
            models=[
                ConfigModelItem(id="qwen/qwen3-coder-480b-a35b-instruct", name="Qwen3 Coder 480B (代码最强)", priority=1, provider="nvidia", timeout=90),
                ConfigModelItem(id="z-ai/glm-5.1", name="glm5.1", priority=2, provider="nvidia", timeout=60),
                ConfigModelItem(id="minimaxai/minimax-m2.5", name="MiniMax M2.5", priority=3, provider="nvidia", timeout=60),
                ConfigModelItem(id="minimaxai/minimax-m2.7", name="MiniMax M2.7", priority=4, provider="nvidia", timeout=60),
                ConfigModelItem(id="z-ai/glm5", name="glm5", priority=5, provider="nvidia", timeout=60),
            ],
        ),
        GroupItem(
            name="复杂组",
            description="复杂任务处理(全部支持tools)",
            port=8192,
            enabled=False,
            models=[
                ConfigModelItem(id="meta/llama-3.1-405b-instruct", name="Llama 3.1 405B Instruct", priority=1, provider="nvidia", timeout=90),
                ConfigModelItem(id="nvidia/llama-3.3-nemotron-super-49b-v1", name="Llama 3.3 Nemotron Super 49B V1", priority=2, provider="nvidia", timeout=60),
                ConfigModelItem(id="qwen/qwen3-next-80b-a3b-instruct", name="Qwen3 Next 80B", priority=3, provider="nvidia", timeout=60),
            ],
        ),
        GroupItem(
            name="图像组",
            description="图像解析(全部支持tools)",
            port=8193,
            enabled=False,
            models=[
                ConfigModelItem(id="qwen/qwen3-next-80b-a3b-instruct", name="Qwen3 Next 80B", priority=1, provider="nvidia", timeout=60),
                ConfigModelItem(id="stepfun-ai/step-3.5-flash", name="Step 3.5 Flash", priority=2, provider="nvidia", timeout=60),
                ConfigModelItem(id="minimaxai/minimax-m2.5", name="MiniMax M2.5", priority=3, provider="nvidia", timeout=60),
            ],
        ),
        GroupItem(
            name="语音处理",
            description="语音处理(全部支持tools)",
            port=8194,
            enabled=False,
            models=[
                ConfigModelItem(id="qwen/qwen3-next-80b-a3b-instruct", name="Qwen3 Next 80B", priority=1, provider="nvidia", timeout=60),
                ConfigModelItem(id="minimaxai/minimax-m2.5", name="MiniMax M2.5", priority=2, provider="nvidia", timeout=60),
                ConfigModelItem(id="stepfun-ai/step-3.5-flash", name="Step 3.5 Flash", priority=3, provider="nvidia", timeout=60),
            ],
        ),
    ]


class _ConfigManager:
    """配置管理器（线程安全），使用 JSON 文件持久化分组配置。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_groups: Optional[List[GroupItem]] = None

    # ── 内部辅助 ──────────────────────────────────────────────

    def _ensure_config_dir(self) -> None:
        """确保配置目录存在"""
        os.makedirs(_CONFIG_DIR, exist_ok=True)

    # ── 公开接口 ──────────────────────────────────────────────

    def load_groups(self) -> List[GroupItem]:
        """
        从 config/groups.json 加载分组配置。
        若文件不存在或内容为空，则使用默认配置并保存。
        """
        if not os.path.isfile(_CONFIG_FILE):
            groups = _get_default_groups()
            self._save_to_file(groups)
            return groups

        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                groups = _get_default_groups()
                self._save_to_file(groups)
                return groups
            groups = [GroupItem.from_dict(g) for g in data]
            if not groups:
                groups = _get_default_groups()
                self._save_to_file(groups)
            return groups
        except Exception:
            groups = _get_default_groups()
            self._save_to_file(groups)
            return groups

    def reload(self) -> None:
        """清除缓存，下次访问时强制重新读取文件"""
        with self._lock:
            self._cached_groups = None

    def get_groups(self) -> List[GroupItem]:
        """返回缓存的分组列表，若缓存为空则加载"""
        with self._lock:
            if self._cached_groups is None:
                self._cached_groups = self.load_groups()
            return self._cached_groups

    def save_groups(self, groups: List[GroupItem]) -> None:
        """保存分组列表到 JSON 文件并更新缓存"""
        with self._lock:
            self._save_to_file(groups)
            self._cached_groups = groups

    def get_all_groups(self) -> List[GroupItem]:
        """获取所有分组（get_groups 的别名）"""
        return self.get_groups()

    def get_default_group(self) -> str:
        """返回默认分组名称"""
        return "综合对话组"

    def get_group_by_port(self, port: int) -> str:
        """根据端口号查找分组名称，未找到则返回默认分组"""
        for g in self.get_groups():
            if g.port == port:
                return g.name
        return "综合对话组"

    def get_model_timeout(self, model_id: str) -> int:
        """在所有分组中查找模型的超时时间，未找到则返回 60"""
        for g in self.get_groups():
            for m in g.models:
                if m.id == model_id:
                    return m.timeout
        return 60

    def select_fastest_model(self, group_name: str) -> Optional[str]:
        """
        核心路由算法：在指定分组中选择最快的模型。
        1. 查找已启用的分组
        2. 筛选已启用的模型
        3. 从 router_state 获取速度测试结果
        4. 筛选响应时间在 0~120000ms 之间的模型
        5. 若无速度结果，则所有已启用模型作为候选
        6. 按响应时间排序，获取最佳时间
        7. 筛选最佳时间 1.5 倍范围内的候选
        8. 在其中选择活跃连接数最少的模型
        """
        group = None
        for g in self.get_groups():
            if g.name == group_name and g.enabled:
                group = g
                break
        if group is None:
            return None

        enabled_models = [m for m in group.models if m.enabled]
        if not enabled_models:
            return None

        speed_results = router_state.get_speed_test_results()

        available = [
            m for m in enabled_models
            if speed_results.get(m.id, -1) >= 0 and speed_results.get(m.id, -1) <= 120_000
        ]

        candidates = available if available else enabled_models

        sorted_by_speed = sorted(
            candidates,
            key=lambda m: speed_results.get(m.id, float("inf")),
        )

        best_time = speed_results.get(sorted_by_speed[0].id, float("inf"))

        near_best = [
            m for m in sorted_by_speed
            if speed_results.get(m.id) is None or speed_results.get(m.id, 0) <= best_time * 1.5
        ]

        best = min(near_best, key=lambda m: router_state.get_active_connections(m.id))
        return best.id

    def get_config(self) -> Dict[str, Any]:
        """返回分组配置的字典表示"""
        return {
            "groups": [g.to_dict() for g in self.get_groups()],
        }

    # ── 私有方法 ──────────────────────────────────────────────

    def _save_to_file(self, groups: List[GroupItem]) -> None:
        """将分组列表序列化为 JSON 并写入文件"""
        self._ensure_config_dir()
        data = [g.to_dict() for g in groups]
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 模块级单例
config_manager = _ConfigManager()
