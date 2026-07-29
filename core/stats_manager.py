"""统计管理器 - 记录模型调用和错误统计"""

import threading
import time
from collections import deque


# 最近错误记录的最大条数
MAX_RECENT_ERRORS = 20


class _StatsManager:
    """模型调用统计管理器，线程安全"""

    def __init__(self):
        self._lock = threading.Lock()
        self.call_stats: dict[str, int] = {}
        self.error_stats: dict[str, int] = {}
        self.total_calls: int = 0
        self.total_errors: int = 0
        # 最近错误记录，每条包含 model_id, error, timestamp
        self._recent_errors: deque = deque(maxlen=MAX_RECENT_ERRORS)

    def record_call(self, model_id: str, success: bool, error_msg: str = "") -> None:
        """记录一次模型调用

        Args:
            model_id: 模型 ID
            success: 是否成功
            error_msg: 失败时的错误信息（可选）
        """
        with self._lock:
            self.total_calls += 1
            self.call_stats[model_id] = self.call_stats.get(model_id, 0) + 1
            if not success:
                self.total_errors += 1
                self.error_stats[model_id] = self.error_stats.get(model_id, 0) + 1
                self._recent_errors.append({
                    "model_id": model_id,
                    "error": error_msg or "未知错误",
                    "timestamp": time.strftime("%H:%M:%S"),
                })

    def get_stats(self) -> dict[str, int]:
        """返回调用统计的副本"""
        with self._lock:
            return self.call_stats.copy()

    def get_total_calls(self) -> int:
        """返回总调用次数"""
        with self._lock:
            return self.total_calls

    def get_total_errors(self) -> int:
        """返回总错误次数"""
        with self._lock:
            return self.total_errors

    def get_model_stats(self) -> dict[str, int]:
        """返回模型调用统计的副本"""
        with self._lock:
            return self.call_stats.copy()

    def get_error_stats(self) -> dict[str, int]:
        """返回模型错误统计的副本"""
        with self._lock:
            return self.error_stats.copy()

    def get_recent_errors(self, count: int = 5) -> list[dict]:
        """获取最近 N 条错误记录（按时间倒序）

        Args:
            count: 返回的最大条数

        Returns:
            错误记录列表，每条包含 model_id, error, timestamp
        """
        with self._lock:
            errors = list(self._recent_errors)
            # 返回最后 count 条，倒序（最新的在前）
            return errors[-count:][::-1]

    def cleanup_stale_data(self, active_model_ids: set[str]) -> None:
        """移除不在活跃模型集合中的统计条目"""
        with self._lock:
            stale_call = [k for k in self.call_stats if k not in active_model_ids]
            for k in stale_call:
                del self.call_stats[k]

            stale_error = [k for k in self.error_stats if k not in active_model_ids]
            for k in stale_error:
                del self.error_stats[k]

            # 清理最近错误中已不活跃的模型
            self._recent_errors = deque(
                (e for e in self._recent_errors if e["model_id"] in active_model_ids),
                maxlen=MAX_RECENT_ERRORS,
            )


stats_manager = _StatsManager()
