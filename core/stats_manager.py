"""统计管理器 - 记录模型调用和错误统计"""

import threading


class _StatsManager:
    """模型调用统计管理器，线程安全"""

    def __init__(self):
        self._lock = threading.Lock()
        self.call_stats: dict[str, int] = {}
        self.error_stats: dict[str, int] = {}
        self.total_calls: int = 0
        self.total_errors: int = 0

    def record_call(self, model_id: str, success: bool) -> None:
        """记录一次模型调用"""
        with self._lock:
            self.total_calls += 1
            self.call_stats[model_id] = self.call_stats.get(model_id, 0) + 1
            if not success:
                self.total_errors += 1
                self.error_stats[model_id] = self.error_stats.get(model_id, 0) + 1

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

    def cleanup_stale_data(self, active_model_ids: set[str]) -> None:
        """移除不在活跃模型集合中的统计条目"""
        with self._lock:
            stale_call = [k for k in self.call_stats if k not in active_model_ids]
            for k in stale_call:
                del self.call_stats[k]

            stale_error = [k for k in self.error_stats if k not in active_model_ids]
            for k in stale_error:
                del self.error_stats[k]


stats_manager = _StatsManager()
