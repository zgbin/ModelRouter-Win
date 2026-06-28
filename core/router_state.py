"""
运行时状态管理器 - ModelRouter Windows 应用
从 Android Kotlin RouterState 移植而来，使用线程安全的设计。
"""

import threading
import time


# 健康检查常量
HEALTH_CHECK_BASE_INTERVAL_MS = 120_000       # 2 分钟
HEALTH_CHECK_BACKOFF_THRESHOLD = 10            # 前 10 次使用固定间隔
HEALTH_CHECK_MAX_INTERVAL_MS = 1_800_000       # 30 分钟


class _RouterState:
    """路由器运行时状态管理（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()

        # group_name -> model_id
        self._locked_models: dict[str, str] = {}

        # model_id -> response_time_ms
        self._speed_test_results: dict[str, int] = {}

        # model_id -> error message
        self._model_errors: dict[str, str] = {}

        # model_id -> is_available
        self._model_availability: dict[str, bool] = {}

        # model_id -> connection count
        self._active_connections: dict[str, int] = {}

        # model_id -> failure count
        self._health_check_failures: dict[str, int] = {}

        # model_id -> next check timestamp (毫秒)
        self._next_health_check_time: dict[str, float] = {}

    # ── 连接管理 ──────────────────────────────────────────────

    def acquire_model(self, model_id: str) -> int:
        """增加模型的连接计数，返回更新后的计数"""
        with self._lock:
            count = self._active_connections.get(model_id, 0) + 1
            self._active_connections[model_id] = count
            return count

    def release_model(self, model_id: str) -> int:
        """减少模型的连接计数，减到 0 时移除条目，返回更新后的计数"""
        with self._lock:
            count = self._active_connections.get(model_id, 0) - 1
            if count <= 0:
                self._active_connections.pop(model_id, None)
                return 0
            self._active_connections[model_id] = count
            return count

    def get_active_connections(self, model_id: str) -> int:
        """获取指定模型的活跃连接数"""
        with self._lock:
            return self._active_connections.get(model_id, 0)

    def get_active_connections_map(self) -> dict[str, int]:
        """获取所有模型的活跃连接映射（副本）"""
        with self._lock:
            return dict(self._active_connections)

    # ── 模型锁定 ──────────────────────────────────────────────

    def lock_model(self, group_name: str, model_id: str) -> None:
        """为指定分组锁定模型"""
        with self._lock:
            self._locked_models[group_name] = model_id

    def unlock_group(self, group_name: str) -> None:
        """解除指定分组的模型锁定"""
        with self._lock:
            self._locked_models.pop(group_name, None)

    def unlock_all(self) -> None:
        """解除所有分组的模型锁定"""
        with self._lock:
            self._locked_models.clear()

    def get_locked_model(self, group_name: str) -> str | None:
        """获取指定分组锁定的模型 ID"""
        with self._lock:
            return self._locked_models.get(group_name)

    def get_locked_models(self) -> dict[str, str]:
        """获取所有分组锁定映射（副本）"""
        with self._lock:
            return dict(self._locked_models)

    # ── 速度测试 ──────────────────────────────────────────────

    def update_speed_test_result(self, model_id: str, response_time: int) -> None:
        """更新速度测试结果；若 response_time >= 0 则标记可用并清除错误"""
        with self._lock:
            self._speed_test_results[model_id] = response_time
            if response_time >= 0:
                self._model_availability[model_id] = True
                self._model_errors.pop(model_id, None)

    # ── 模型错误 ──────────────────────────────────────────────

    def update_model_error(self, model_id: str, error: str) -> None:
        """记录模型错误，标记不可用，若尚未安排健康检查则调度一次"""
        with self._lock:
            self._model_errors[model_id] = error
            self._speed_test_results[model_id] = -1
            self._model_availability[model_id] = False
            # 仅在尚未调度健康检查时安排
            if model_id not in self._next_health_check_time:
                now_ms = time.time() * 1000
                self._next_health_check_time[model_id] = now_ms + HEALTH_CHECK_BASE_INTERVAL_MS

    def clear_model_error(self, model_id: str) -> None:
        """清除模型错误，重置健康检查状态，标记可用"""
        with self._lock:
            self._model_errors.pop(model_id, None)
            self._health_check_failures.pop(model_id, None)
            self._next_health_check_time.pop(model_id, None)
            self._model_availability[model_id] = True

    # ── 健康检查 ──────────────────────────────────────────────

    def record_health_check_failure(self, model_id: str) -> float:
        """
        记录一次健康检查失败，计算下次检查间隔并调度。
        - 前 10 次失败：固定 2 分钟间隔
        - 超过 10 次：指数退避 2^n * 2 分钟，上限 30 分钟
        返回间隔毫秒数。
        """
        with self._lock:
            failures = self._health_check_failures.get(model_id, 0) + 1
            self._health_check_failures[model_id] = failures

            # 计算间隔
            if failures <= HEALTH_CHECK_BACKOFF_THRESHOLD:
                interval_ms = float(HEALTH_CHECK_BASE_INTERVAL_MS)
            else:
                exponent = failures - HEALTH_CHECK_BACKOFF_THRESHOLD
                interval_ms = min(
                    (2 ** exponent) * HEALTH_CHECK_BASE_INTERVAL_MS,
                    float(HEALTH_CHECK_MAX_INTERVAL_MS),
                )

            # 调度下次检查
            now_ms = time.time() * 1000
            self._next_health_check_time[model_id] = now_ms + interval_ms
            return interval_ms

    def get_models_needing_health_check(self) -> list[str]:
        """返回当前时间已超过下次健康检查时间的模型 ID 列表"""
        with self._lock:
            now_ms = time.time() * 1000
            return [
                model_id
                for model_id, next_time in self._next_health_check_time.items()
                if now_ms >= next_time
            ]

    def get_health_check_failures(self, model_id: str) -> int:
        """获取指定模型的健康检查连续失败次数"""
        with self._lock:
            return self._health_check_failures.get(model_id, 0)

    # ── 状态查询 ──────────────────────────────────────────────

    def get_model_errors(self) -> dict[str, str]:
        """获取所有模型错误映射（副本）"""
        with self._lock:
            return dict(self._model_errors)

    def get_model_error(self, model_id: str) -> str | None:
        """获取指定模型的错误信息"""
        with self._lock:
            return self._model_errors.get(model_id)

    def get_speed_test_results(self) -> dict[str, int]:
        """获取所有速度测试结果映射（副本）"""
        with self._lock:
            return dict(self._speed_test_results)

    def is_model_available(self, model_id: str) -> bool:
        """
        判断模型是否可用：
        - 无错误 且 (无速度测试结果 或 速度测试结果在 0~120000ms 之间)
        """
        with self._lock:
            has_error = model_id in self._model_errors
            if has_error:
                return False
            speed = self._speed_test_results.get(model_id)
            if speed is None:
                return True
            return 0 <= speed <= 120_000

    def get_model_availability(self) -> dict[str, bool]:
        """获取所有模型可用性映射（副本）"""
        with self._lock:
            return dict(self._model_availability)

    # ── 数据清理 ──────────────────────────────────────────────

    def cleanup_stale_data(self, active_model_ids: set[str]) -> None:
        """移除不在 active_model_ids 中的模型的所有状态条目"""
        with self._lock:
            for store in (
                self._speed_test_results,
                self._model_errors,
                self._model_availability,
                self._active_connections,
                self._health_check_failures,
                self._next_health_check_time,
            ):
                stale = [k for k in store if k not in active_model_ids]
                for k in stale:
                    store.pop(k, None)

            # 清理锁定映射中指向非活跃模型的条目
            stale_groups = [
                g for g, m in self._locked_models.items() if m not in active_model_ids
            ]
            for g in stale_groups:
                self._locked_models.pop(g, None)


# 模块级单例
router_state = _RouterState()
