"""
Web 控制台后端 - ModelRouter Windows 应用

提供 REST API 用于 Web 管理控制台，替代 Android UI
(DashboardFragment, ConfigFragment, ApiKeysFragment, ModelsFragment)。
基于 aiohttp.web 实现 HTTP 服务，默认端口 8100。
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

from core.config_manager import config_manager
from core.models import (
    ConfigModelItem,
    GroupItem,
    KeySwitchStrategy,
    ProviderInfo,
    ProviderModel,
    RateLimitType,
)
from core.provider_manager import provider_manager
from core.router_state import router_state
from core.speed_tester import SpeedTester
from core.stats_manager import stats_manager

logger = logging.getLogger(__name__)

# Web 静态文件目录（相对于应用根目录）
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class _WebConsole:
    """Web 控制台 HTTP 服务"""

    def __init__(self, port: int = 8100):
        self.port = port
        self._runner: Optional[aiohttp.web.AppRunner] = None
        self._speed_tester = SpeedTester()

    # ================================================================
    # 服务器生命周期
    # ================================================================

    def create_app(self) -> web.Application:
        """创建 aiohttp 应用并注册所有路由"""
        app = web.Application()

        # ---- Dashboard APIs ----
        app.router.add_get("/api/console/dashboard", self.handle_dashboard)
        app.router.add_post("/api/console/speed_test", self.handle_speed_test)
        app.router.add_post("/api/console/batch_speed_test", self.handle_batch_speed_test)
        app.router.add_post("/api/console/lock_model", self.handle_lock_model)
        app.router.add_post("/api/console/unlock_model", self.handle_unlock_model)

        # ---- Config APIs (Groups) ----
        app.router.add_get("/api/console/groups", self.handle_get_groups)
        app.router.add_post("/api/console/groups", self.handle_add_group)
        app.router.add_put("/api/console/groups/{name}", self.handle_update_group)
        app.router.add_delete("/api/console/groups/{name}", self.handle_delete_group)
        app.router.add_post("/api/console/groups/{name}/models", self.handle_add_model_to_group)
        app.router.add_put("/api/console/groups/{name}/models/{model_id:.+}", self.handle_update_model_in_group)
        app.router.add_delete("/api/console/groups/{name}/models/{model_id:.+}", self.handle_remove_model_from_group)
        app.router.add_post("/api/console/groups/{name}/toggle", self.handle_toggle_group)
        app.router.add_post("/api/console/groups/{name}/models/{model_id:.+}/toggle", self.handle_toggle_model)
        app.router.add_post("/api/console/save_groups", self.handle_save_groups)

        # ---- Provider APIs ----
        app.router.add_get("/api/console/providers", self.handle_get_providers)
        app.router.add_post("/api/console/providers", self.handle_add_provider)
        app.router.add_put("/api/console/providers/{id}", self.handle_update_provider)
        app.router.add_delete("/api/console/providers/{id}", self.handle_delete_provider)
        app.router.add_post("/api/console/providers/{id}/keys", self.handle_add_api_key)
        app.router.add_delete("/api/console/providers/{id}/keys", self.handle_remove_api_key)
        app.router.add_put("/api/console/providers/{id}/rate_limit", self.handle_update_rate_limit)
        app.router.add_put("/api/console/providers/{id}/key_strategy", self.handle_update_key_strategy)
        app.router.add_put("/api/console/providers/{id}/base_url", self.handle_update_base_url)
        app.router.add_get("/api/console/providers/{id}/fetch_models", self.handle_fetch_provider_models)

        # ---- System APIs ----
        app.router.add_post("/api/console/reload", self.handle_reload)
        app.router.add_get("/api/console/version", self.handle_version)

        # ---- 静态文件服务 ----
        if os.path.isdir(_WEB_DIR):
            # 首页路由必须在 static 之前注册
            app.router.add_get("/", self.handle_index)
            app.router.add_static("/", _WEB_DIR, append_version=True)

        return app

    async def start(self) -> None:
        """启动 Web 控制台"""
        app = self.create_app()
        self._runner = aiohttp.web.AppRunner(app)
        await self._runner.setup()
        site = aiohttp.web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        logger.info("WebConsole started on port %d", self.port)

    async def stop(self) -> None:
        """停止 Web 控制台"""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        logger.info("WebConsole stopped")

    # ================================================================
    # Dashboard APIs
    # ================================================================

    async def handle_dashboard(self, request: web.Request) -> web.Response:
        """GET /api/console/dashboard - 聚合仪表盘数据"""
        try:
            groups = config_manager.get_all_groups()
            locked_models = router_state.get_locked_models()
            speed_results = router_state.get_speed_test_results()
            availability = router_state.get_model_availability()
            model_errors = router_state.get_model_errors()
            model_stats = stats_manager.get_model_stats()
            error_stats = stats_manager.get_error_stats()
            active_connections = router_state.get_active_connections_map()

            groups_data = []
            for g in groups:
                locked_model = locked_models.get(g.name)
                current_model = locked_model or config_manager.select_fastest_model(g.name)

                models_data = []
                for m in g.models:
                    rt = speed_results.get(m.id)
                    is_available = router_state.is_model_available(m.id)
                    error_msg = model_errors.get(m.id)
                    connections = active_connections.get(m.id, 0)
                    is_current = (m.id == current_model)
                    is_locked = (m.id == locked_model)
                    provider_info = provider_manager.get_provider(m.provider_id)

                    models_data.append({
                        "id": m.id,
                        "name": m.name,
                        "provider_id": m.provider_id,
                        "provider_name": provider_info.name if provider_info else m.provider_id,
                        "enabled": m.enabled,
                        "timeout": m.timeout,
                        "status": {
                            "is_healthy": is_available,
                            "response_time": rt,
                            "requests": model_stats.get(m.id, 0),
                            "errors": error_stats.get(m.id, 0),
                            "error_message": error_msg,
                            "active_connections": connections,
                            "is_current": is_current,
                            "is_locked": is_locked,
                        },
                    })

                groups_data.append({
                    "name": g.name,
                    "description": g.description,
                    "port": g.port,
                    "enabled": g.enabled,
                    "models": models_data,
                    "current_model": current_model or "",
                    "locked_model": locked_model or "",
                })

            # 按分组统计
            group_stats = {}
            for g in groups:
                g_calls = 0
                g_errors = 0
                for m in g.models:
                    if m.enabled:
                        g_calls += model_stats.get(m.id, 0)
                        g_errors += error_stats.get(m.id, 0)
                group_stats[g.name] = {"calls": g_calls, "errors": g_errors}

            # 锁定状态
            lock_status_list = [
                {"group": group, "model_id": mid, "locked": True}
                for group, mid in locked_models.items()
            ]

            # Provider 状态
            providers_data = []
            for p in provider_manager.get_all_providers():
                providers_data.append({
                    "id": p.id,
                    "name": p.name,
                    "base_url": p.base_url,
                    "rate_limit_type": p.effective_rate_limit_type.value,
                    "rate_limit_value": p.rate_limit_value,
                    "switch_threshold": p.switch_threshold,
                    "api_key_count": len(p.api_keys),
                    "model_count": len(p.models),
                    "enabled": p.enabled,
                    "is_default": p.is_default,
                    "request_counts": provider_manager.get_request_counts(p.id),
                })

            return self._json_ok({
                "groups": groups_data,
                "api_call_stats": {
                    "total_calls": stats_manager.get_total_calls(),
                    "total_errors": stats_manager.get_total_errors(),
                    "group_stats": group_stats,
                },
                "lock_status": {
                    "locked_models": locked_models,
                    "locks": lock_status_list,
                },
                "providers": providers_data,
                "active_connections": active_connections,
            })
        except Exception as e:
            logger.exception("Dashboard error")
            return self._json_error("api_error", str(e), 500)

    async def handle_speed_test(self, request: web.Request) -> web.Response:
        """POST /api/console/speed_test - 对指定模型执行测速"""
        try:
            body = await request.json()
            model_id = body.get("model_id", "")
            provider_id = body.get("provider_id", "")

            if not model_id:
                return self._json_error("bad_request", "model_id 必填", 400)

            if not provider_id:
                provider_id = self._find_provider_id_for_model(model_id)

            # 异步启动测速，立即返回
            asyncio.ensure_future(self._run_speed_test(model_id, provider_id))

            return self._json_ok({"status": "started", "model_id": model_id, "provider_id": provider_id})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_batch_speed_test(self, request: web.Request) -> web.Response:
        """POST /api/console/batch_speed_test - 对所有已启用模型执行批量测速（5并发）"""
        try:
            groups = config_manager.get_all_groups()
            tasks = []
            for g in groups:
                if not g.enabled:
                    continue
                for m in g.models:
                    if m.enabled:
                        provider_id = m.provider_id or self._find_provider_id_for_model(m.id)
                        tasks.append((m.id, provider_id))

            # 5 并发执行
            semaphore = asyncio.Semaphore(5)

            async def _limited_test(mid, pid):
                async with semaphore:
                    await self._run_speed_test(mid, pid)

            asyncio.ensure_future(self._run_batch_speed_test(tasks, _limited_test))

            return self._json_ok({
                "status": "started",
                "total_models": len(tasks),
            })
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_lock_model(self, request: web.Request) -> web.Response:
        """POST /api/console/lock_model - 锁定模型到分组"""
        try:
            body = await request.json()
            group = body.get("group", "")
            model_id = body.get("model_id", "")

            if not group or not model_id:
                return self._json_error("bad_request", "group 和 model_id 必填", 400)

            # 验证分组存在
            found_group = self._find_group(group)
            if found_group is None:
                return self._json_error("not_found", f"分组 '{group}' 不存在", 404)

            router_state.lock_model(group, model_id)
            return self._json_ok({"success": True, "group": group, "model_id": model_id})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_unlock_model(self, request: web.Request) -> web.Response:
        """POST /api/console/unlock_model - 解锁分组中的模型"""
        try:
            body = await request.json()
            group = body.get("group", "")

            if not group:
                return self._json_error("bad_request", "group 必填", 400)

            old = router_state.get_locked_model(group)
            router_state.unlock_group(group)
            return self._json_ok({"success": True, "group": group, "unlocked_model": old or ""})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    # ================================================================
    # Config APIs (Groups)
    # ================================================================

    async def handle_get_groups(self, request: web.Request) -> web.Response:
        """GET /api/console/groups - 获取所有分组及模型"""
        try:
            groups = config_manager.get_all_groups()
            return self._json_ok({
                "groups": [g.to_dict() for g in groups],
            })
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_add_group(self, request: web.Request) -> web.Response:
        """POST /api/console/groups - 添加新分组"""
        try:
            body = await request.json()
            name = body.get("name", "")
            if not name:
                return self._json_error("bad_request", "name 必填", 400)

            # 检查是否重名
            groups = config_manager.get_all_groups()
            if any(g.name == name for g in groups):
                return self._json_error("conflict", f"分组 '{name}' 已存在", 409)

            new_group = GroupItem(
                name=name,
                description=body.get("description", ""),
                port=body.get("port", 8195),
                models=[],
                enabled=body.get("enabled", True),
            )
            groups.append(new_group)
            config_manager.save_groups(groups)
            return self._json_ok({"success": True, "group": new_group.to_dict()})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_update_group(self, request: web.Request) -> web.Response:
        """PUT /api/console/groups/{name} - 更新分组"""
        try:
            name = request.match_info["name"]
            body = await request.json()

            groups = config_manager.get_all_groups()
            found = False
            for i, g in enumerate(groups):
                if g.name == name:
                    new_name = body.get("name", g.name)
                    groups[i] = GroupItem(
                        name=new_name,
                        description=body.get("description", g.description),
                        port=body.get("port", g.port),
                        models=g.models,
                        enabled=body.get("enabled", g.enabled),
                        is_backup=g.is_backup,
                    )
                    found = True
                    break

            if not found:
                return self._json_error("not_found", f"分组 '{name}' 不存在", 404)

            config_manager.save_groups(groups)
            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_delete_group(self, request: web.Request) -> web.Response:
        """DELETE /api/console/groups/{name} - 删除分组"""
        try:
            name = request.match_info["name"]
            groups = config_manager.get_all_groups()
            new_groups = [g for g in groups if g.name != name]

            if len(new_groups) == len(groups):
                return self._json_error("not_found", f"分组 '{name}' 不存在", 404)

            config_manager.save_groups(new_groups)
            router_state.unlock_group(name)
            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_add_model_to_group(self, request: web.Request) -> web.Response:
        """POST /api/console/groups/{name}/models - 向分组添加模型"""
        try:
            group_name = request.match_info["name"]
            body = await request.json()
            model_id = body.get("id", "") or body.get("model_id", "")

            if not model_id:
                return self._json_error("bad_request", "id 必填", 400)

            groups = config_manager.get_all_groups()
            found = False
            for i, g in enumerate(groups):
                if g.name == group_name:
                    # 检查模型是否已存在
                    if any(m.id == model_id for m in g.models):
                        return self._json_error("conflict", f"模型 '{model_id}' 已在该分组中", 409)

                    new_model = ConfigModelItem(
                        id=model_id,
                        name=body.get("name", model_id),
                        provider_id=body.get("provider_id", "nvidia"),
                        provider=body.get("provider_id", "nvidia"),
                        timeout=body.get("timeout", 60),
                        enabled=body.get("enabled", True),
                    )
                    groups[i] = GroupItem(
                        name=g.name,
                        description=g.description,
                        port=g.port,
                        models=g.models + [new_model],
                        enabled=g.enabled,
                        is_backup=g.is_backup,
                    )
                    found = True
                    break

            if not found:
                return self._json_error("not_found", f"分组 '{group_name}' 不存在", 404)

            config_manager.save_groups(groups)
            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_update_model_in_group(self, request: web.Request) -> web.Response:
        """PUT /api/console/groups/{name}/models/{model_id} - 更新分组中的模型"""
        try:
            group_name = request.match_info["name"]
            model_id = request.match_info["model_id"]
            body = await request.json()

            groups = config_manager.get_all_groups()
            found = False
            for gi, g in enumerate(groups):
                if g.name == group_name:
                    for mi, m in enumerate(g.models):
                        if m.id == model_id:
                            updated = ConfigModelItem(
                                id=m.id,
                                name=body.get("name", m.name),
                                priority=m.priority,
                                provider=body.get("provider_id", m.provider_id),
                                provider_id=body.get("provider_id", m.provider_id),
                                timeout=body.get("timeout", m.timeout),
                                endpoint=m.endpoint,
                                api_key=m.api_key,
                                enabled=body.get("enabled", m.enabled),
                                is_fallback=m.is_fallback,
                            )
                            new_models = list(g.models)
                            new_models[mi] = updated
                            groups[gi] = GroupItem(
                                name=g.name,
                                description=g.description,
                                port=g.port,
                                models=new_models,
                                enabled=g.enabled,
                                is_backup=g.is_backup,
                            )
                            found = True
                            break
                    break

            if not found:
                return self._json_error("not_found", f"分组 '{group_name}' 中的模型 '{model_id}' 不存在", 404)

            config_manager.save_groups(groups)
            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_remove_model_from_group(self, request: web.Request) -> web.Response:
        """DELETE /api/console/groups/{name}/models/{model_id} - 从分组中移除模型"""
        try:
            group_name = request.match_info["name"]
            model_id = request.match_info["model_id"]

            groups = config_manager.get_all_groups()
            found = False
            for gi, g in enumerate(groups):
                if g.name == group_name:
                    new_models = [m for m in g.models if m.id != model_id]
                    if len(new_models) == len(g.models):
                        return self._json_error("not_found", f"模型 '{model_id}' 不在该分组中", 404)
                    groups[gi] = GroupItem(
                        name=g.name,
                        description=g.description,
                        port=g.port,
                        models=new_models,
                        enabled=g.enabled,
                        is_backup=g.is_backup,
                    )
                    found = True
                    break

            if not found:
                return self._json_error("not_found", f"分组 '{group_name}' 不存在", 404)

            config_manager.save_groups(groups)
            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_toggle_group(self, request: web.Request) -> web.Response:
        """POST /api/console/groups/{name}/toggle - 切换分组启用状态"""
        try:
            group_name = request.match_info["name"]
            body = await request.json()
            enabled = body.get("enabled")

            if enabled is None:
                return self._json_error("bad_request", "enabled 必填", 400)

            groups = config_manager.get_all_groups()
            found = False
            for i, g in enumerate(groups):
                if g.name == group_name:
                    groups[i] = GroupItem(
                        name=g.name,
                        description=g.description,
                        port=g.port,
                        models=g.models,
                        enabled=enabled,
                        is_backup=g.is_backup,
                    )
                    found = True
                    break

            if not found:
                return self._json_error("not_found", f"分组 '{group_name}' 不存在", 404)

            config_manager.save_groups(groups)
            return self._json_ok({"success": True, "name": group_name, "enabled": enabled})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_toggle_model(self, request: web.Request) -> web.Response:
        """POST /api/console/groups/{name}/models/{model_id}/toggle - 切换模型启用状态"""
        try:
            group_name = request.match_info["name"]
            model_id = request.match_info["model_id"]
            body = await request.json()
            enabled = body.get("enabled")

            if enabled is None:
                return self._json_error("bad_request", "enabled 必填", 400)

            groups = config_manager.get_all_groups()
            found = False
            for gi, g in enumerate(groups):
                if g.name == group_name:
                    for mi, m in enumerate(g.models):
                        if m.id == model_id:
                            updated = ConfigModelItem(
                                id=m.id,
                                name=m.name,
                                priority=m.priority,
                                provider=m.provider,
                                provider_id=m.provider_id,
                                timeout=m.timeout,
                                endpoint=m.endpoint,
                                api_key=m.api_key,
                                enabled=enabled,
                                is_fallback=m.is_fallback,
                            )
                            new_models = list(g.models)
                            new_models[mi] = updated
                            groups[gi] = GroupItem(
                                name=g.name,
                                description=g.description,
                                port=g.port,
                                models=new_models,
                                enabled=g.enabled,
                                is_backup=g.is_backup,
                            )
                            found = True
                            break
                    break

            if not found:
                return self._json_error("not_found", f"分组 '{group_name}' 中的模型 '{model_id}' 不存在", 404)

            config_manager.save_groups(groups)
            return self._json_ok({"success": True, "model_id": model_id, "enabled": enabled})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_save_groups(self, request: web.Request) -> web.Response:
        """POST /api/console/save_groups - 保存完整分组配置并重载"""
        try:
            body = await request.json()
            groups_data = body.get("groups", [])
            groups = [GroupItem.from_dict(g) for g in groups_data]
            config_manager.save_groups(groups)
            config_manager.reload()
            return self._json_ok({"success": True, "message": "分组配置已保存并重载"})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    # ================================================================
    # Provider APIs
    # ================================================================

    async def handle_get_providers(self, request: web.Request) -> web.Response:
        """GET /api/console/providers - 获取所有 Provider"""
        try:
            providers = provider_manager.get_all_providers()
            return self._json_ok({
                "providers": [p.to_dict() for p in providers],
            })
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_add_provider(self, request: web.Request) -> web.Response:
        """POST /api/console/providers - 添加 Provider"""
        try:
            body = await request.json()
            pid = body.get("id", "")
            if not pid:
                return self._json_error("bad_request", "id 必填", 400)

            new_provider = ProviderInfo(
                id=pid,
                name=body.get("name", pid),
                base_url=body.get("base_url", ""),
                api_keys=body.get("api_keys", []),
                rate_limit_type=_enum_from_str(RateLimitType, body.get("rate_limit_type"), RateLimitType.PER_MINUTE),
                rate_limit_value=body.get("rate_limit_value", 40),
                switch_threshold=body.get("switch_threshold", 35),
                key_switch_strategy=_enum_from_str(KeySwitchStrategy, body.get("key_switch_strategy"), KeySwitchStrategy.THRESHOLD),
                models=[ProviderModel.from_dict(m) for m in body.get("models", [])],
                enabled=body.get("enabled", True),
                is_default=False,
            )

            if not provider_manager.add_provider(new_provider):
                return self._json_error("conflict", f"Provider '{pid}' 已存在", 409)

            return self._json_ok({"success": True, "provider": new_provider.to_dict()})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_update_provider(self, request: web.Request) -> web.Response:
        """PUT /api/console/providers/{id} - 更新 Provider"""
        try:
            pid = request.match_info["id"]
            body = await request.json()

            existing = provider_manager.get_provider(pid)
            if existing is None:
                return self._json_error("not_found", f"Provider '{pid}' 不存在", 404)

            updated = ProviderInfo(
                id=pid,
                name=body.get("name", existing.name),
                base_url=body.get("base_url", existing.base_url),
                api_keys=body.get("api_keys", existing.api_keys),
                rate_limit_type=_enum_from_str(RateLimitType, body.get("rate_limit_type"), existing.effective_rate_limit_type),
                rate_limit_value=body.get("rate_limit_value", existing.rate_limit_value),
                switch_threshold=body.get("switch_threshold", existing.switch_threshold),
                key_switch_strategy=_enum_from_str(KeySwitchStrategy, body.get("key_switch_strategy"), existing.effective_key_switch_strategy),
                models=[ProviderModel.from_dict(m) for m in body.get("models", [m.to_dict() for m in existing.models])],
                enabled=body.get("enabled", existing.enabled),
                is_default=existing.is_default,
            )

            if not provider_manager.update_provider(updated):
                return self._json_error("not_found", f"Provider '{pid}' 更新失败", 404)

            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_delete_provider(self, request: web.Request) -> web.Response:
        """DELETE /api/console/providers/{id} - 删除 Provider（不允许删除默认）"""
        try:
            pid = request.match_info["id"]
            existing = provider_manager.get_provider(pid)
            if existing is None:
                return self._json_error("not_found", f"Provider '{pid}' 不存在", 404)

            if existing.is_default:
                return self._json_error("forbidden", "不允许删除默认 Provider", 403)

            if not provider_manager.remove_provider(pid):
                return self._json_error("api_error", f"删除 Provider '{pid}' 失败", 500)

            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_add_api_key(self, request: web.Request) -> web.Response:
        """POST /api/console/providers/{id}/keys - 添加 API Key"""
        try:
            pid = request.match_info["id"]
            body = await request.json()
            key = body.get("key", "")

            if not key:
                return self._json_error("bad_request", "key 必填", 400)

            if not provider_manager.add_api_key(pid, key):
                return self._json_error("not_found", f"Provider '{pid}' 不存在或 key 已存在", 404)

            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_remove_api_key(self, request: web.Request) -> web.Response:
        """DELETE /api/console/providers/{id}/keys - 移除 API Key"""
        try:
            pid = request.match_info["id"]
            body = await request.json()
            key = body.get("key", "")

            if not key:
                return self._json_error("bad_request", "key 必填", 400)

            if not provider_manager.remove_api_key(pid, key):
                return self._json_error("not_found", "Key 不存在或 Provider 不存在或只剩一个 key", 404)

            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_update_rate_limit(self, request: web.Request) -> web.Response:
        """PUT /api/console/providers/{id}/rate_limit - 更新速率限制"""
        try:
            pid = request.match_info["id"]
            body = await request.json()
            rate_limit_type = _enum_from_str(RateLimitType, body.get("rate_limit_type"), RateLimitType.PER_MINUTE)
            rate_limit_value = body.get("rate_limit_value", 40)
            switch_threshold = body.get("switch_threshold", 35)

            if not provider_manager.update_rate_limit(pid, rate_limit_type, rate_limit_value, switch_threshold):
                return self._json_error("not_found", f"Provider '{pid}' 不存在", 404)

            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_update_key_strategy(self, request: web.Request) -> web.Response:
        """PUT /api/console/providers/{id}/key_strategy - 更新密钥切换策略"""
        try:
            pid = request.match_info["id"]
            body = await request.json()
            strategy = _enum_from_str(KeySwitchStrategy, body.get("strategy"), KeySwitchStrategy.THRESHOLD)

            if not provider_manager.update_key_switch_strategy(pid, strategy):
                return self._json_error("not_found", f"Provider '{pid}' 不存在", 404)

            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_update_base_url(self, request: web.Request) -> web.Response:
        """PUT /api/console/providers/{id}/base_url - 更新 Base URL"""
        try:
            pid = request.match_info["id"]
            body = await request.json()
            base_url = body.get("base_url", "")

            if not base_url:
                return self._json_error("bad_request", "base_url 必填", 400)

            if not provider_manager.update_base_url(pid, base_url):
                return self._json_error("not_found", f"Provider '{pid}' 不存在", 404)

            return self._json_ok({"success": True})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_fetch_provider_models(self, request: web.Request) -> web.Response:
        """GET /api/console/providers/{id}/fetch_models - 从 Provider API 获取可用模型列表"""
        try:
            pid = request.match_info["id"]
            provider = provider_manager.get_provider(pid)
            if provider is None:
                return self._json_error("not_found", f"Provider '{pid}' 不存在", 404)

            api_key = provider.api_keys[0] if provider.api_keys else ""
            if not api_key:
                return self._json_error("auth_error", "无可用 API Key", 401)

            url = f"{provider.base_url.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            timeout = aiohttp.ClientTimeout(total=15, connect=5)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return self._json_error(
                            "upstream_error",
                            f"获取模型列表失败 (HTTP {resp.status}): {error_text[:200]}",
                            502,
                        )
                    data = await resp.json()
                    model_list = data.get("data", [])
                    models = []
                    for item in model_list:
                        if isinstance(item, dict):
                            mid = item.get("id")
                            if mid:
                                models.append({
                                    "id": mid,
                                    "name": item.get("id", mid),
                                    "owned_by": item.get("owned_by", ""),
                                })

                    return self._json_ok({"models": models, "count": len(models)})
        except asyncio.TimeoutError:
            return self._json_error("timeout_error", "获取模型列表超时", 504)
        except aiohttp.ClientConnectorError:
            return self._json_error("connection_error", "连接 Provider 失败", 502)
        except Exception as e:
            logger.exception("Fetch provider models error")
            return self._json_error("api_error", str(e), 500)

    # ================================================================
    # System APIs
    # ================================================================

    async def handle_reload(self, request: web.Request) -> web.Response:
        """POST /api/console/reload - 重载所有配置"""
        try:
            config_manager.reload()
            provider_manager.reload()
            return self._json_ok({"success": True, "message": "配置已重载"})
        except Exception as e:
            return self._json_error("api_error", str(e), 500)

    async def handle_version(self, request: web.Request) -> web.Response:
        """GET /api/console/version - 获取版本信息"""
        return self._json_ok({
            "version": "4.1",
            "platform": "windows",
        })

    # ================================================================
    # 静态文件服务
    # ================================================================

    async def handle_index(self, request: web.Request) -> web.Response:
        """GET / - 返回 index.html"""
        index_path = os.path.join(_WEB_DIR, "index.html")
        if os.path.isfile(index_path):
            return web.FileResponse(index_path)
        return self._json_ok({"message": "ModelRouter Web Console", "version": "4.1"})

    # ================================================================
    # 内部辅助方法
    # ================================================================

    async def _run_speed_test(self, model_id: str, provider_id: str) -> None:
        """执行测速并更新状态"""
        try:
            result = await self._speed_tester.test_model(model_id, provider_id)
            if result.success:
                router_state.update_speed_test_result(model_id, int(result.response_time))
                logger.info("Speed test passed for %s: %.0fms", model_id, result.response_time)
            else:
                router_state.update_model_error(model_id, result.error or "测速失败")
                logger.warning("Speed test failed for %s: %s", model_id, result.error)
        except Exception:
            logger.exception("Speed test exception for %s", model_id)

    async def _run_batch_speed_test(self, tasks, limited_test_fn) -> None:
        """批量执行测速"""
        coros = [limited_test_fn(mid, pid) for mid, pid in tasks]
        await asyncio.gather(*coros, return_exceptions=True)
        logger.info("Batch speed test completed (%d models)", len(tasks))

    def _find_group(self, name: str) -> Optional[GroupItem]:
        """查找分组"""
        for g in config_manager.get_all_groups():
            if g.name == name:
                return g
        return None

    def _find_provider_id_for_model(self, model_id: str) -> str:
        """根据模型 ID 查找 Provider ID"""
        for g in config_manager.get_all_groups():
            for m in g.models:
                if m.id == model_id:
                    return m.provider_id
        return provider_manager.get_provider_id_for_model(model_id)

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
    def _json_error(error_type: str, message: str, status: int = 500) -> web.Response:
        """返回 JSON 错误响应"""
        return web.Response(
            status=status,
            content_type="application/json",
            text=json.dumps(
                {"error": {"type": error_type, "message": message}},
                ensure_ascii=False,
            ),
        )


def _enum_from_str(enum_cls, value, default):
    """从字符串安全还原枚举"""
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        try:
            return enum_cls[value]
        except KeyError:
            return default


# 模块级单例
web_console = _WebConsole()
