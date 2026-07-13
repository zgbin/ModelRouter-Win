import asyncio
import logging

from core.router_state import router_state
from core.config_manager import config_manager
from core.speed_tester import SpeedTester

logger = logging.getLogger(__name__)


class _HealthChecker:
    SCAN_INTERVAL_S = 15  # seconds

    def __init__(self):
        self.running: bool = False
        self._task: asyncio.Task | None = None
        self._speed_tester = SpeedTester()

    def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info("HealthChecker started")

    def stop(self):
        self.running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.info("HealthChecker stopped")

    async def _check_loop(self):
        while self.running:
            try:
                await self._check_failed_models()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in health check loop")
            await asyncio.sleep(self.SCAN_INTERVAL_S)

    async def _check_failed_models(self):
        models = router_state.get_models_needing_health_check()
        if not models:
            return

        # Build model -> providerId map from config
        model_provider_map: dict[str, str] = {}
        for group in config_manager.get_all_groups():
            group_models = getattr(group, "models", None)
            if not isinstance(group_models, list):
                continue
            for model in group_models:
                if hasattr(model, "id"):
                    model_id = model.id
                    provider_id = getattr(model, "provider_id", None) or "nvidia"
                elif isinstance(model, dict):
                    model_id = model.get("id") or str(model)
                    provider_id = model.get("provider_id") or "nvidia"
                else:
                    model_id = str(model)
                    provider_id = "nvidia"
                model_provider_map[model_id] = provider_id

        for model_id in models:
            provider_id = model_provider_map.get(model_id, "nvidia")
            try:
                success, response_time = await self._speed_tester.test_model(model_id, provider_id)
                if success:
                    router_state.clear_model_error(model_id)
                    router_state.update_speed_test_result(model_id, response_time)
                    logger.info("Health check passed for model %s (response_time=%.2fs)", model_id, response_time)
                else:
                    router_state.record_health_check_failure(model_id)
                    logger.warning("Health check failed for model %s", model_id)
            except Exception:
                router_state.record_health_check_failure(model_id)
                logger.exception("Exception during health check for model %s", model_id)


health_checker = _HealthChecker()
