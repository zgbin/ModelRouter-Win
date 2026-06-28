"""
ModelRouter Windows - AI模型路由代理服务器
主入口：启动路由服务器、Web控制台、健康检查
"""
import asyncio
import signal
import sys
import os
import logging

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config_manager import config_manager
from core.provider_manager import provider_manager
from core.router_state import router_state
from core.stats_manager import stats_manager
from core.health_checker import health_checker
from server.router_server import _RouterServer
from server.web_console import web_console

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('ModelRouter')


async def run_startup_speed_test():
    """启动时对所有启用的模型执行一次测速"""
    from core.speed_tester import SpeedTester
    tester = SpeedTester()
    groups = config_manager.get_all_groups()
    tasks = []
    sem = asyncio.Semaphore(5)

    async def test_one(model_id, provider_id):
        async with sem:
            result = await tester.test_model(model_id, provider_id)
            if result.success:
                router_state.update_speed_test_result(model_id, result.response_time)
                logger.info(f"启动测速 {model_id}: {result.response_time}ms")
            else:
                router_state.update_model_error(model_id, result.error or "测速失败")
                logger.warning(f"启动测速 {model_id}: 失败 - {result.error}")

    for group in groups:
        if not group.enabled:
            continue
        for model in group.models:
            if not model.enabled:
                continue
            tasks.append(test_one(model.id, model.provider_id))

    if tasks:
        logger.info(f"开始启动测速，共 {len(tasks)} 个模型...")
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("启动测速完成")


async def main():
    logger.info("=" * 50)
    logger.info("ModelRouter v4.1 (Windows) 启动中...")
    logger.info("=" * 50)

    # 初始化配置
    groups = config_manager.get_all_groups()
    logger.info(f"已加载 {len(groups)} 个分组")
    providers = provider_manager.get_all_providers()
    logger.info(f"已加载 {len(providers)} 个提供商")

    # 收集启用的端口
    enabled_ports = [g.port for g in groups if g.enabled]
    logger.info(f"启用的端口: {enabled_ports}")

    # 启动路由服务器（每个启用的分组端口一个）
    servers = []
    for port in enabled_ports:
        server = _RouterServer(port)
        await server.start()
        servers.append(server)
        logger.info(f"路由服务器已启动 - 端口 {port}")

    # 启动Web控制台
    await web_console.start()
    console_port = web_console.port
    logger.info(f"Web控制台已启动 - http://127.0.0.1:{console_port}")

    # 启动健康检查
    health_checker.start()
    logger.info("健康检查器已启动")

    # 启动测速
    asyncio.create_task(run_startup_speed_test())

    logger.info("=" * 50)
    logger.info("ModelRouter 启动完成!")
    logger.info(f"API 端口: {', '.join(str(p) for p in enabled_ports)}")
    logger.info(f"控制台: http://127.0.0.1:{console_port}")
    logger.info("按 Ctrl+C 停止服务")
    logger.info("=" * 50)

    # 等待停止信号
    stop_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info("收到停止信号...")
        stop_event.set()

    # Windows 上 SIGINT 可以工作，SIGTERM 不一定
    signal.signal(signal.SIGINT, signal_handler)

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass

    # 优雅停止
    logger.info("正在停止服务...")
    health_checker.stop()
    for server in servers:
        await server.stop()
    await web_console.stop()
    logger.info("ModelRouter 已停止")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务已停止")
