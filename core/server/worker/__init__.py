# coding: utf-8
"""
识别子进程工作包 (Worker Package)

包含模型加载、任务处理和 Worker 门面类。
"""

from multiprocessing import Queue
from multiprocessing.managers import ListProxy
from .. import logger
from .worker import RecognizerWorker

def start_worker(
    queue_in: Queue,
    queue_out: Queue,
    sockets_id: ListProxy,
    stdin_fn: int,
    active_inference=None,
    gpu_boost_config=None,
):
    """识别子进程启动入口"""
    if gpu_boost_config is not None:
        # Windows spawn imports config_server afresh instead of inheriting the
        # parent's resolved environment/saved settings or Python overrides.
        from config_server import ServerConfig
        for name in ("gpu_boost_enabled", "gpu_boost_backend_active", "gpu_boost_cmd", "gpu_unboost_cmd",
                     "gpu_unboost_timeout"):
            if name in gpu_boost_config:
                setattr(ServerConfig, name, gpu_boost_config[name])
    worker = RecognizerWorker(
        queue_in,
        queue_out,
        sockets_id,
        stdin_fn,
        active_inference=active_inference,
    )
    worker.run()

__all__ = ['RecognizerWorker', 'start_worker']
