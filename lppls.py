import numpy as np

def lppls_func(t, tc, m, w, A, B, C, phi):
    """
    LPPLS 核心函数
    t: 时间数组
    tc: 临界时间
    m: 幂律指数
    w: 对数周期频率
    A, B, C: 振幅参数
    phi: 相位
    """
    dt = np.maximum(tc - t, 1e-6)
    power_law = dt ** m
    log_periodic = np.cos(w * np.log(dt) + phi)
    return A + B * power_law + C * power_law * log_periodic