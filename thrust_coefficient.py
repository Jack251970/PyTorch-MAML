import numpy as np
import pandas as pd


def get_thrust_coefficient(wind_speed):
    # 定义查找表 (Senvion MM82 近似数据)
    ct_table = pd.DataFrame({
        'ws': [0, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25],
        'ct': [0, 0, 0.82, 0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.73, 0.68, 0.58, 0.45, 0.35, 0.28, 0.22, 0.15, 0.11,
               0.06]
    })

    # 简单的线性插值
    return np.interp(wind_speed, ct_table['ws'], ct_table['ct'])


# 示例：获取 8.5 m/s 时的推力系数
# current_ct = get_thrust_coefficient(8.5)
# print(f"Wind Speed 8.5 m/s -> Ct: {current_ct:.3f}")
