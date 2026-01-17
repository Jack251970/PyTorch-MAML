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


# 绘制出speed与trust coefficient的关系曲线
if __name__ == "__main__":
    # 示例：获取 8.5 m/s 时的推力系数
    current_ct = get_thrust_coefficient(8.5)
    print(f"Wind Speed 8.5 m/s -> Ct: {current_ct:.3f}")

    import matplotlib.pyplot as plt

    wind_speeds = np.linspace(0, 25, 100)
    ct_values = get_thrust_coefficient(wind_speeds)

    plt.plot(wind_speeds, ct_values, label='Thrust Coefficient (Ct)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Thrust Coefficient (Ct)')
    plt.title('Wind Speed vs Thrust Coefficient for Senvion MM82')
    plt.grid()
    plt.legend()
    plt.show()
