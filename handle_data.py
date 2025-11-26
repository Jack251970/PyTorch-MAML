import os

last_colum_number = 0

a = [f"Turbine_Data_Penmanshiel_{i:02d}_2022-01-01_-_2023-01-01.csv"
     for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
for item in a:
    path = os.path.join('./.materials/Penmanshiel_SCADA_2022_WT01-15/', item)
    if os.path.exists(path):
        print(f"{item} Exists")

        import pandas as pd


        def load_selected_features(csv_path):
            # 读取数据
            df = pd.read_csv(csv_path)

            # 关键词映射：每类特征对应关键字列表（不区分大小写）
            keywords = {
                "time": ["time", "date"],
                "wind_speed": ["wind speed", "density adjusted wind"],
                "wind_direction": ["wind direction"],
                "nacelle": ["nacelle position"],
                # "setpoint": ["power setpoint"],
                # "capacity": ["available capacity"],
                # "power_factor": ["power factor"],
                # "reactive_power": ["reactive power"],
                # "voltage": ["voltage"],
                # "current": ["current"],
                "ambient_temp": ["ambient temperature"],
                "nacelle_temp": ["nacelle temperature"],
                "stator_temp": ["stator temperature"],
                "gear_oil_temp": ["gear oil temperature"],
            }

            # 自动筛选字段
            selected_columns = []

            for col in df.columns:
                col_low = col.lower()
                for group_keywords in keywords.values():
                    if any(kw in col_low for kw in group_keywords):
                        selected_columns.append(col)
                        break

            selected_columns.append('Power (kW)')

            # 按照你需要的字段顺序排序（时间字段放到最前）
            time_cols = [c for c in selected_columns if "time" in c.lower() or "date" in c.lower()]
            other_cols = [c for c in selected_columns if c not in time_cols]

            selected_df = df[time_cols + other_cols]

            return selected_df

        filtered_df = load_selected_features(path)

        # 检查一列中是否有超过50%的缺失值，并删除这些列
        threshold = 0.5 * len(filtered_df)
        filtered_df = filtered_df.loc[:, filtered_df.isnull().sum() <= threshold]

        colum_number = len(filtered_df.columns)
        if last_colum_number != 0 and colum_number != last_colum_number:
            print(f"Column number changed: {last_colum_number} -> {colum_number}")

        head = filtered_df.head()
        print(head)

        # save to new csv
        save_path = f'./.materials/Penmanshiel_SCADA_2022_WT01-15/filtered_{item}'
        filtered_df.to_csv(save_path, index=False)
    else:
        print(f"{item} Not Exists")
