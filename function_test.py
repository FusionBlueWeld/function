# import math
# import numpy as np
# import matplotlib.pyplot as plt

# # triggerの定義
# trigger = {
#     "x1": "x_var",
#     "x2": "y_var",
#     "x3": 1.0,
#     "x4": 1000
# }

# # 各変数の範囲の定義
# x1_values = np.linspace(0, 800, 100)    # welding_speedの範囲
# x2_values = np.linspace(-6, 6, 100)     # head_positionの範囲
# x3_values = np.arange(0, 7, 0.1)        # material_thicknessの範囲
# x4_values = np.arange(0, 2000, 500)     # laser_powerの範囲

# # 変数の値の対応関係
# variable_values = {
#     "x1": x1_values,
#     "x2": x2_values,
#     "x3": x3_values,
#     "x4": x4_values
# }

# # 変数名とラベルの対応関係
# variable_labels = {
#     "x1": "welding speed",
#     "x2": "head position",
#     "x3": "material thickness",
#     "x4": "laser power"
# }

# unit_labels = {
#     "x1": "mm/sec",
#     "x2": "mm",
#     "x3": "mm",
#     "x4": "W"
# }

# # h(x1, x2, x3, x4)の定義
# def h(x1, x2, x3, x4):
#     # welding_speedの関数
#     lambda_1 = 0.0045
#     g_x1 =  np.exp(-lambda_1 * abs(x1))

#     # head_positionの関数
#     mu = 0
#     sigma = 2
#     f_x2 = (1 / math.sqrt(2 * math.pi * sigma**2)) * np.exp(-((x2 - mu)**2) / (2 * sigma**2))

#     # material_thicknessの関数
#     lambda_3 = 1.0
#     C = 0.5
#     s_x3 = C + (1 - C) * np.exp(-lambda_3 * x3)

#     # laser_powerの関数
#     alpha = 1.0
#     l_x4 = 1 + alpha * (x4 / 100)

#     return f_x2 * g_x1 * s_x3 * l_x4

# # triggerから変数と固定値を抽出するための初期化
# h_dict = {}
# for key in trigger.keys():
#     if trigger[key] == "x_var":
#         x_axis = (key, variable_values[key])
#         h_dict[key] = variable_values[key]
#     elif trigger[key] == "y_var":
#         y_axis = (key, variable_values[key])
#         h_dict[key] = variable_values[key]
#     else:
#         h_dict[key] = trigger[key]

# # グリッドを作成
# X, Y = np.meshgrid(x_axis[1], y_axis[1])
# x1 = x2 = x3 = x4 = None

# if x_axis[0] == "x1":
#     x1 = X
# elif x_axis[0] == "x2":
#     x2 = X
# elif x_axis[0] == "x3":
#     x3 = X
# else:
#     x4 = X

# if y_axis[0] == "x1":
#     x1 = Y
# elif y_axis[0] == "x2":
#     x2 = Y
# elif y_axis[0] == "x3":
#     x3 = Y
# else:
#     x4 = Y

# if x1 is None:
#     x1 = trigger["x1"]
# elif x2 is None:
#     x2 = trigger["x2"]
# elif x3 is None:
#     x3 = trigger["x3"]
# else:
#     x4 = trigger["x4"]

# if x1 is None:
#     x1 = trigger["x1"]
# elif x2 is None:
#     x2 = trigger["x2"]
# elif x3 is None:
#     x3 = trigger["x3"]
# else:
#     x4 = trigger["x4"]

# h_values = [x1, x2, x3, x4]

# # h(x1, x2, x3, x4)の計算
# Z = h(h_values[0], h_values[1], h_values[2], h_values[3])

# # グラフの作成
# plt.figure()
# plt.pcolormesh(X, Y, Z, cmap='jet')
# plt.colorbar(label='Amplitude')
# plt.grid()

# for key, value in trigger.items():
#     if value == "x_var":
#         keys_with_x_var = key

# for key, value in trigger.items():
#     if value == "y_var":
#         keys_with_y_var = key

# keys_with_not_xy_var = []

# for key, value in trigger.items():
#     if value != "x_var":
#         keys_with_not_xy_var.append(key)

# for key in keys_with_not_xy_var:
#     if trigger[key] == "y_var":
#         keys_with_not_xy_var.remove(key)


# # 軸ラベルとタイトルの設定
# plt.xlabel(f"{variable_labels[keys_with_x_var]} [{unit_labels[keys_with_x_var]}]")
# plt.ylabel(f"{variable_labels[keys_with_y_var]} [{unit_labels[keys_with_y_var]}]")

# plt.title(f"{variable_labels[keys_with_not_xy_var[0]]}: {trigger[keys_with_not_xy_var[0]]} [{unit_labels[keys_with_not_xy_var[0]]}], {variable_labels[keys_with_not_xy_var[1]]}: {trigger[keys_with_not_xy_var[1]]} [{unit_labels[keys_with_not_xy_var[1]]}]")

# # グラフの表示
# plt.show()


import math
import numpy as np
import matplotlib.pyplot as plt

def define_variables():
    return {
        "x1": np.linspace(0, 800, 200),
        "x2": np.linspace(-6, 6, 200),
        "x3": np.linspace(0, 7, 200),
        "x4": np.linspace(0, 2000, 200)
    }

def define_labels():
    variable_labels = {
        "x1": "welding speed",
        "x2": "head position",
        "x3": "material thickness",
        "x4": "laser power"
    }

    unit_labels = {
        "x1": "mm/sec",
        "x2": "mm",
        "x3": "mm",
        "x4": "W"
    }

    return variable_labels, unit_labels

def h(x1, x2, x3, x4):
    # welding_speedの関数
    lambda_1 = 0.0045
    g_x1 =  np.exp(-lambda_1 * abs(x1))

    # head_positionの関数
    mu = 0
    sigma = 2
    f_x2 = (1 / math.sqrt(2 * math.pi * sigma**2)) * np.exp(-((x2 - mu)**2) / (2 * sigma**2))

    # material_thicknessの関数
    lambda_3 = 1.0
    C = 0.5
    s_x3 = C + (1 - C) * np.exp(-lambda_3 * x3)

    # laser_powerの関数
    alpha = 1.0
    l_x4 = 1 + alpha * (x4 / 100)

    return f_x2 * g_x1 * s_x3 * l_x4

def initialize_h_dict(trigger, variable_values):
    h_dict = {}
    for key in trigger.keys():
        if trigger[key] == "x_var":
            x_axis = (key, variable_values[key])
            h_dict[key] = variable_values[key]
        elif trigger[key] == "y_var":
            y_axis = (key, variable_values[key])
            h_dict[key] = variable_values[key]
        else:
            h_dict[key] = trigger[key]
    return h_dict, x_axis, y_axis

def create_grid(x_axis, y_axis):
    X, Y = np.meshgrid(x_axis[1], y_axis[1])
    return X, Y

def assign_variable_values(x_axis, y_axis, h_dict, trigger):
    x1 = x2 = x3 = x4 = None

    if x_axis[0] == "x1":
        x1 = X
    elif x_axis[0] == "x2":
        x2 = X
    elif x_axis[0] == "x3":
        x3 = X
    else:
        x4 = X

    if y_axis[0] == "x1":
        x1 = Y
    elif y_axis[0] == "x2":
        x2 = Y
    elif y_axis[0] == "x3":
        x3 = Y
    else:
        x4 = Y

    if x1 is None:
        x1 = trigger["x1"]
    elif x2 is None:
        x2 = trigger["x2"]
    elif x3 is None:
        x3 = trigger["x3"]
    else:
        x4 = trigger["x4"]

    if x1 is None:
        x1 = trigger["x1"]
    elif x2 is None:
        x2 = trigger["x2"]
    elif x3 is None:
        x3 = trigger["x3"]
    else:
        x4 = trigger["x4"]

    h_values = [x1, x2, x3, x4]
    return h_values

def plot_graph(X, Y, Z, trigger, variable_labels, unit_labels):
    plt.figure()
    plt.pcolormesh(X, Y, Z, cmap='jet')
    plt.colorbar(label='Amplitude')
    plt.grid()

    keys_with_x_var = [key for key, value in trigger.items() if value == "x_var"][0]
    keys_with_y_var = [key for key, value in trigger.items() if value == "y_var"][0]
    keys_with_not_xy_var = [key for key, value in trigger.items() if value not in ["x_var", "y_var"]]

    plt.xlabel(f"{variable_labels[keys_with_x_var]} [{unit_labels[keys_with_x_var]}]")
    plt.ylabel(f"{variable_labels[keys_with_y_var]} [{unit_labels[keys_with_y_var]}]")

    plt.title(f"{variable_labels[keys_with_not_xy_var[0]]}: {trigger[keys_with_not_xy_var[0]]} [{unit_labels[keys_with_not_xy_var[0]]}], {variable_labels[keys_with_not_xy_var[1]]}: {trigger[keys_with_not_xy_var[1]]} [{unit_labels[keys_with_not_xy_var[1]]}]")

    plt.show()

trigger = {
    "x1": "x_var",
    "x2": 0,
    "x3": 1.0,
    "x4": "y_var"
}

variable_values = define_variables()
variable_labels, unit_labels = define_labels()
h_dict, x_axis, y_axis = initialize_h_dict(trigger, variable_values)
X, Y = create_grid(x_axis, y_axis)
h_values = assign_variable_values(x_axis, y_axis, h_dict, trigger)
Z = h(h_values[0], h_values[1], h_values[2], h_values[3])
plot_graph(X, Y, Z, trigger, variable_labels, unit_labels)
