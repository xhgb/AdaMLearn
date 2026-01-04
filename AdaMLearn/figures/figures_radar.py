import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

nature_palette = {
    0: (0.84, 0.15, 0.16),  # 红色
    1: (1.00, 0.50, 0.05),  # 橙色
    2: (0.17, 0.63, 0.17),  # 绿色
    3: (0.12, 0.47, 0.71),  # 蓝色
    4: (0.58, 0.40, 0.74),  # 紫色
    5: (0.55, 0.34, 0.29),  # 棕色
    6: (0.89, 0.47, 0.76),  # 粉色
    7: (0.50, 0.50, 0.50),  # 灰色
    8: (0.74, 0.74, 0.13),  # 黄色
    9: (0.09, 0.75, 0.81),  # 青色
}

nature_palette_hex_no = {
    0: "#74C476",  # 柔和绿
    1: "#8DA0CB",  # 蓝灰
    2: "#E7969C",  # 柔和红
    3: "#B3B3B3",  # 中性灰
    4: "#CAB2D6",  # 淡紫灰
    5: "#A6CEE3",  # 淡蓝
    6: "#FDBF6F",  # 柔和橙（压低饱和度的版本）
    7: "#CCEBC5",  # 薄荷绿（更淡的绿，补充 #74C476）
    8: "#FCBBA1",  # 淡珊瑚红（和 #E7969C 区分）
    9: "#D9B38C",  # 米棕色（自然、安静）
    10: "#999999"  # 深灰（对比度略强，用于辅助线/额外方法）
}

nature_palette_hex = {
    0: "#6BAED6",  # 蓝
    1: "#74C476",  # 绿
    2: "#FDAE6B",  # 橙
    3: "#9E9AC8",  # 紫
    4: "#D9B38C",  # 棕
    5: "#E7969C",  # 柔和红
    6: "#8DA0CB",  # 蓝灰
    7: "#A6D854",  # 柔和青绿
    8: "#66C2A5",  # 青色（替代桃橙，更冷静）
    9: "#CAB2D6",  # 淡紫灰
    10: "#B3B3B3"  # 中性灰
}

custom_palette = {
    'AdaMLearn': nature_palette_hex[1],
    "LUCIR": nature_palette_hex[5],
    "BIC": nature_palette_hex[6],
    'XDER': nature_palette_hex[3],
    "AGEM": nature_palette_hex[7],
    'DER': nature_palette_hex[0],
    'DERpp': nature_palette_hex[2],
    'ER': nature_palette_hex[4],
    "LwF": nature_palette_hex[9],
    "SI": nature_palette_hex[8],
    "EWC": nature_palette_hex[10],
}

def draw_radar():
    labels5 = [r'$\mathcal{T}_1$', r'$\mathcal{T}_2$', r'$\mathcal{T}_3$', r'$\mathcal{T}_4$', r'$\mathcal{T}_5$']
    labels10 = [r'$\mathcal{T}_1$', r'$\mathcal{T}_2$', r'$\mathcal{T}_3$', r'$\mathcal{T}_4$', r'$\mathcal{T}_5$',
              r'$\mathcal{T}_6$', r'$\mathcal{T}_7$', r'$\mathcal{T}_8$', r'$\mathcal{T}_9$', r'$\mathcal{T}_{10}$']
    labels20 = [r'$\mathcal{T}_1$', r'$\mathcal{T}_2$', r'$\mathcal{T}_3$', r'$\mathcal{T}_4$', r'$\mathcal{T}_5$',
               r'$\mathcal{T}_6$', r'$\mathcal{T}_7$', r'$\mathcal{T}_8$', r'$\mathcal{T}_9$', r'$\mathcal{T}_{10}$',
               r'$\mathcal{T}_{11}$', r'$\mathcal{T}_{12}$', r'$\mathcal{T}_{13}$', r'$\mathcal{T}_{14}$', r'$\mathcal{T}_{15}$',
               r'$\mathcal{T}_{16}$', r'$\mathcal{T}_{17}$', r'$\mathcal{T}_{18}$', r'$\mathcal{T}_{19}$', r'$\mathcal{T}_{20}$']

    angles_5 = np.linspace(0, 2 * np.pi, len(labels5), endpoint=False).tolist()
    angles_10 = np.linspace(0, 2 * np.pi, len(labels10), endpoint=False).tolist()
    angles_20 = np.linspace(0, 2 * np.pi, len(labels20), endpoint=False).tolist()

    data_dict_cifar100 = {
        "EN(ours)": [78.0, 68.5, 78.2, 76.1, 79.5, 77.7, 77.8, 75.7, 77.8, 84.1],
        "GPM": [78.9, 69.9, 74.6, 72.1, 74.5, 73.4, 72.3, 73.0, 72.0, 78.2],
        "AGEM": [73.0, 66.6, 75.1, 68.4, 72.2, 74.4, 74.5, 71.7, 73.6, 78.6],
        "DER": [72.9, 64.6, 74.1, 68.0, 72.7, 72.4, 73.8, 69.6, 71.6, 76.1],
        "ER": [70.5, 67.3, 75.4, 66.3, 73.0, 73.2, 73.1, 72.2, 72.1, 80.0],
        "SI": [68.5, 63.8, 75.7, 68.4, 72.3, 72.3, 76.3, 70.4, 72.3, 78.0],
        "LwF": [68.7, 64.6, 71.9, 66.8, 74.6, 71.5, 74.5, 73.0, 73.2, 78.9],
        "LUCIR": [75.9, 70.7, 75.3, 69.2, 71.9, 72.4, 73.1, 73.9, 74.0, 80.3],
        "XDER": [70.1, 61.6, 73.3, 70.1, 69.5, 72.4, 76.4, 72.3, 71.0, 78.0]
    }

    data_dict_cifar_superclass = {
        "EN(ours)": [55.2, 68.2, 64.4, 70.2, 67.0, 65.0, 67.0, 66.4, 62.2, 70.0, 81.0, 64.0, 66.2, 62.2, 37.4, 57.6, 55.0, 53.0, 64.8, 74.0],
        "GPM": [52.6, 67.0, 61.6, 62.8, 64.0, 59.6, 64.0, 61.6, 52.8, 63.0, 76.4, 53.0, 60.6, 58.0, 32.8, 54.4, 52.0, 52.2, 59.8, 65.6],
        "AGEM": [52.4, 62.2, 54.0, 60.8, 63.4, 64.8, 59.6, 59.8, 59.0, 70.4, 74.4, 53.8, 59.4, 59.0, 32.8, 52.0, 42.0, 52.0, 59.4, 73.4],
        "DER": [52.2, 66.0, 58.6, 65.0, 70.6, 66.2, 59.8, 61.4, 58.6, 72.4, 80.0, 55.8, 59.6, 58.2, 33.0, 52.6, 48.0, 56.6, 68.4, 75.4],
        "ER": [47.0, 61.6, 58.0, 62.4, 70.4, 63.2, 57.8, 59.8, 60.2, 70.6, 77.4, 60.4, 65.2, 56.4, 35.2, 53.8, 46.8, 56.6, 64.0, 77.8],
        "SI": [51.8, 68.0, 59.2, 68.4, 74.0, 66.8, 60.6, 61.2, 60.8, 72.6, 79.8, 58.4, 67.0, 60.4, 34.4, 56.8, 51.4, 60.2, 67.6, 74.0],
        "LwF": [51.2, 65.6, 60.6, 70.4, 75.6, 69.0, 65.4, 61.4, 61.2, 72.0, 80.6, 59.2, 65.2, 62.8, 37.4, 54.2, 51.4, 61.2, 72.0, 77.0], # seed4
        "LUCIR": [46.8, 60.6, 62.0, 64.4, 67.8, 63.2, 58.2, 61.2, 56.0, 69.2, 77.2, 59.2, 64.4, 59.4, 39.0, 56.4, 52.4, 58.6, 68.2, 75.8],
        "XDER": [53.4, 64.8, 57.4, 72.2, 68.8, 70.2, 65.0, 61.4, 56.2, 74.4, 78.6, 62.0, 63.4, 64.4, 31.4, 58.4, 52.6, 58.0, 69.6, 80.6]
    }

    data_dict_5_datasets = {
        "EN(ours)": [81.4, 99.1, 90.0, 99.2, 94.3], # seed864
        "GPM": [76.2, 99.1, 87.4, 99.2, 93.9],
        "AGEM": [1, 1, 1, 1, 1],
        "DER": [1, 1, 1, 1, 1],
        "ER": [1, 1, 1, 1, 1],
        "SI": [1, 1, 1, 1, 1],
        "LwF": [1, 1, 1, 1, 1],
        "LUCIR": [1, 1, 1, 1, 1],
        "XDER": [1, 1, 1, 1, 1]
    }

    # 让数据闭合（首尾相连）
    for key in data_dict_cifar100:
        data_dict_cifar100[key] += [data_dict_cifar100[key][0]]  # 闭合数据
    angles_10 += [angles_10[0]]  # 闭合角度

    for key in data_dict_cifar_superclass:
        data_dict_cifar_superclass[key] += [data_dict_cifar_superclass[key][0]]  # 闭合数据
    angles_20 += [angles_20[0]]  # 闭合角度

    # 创建雷达图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})

    # 绘制每一组数据
    for i, (key, data) in enumerate(data_dict_cifar100.items()):
        ax[0].plot(angles_10, data, linewidth=2, alpha=0.6, linestyle='solid', label=key, color=nature_palette_hex[i])
        ax[0].fill(angles_10, data, alpha=0.05, color=nature_palette_hex[i])

    for i, (key, data) in enumerate(data_dict_cifar_superclass.items()):
        ax[1].plot(angles_20, data, linewidth=2, alpha=0.6, linestyle='solid', label=key, color=nature_palette_hex[i])
        ax[1].fill(angles_20, data, alpha=0.05, color=nature_palette_hex[i])


    # 设置标签
    yticks = [60, 70, 80]
    ax[0].set_xticks(angles_10[:-1])
    ax[0].set_yticks(yticks)
    ax[0].set_xticklabels(labels10, fontsize=32)
    ax[0].set_yticklabels([str(y) for y in yticks], fontsize=20)
    ax[0].set_ylim(60, 85)
    ax[0].grid(True)

    yticks2 = [40, 50, 60, 70, 80]
    ax[1].set_xticks(angles_20[:-1])
    ax[1].set_yticks(yticks2)
    ax[1].set_xticklabels(labels20, fontsize=32)
    ax[1].set_yticklabels([str(y) for y in yticks2], fontsize=20)
    ax[1].set_ylim(30, 85)
    ax[1].grid(True)

    # 添加图例
    ax[0].set_title('plasticity of CL methods in cifar-100 dataset')
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))
    ax[1].set_title('plasticity of CL methods in cifar-superclass dataset')
    ax[1].legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))

    plt.tight_layout()
    plt.savefig("figure_radar.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def draw_radar_single():
    labels10 = [r'$\mathcal{T}_1$', r'$\mathcal{T}_2$', r'$\mathcal{T}_3$', r'$\mathcal{T}_4$', r'$\mathcal{T}_5$',
                r'$\mathcal{T}_6$', r'$\mathcal{T}_7$', r'$\mathcal{T}_8$', r'$\mathcal{T}_9$', r'$\mathcal{T}_{10}$']

    angles_10 = np.linspace(0, 2 * np.pi, len(labels10), endpoint=False).tolist()

    data_dict_cifar100 = {
        "AdaMLearn": [78.0, 68.5, 78.2, 76.1, 79.5, 77.7, 77.8, 75.7, 77.8, 84.1],
        "LUCIR": [75.9, 70.7, 75.3, 69.2, 71.9, 72.4, 73.1, 73.9, 74.0, 80.3],
        "BIC": [71.0, 62.3, 72.1, 68.9, 71.5, 73.7, 72.6, 71.5, 72.7, 78.5],
        "XDER": [70.1, 61.6, 73.3, 70.1, 69.5, 72.4, 76.4, 72.3, 71.0, 78.0],
        "AGEM": [73.0, 66.6, 75.1, 68.4, 72.2, 74.4, 74.5, 71.7, 73.6, 78.6],
        "DER": [72.9, 64.6, 74.1, 68.0, 72.7, 72.4, 73.8, 69.6, 71.6, 76.1],
        "DERpp": [69.1, 63.3, 73.4, 67.1, 70.6, 72.5, 73.9, 69.9, 71.9, 77.4],
        "ER": [70.5, 67.3, 75.4, 66.3, 73.0, 73.2, 73.1, 72.2, 72.1, 80.0],
        "LwF": [68.7, 64.6, 71.9, 66.8, 74.6, 71.5, 74.5, 73.0, 73.2, 78.9],
        "SI": [68.5, 63.8, 75.7, 68.4, 72.3, 72.3, 76.3, 70.4, 72.3, 78.0]
    }

    # 让数据闭合
    for key in data_dict_cifar100:
        data_dict_cifar100[key] += [data_dict_cifar100[key][0]]
    angles_10 += [angles_10[0]]

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': 'polar'})

    for key, data in data_dict_cifar100.items():
        ax.plot(angles_10, data, linewidth=3, alpha=0.6, linestyle='solid', label=key, color=custom_palette[key])
        ax.fill(angles_10, data, alpha=0.05, color=custom_palette[key])

    # ---- y轴标签修改为 0.60 / 0.70 / 0.80 ----
    yticks = [60, 70, 80]
    ax.set_yticks(yticks)
    ax.set_yticklabels([])  # 不显示默认文本
    ax.set_ylim(60, 85)

    # ---- 手动画新的 ytick 标签 ----
    # 0.60：往下
    # ---- 手动画新的 ytick 标签（百分比显示） ----
    # 60%：往下
    ax.text(np.deg2rad(-90), 60, "60%",
            ha='center', va='center', fontsize=16)

    # 70%：右上
    ax.text(np.deg2rad(35), 70, "70%",
            ha='center', va='center', fontsize=16)

    # 80%：右上
    ax.text(np.deg2rad(35), 80, "80%",
            ha='center', va='center', fontsize=16)

    # x轴标签
    ax.set_xticks(angles_10[:-1])
    ax.set_xticklabels(labels10, fontsize=24)

    ax.grid(True)

    ax.legend(title='Methods', fontsize=14, title_fontsize=18,
              loc='center right', bbox_to_anchor=(1.44, 0.5))

    plt.tight_layout()
    plt.savefig("./pdfs/figure_radar_single.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()



if __name__ == '__main__':
    draw_radar_single()
