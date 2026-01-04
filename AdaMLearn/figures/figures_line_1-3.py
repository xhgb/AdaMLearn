import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

custom_markers = {
    'AdaMLearn': "o",     # 圆
    'XDER': "s",      # 方块
    'ER': "D",    # 菱形
    'DERpp': "^",       # 上三角
    'DER': "v",     # 下三角
    'AGEM': "P",     # 五边形
    'BIC': "X",      # 叉形
    'LUCIR': "*",    # 星形
    'EWC': "<",      # 左三角
    'LwF': ">",      # 右三角
    'SI': "h"        # 六边形
}

nature_palette_hex = {
    0: "#D62629",  # 红色 (0.84, 0.15, 0.16)
    1: "#FF800D",  # 橙色 (1.00, 0.50, 0.05)
    2: "#2BA12B",  # 绿色 (0.17, 0.63, 0.17)
    3: "#1F78B4",  # 蓝色 (0.12, 0.47, 0.71)
    4: "#9467BD",  # 紫色 (0.58, 0.40, 0.74)
    5: "#8C564B",  # 棕色 (0.55, 0.34, 0.29)
    6: "#E377C2",  # 粉色 (0.89, 0.47, 0.76)
    7: "#7F7F7F",  # 灰色 (0.50, 0.50, 0.50)
    8: "#BDBD23",  # 黄色 (0.74, 0.74, 0.13)
    9: "#17BFCB",  # 青色 (0.09, 0.75, 0.81)
}

nature_palette_hex_1 = {
    0: "#6BAED6",  # 蓝
    1: "#74C476",  # 绿
    2: "#FDAE6B",  # 橙
    3: "#9E9AC8",  # 紫
    4: "#D9B38C"   # 棕
} # 图1的配色方案

nature_palette_hex = {
    0: "#74C476",  # 柔和绿（固定）
    1: "#E7969C",  # 柔和红（温和的粉红调）
    2: "#8DA0CB",  # 蓝灰（冷色调，柔和）
    3: "#CAB2D6",  # 淡紫灰（替代刺眼黄色，低饱和）
    4: "#B3B3B3"   # 中性灰（平衡，常用于 baseline）
} # 图2的配色方案

custom_palette = {
    'AdaMLearn': nature_palette_hex_1[1],
    'XDER': nature_palette_hex_1[3],
    'DER': nature_palette_hex_1[0],
    'DERpp': nature_palette_hex_1[2],
    'ER': nature_palette_hex_1[4]
}


def draw_line():
    # 设置样式
    sns.set_style("white")

    # cifar100平均准确率
    # 构建三元组标签（以千为单位）
    groups_cifar100 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0])
    #groups_cifar100 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000])

    values_cifar100_AAC = (list(np.array([75.15, 75.58, 75.89, 76.20, 76.93, 76.83, 76.60, 76.24, 75.75, 77.08, 76.78])) + # ours
                       list(np.array([49.81, 57.14, 61.88, 65.18, 66.38, 66.79, 68.88, 69.69, 70.77, 72.01, 73.19])) +  # XDER
                       list(np.array([39.5, 48.11, 52.33, 54.28, 55.34, 56.69, 58.37, 59.39, 60.20, 63.85, 64.88])) + # DER
                       list(np.array([39.94, 47.84, 49.95, 49.83, 53.60, 56.52, 56.20, 56.53, 58.76, 60.49, 64.71])) + # DERpp
                       list(np.array([38.67, 45.54, 46.72, 48.96, 50.99, 50.74, 50.77, 53.80, 53.33, 56.31, 58.95])) # ER
                       )

    # 方法名列表
    methods = ['AdaMLearn', 'XDER', 'DER', 'DERpp', 'ER']
    n_points = len(groups_cifar100)

    # 创建 DataFrame
    df_line = pd.DataFrame({
        'Memory Size': groups_cifar100.tolist() * 5,
        'Accuracy (%)': values_cifar100_AAC,
        'Method': sum([[m] * n_points for m in methods], [])  # ['Ours']*11 + ['DER']*11 + ...
    })

    # 设置样式
    sns.set(style="whitegrid")

    # 画图
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_line,
        x='Memory Size',
        y='Accuracy (%)',
        hue='Method',
        style='Method',
        markers=custom_markers,
        dashes=False,  # 是否虚线
        palette=custom_palette,  # 你可以自定义颜色
        linewidth=4,
        markersize=12
    )

    # 微调
    #xticks = list(np.arange(0.1, 2.1, 0.1))
    #xticks = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    xticks = [0.08, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    plt.xticks(xticks, labels=["0.1", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0"])
    plt.ylim(bottom=35)
    #plt.xticks(xticks)
    # plt.title("Accuracy vs Buffer Size", fontsize=14)
    plt.xlabel("Memory Size (K)", fontsize=22)
    plt.ylabel("Average Accuracy (%)", fontsize=22)
    plt.legend(title='Methods', fontsize=16, title_fontsize=22)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.grid(False)

    #plt.axvline(x=0.1, ymin=0, ymax=0.06, color='black', linestyle='--', dashes=(2, 2), alpha=1)
    plt.savefig("./pdfs/figure_1-3.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


def draw_line_memory_usage():
    methods_name = ['AdaMLearn (M1)', 'AdaMLearn – w/o forgetting (M2)', 'AdaMLearn – w/o consolidation (M3)',
                    'AdaMLearn w/o consolidation and forgetting (M4)', 'AdaMLearn with consolidation and random forgetting (M5)']
    custom_markers = {
        methods_name[0]: "o",  # 圆
        methods_name[3]: "s",  # 方块
        methods_name[2]: "D",  # 菱形
        methods_name[1]: "^",  # 上三角
        methods_name[4]: "v",  # 下三角
    }
    memory_palette = {
        methods_name[0]: nature_palette_hex[0],
        methods_name[1]: nature_palette_hex[2],
        methods_name[2]: nature_palette_hex[1],
        methods_name[3]: nature_palette_hex[4],
        methods_name[4]: nature_palette_hex[3]
    }

    # 设置样式
    sns.set_style("white")

    # cifar100记忆占用率
    # 构建三元组标签
    groups_cifar100 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    values_cifar100 = (
            list(np.array([263, 404, 422, 446, 454, 471, 470, 486, 528, 539]) / 4096 * 100) +  # EN
            list(np.array([255, 527, 777, 1031, 1282, 1543, 1798, 2062, 2356, 2660]) / 4096 * 100) +  # without forgetting
            list(np.array([380, 548, 570, 573, 574, 577, 567, 572, 582, 574]) / 4096 * 100) +  # without consolidation
            list(np.array([380, 761, 1141, 1520, 1899, 2280, 2660, 3040, 3420, 3801]) / 4096 * 100) +  # without both
            list(np.array([255, 524, 765, 1016, 1261, 1514, 1759, 2011, 2290, 2577]) / 4096 * 100)  # random forgetting
    )

    # cifar_superclass记忆占用率
    # 构建三元组标签
    groups_cifar_superclass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    values_cifar_superclass = (
            list(np.array([55, 99, 136, 128, 148, 155, 162, 164, 191, 185, 144, 198, 226, 212, 247, 235, 244, 264, 294, 298]) / 1600 * 100) +  # EN
            list(np.array([55, 112, 177, 218, 267, 313, 354, 394, 447, 485, 499, 554, 610, 650, 708, 743, 785, 840, 911, 979]) / 1600 * 100) +  # without forgetting
            list(np.array([191, 297, 330, 321, 327, 328, 324, 323, 331, 325, 304, 328, 333, 321, 334, 325, 324, 329, 338, 329]) / 1600 * 100) +  # without consolidation
            list(np.array([191, 381, 572, 763, 953, 1144, 1334, 1525, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600]) / 1600 * 100) +  # without both
            list(np.array([55, 111, 173, 212, 262, 303, 342, 383, 418, 441, 448, 493, 539, 575, 639, 682, 729, 792, 901, 1007]) / 1600 * 100)  # random forgetting
    )

    # 方法名列表
    methods = methods_name
    n_cifar100 = len(groups_cifar100)
    n_superclass = len(groups_cifar_superclass)
    # 创建 DataFrame
    df_cifar100 = pd.DataFrame({
        'Number of tasks': groups_cifar100.tolist() * 5,
        'Memory usage (%)': values_cifar100,
        'Method': sum([[m] * n_cifar100 for m in methods], [])  # ['Ours']*11 + ['DER']*11 + ...
    })
    df_superclass = pd.DataFrame({
        'Number of tasks': groups_cifar_superclass.tolist() * 5,
        'Memory usage (%)': values_cifar_superclass,
        'Method': sum([[m] * n_superclass for m in methods], [])
    })

    # 设置样式
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharex=False)

    # 图 1：CIFAR100
    plot1 = sns.lineplot(
        data=df_cifar100,
        x='Number of tasks',
        y='Memory usage (%)',
        hue='Method',
        style='Method',
        markers=custom_markers,
        dashes=False,
        palette=memory_palette,
        linewidth=3.5,
        markersize=14,
        ax=axes[0]
    )
    axes[0].set_title("CIFAR-100", fontsize=25)
    axes[0].set_ylabel("Memory usage (%)", fontsize=25)
    axes[0].set_xlabel("Number of tasks", fontsize=25)
    axes[0].tick_params(axis='both', labelsize=20)
    # 从plot对象获取图例
    handles, labels = plot1.get_legend_handles_labels()
    axes[0].legend_.remove()  # 移除当前子图图例

    # 图 2：CIFAR-Superclass
    plot2 = sns.lineplot(
        data=df_superclass,
        x='Number of tasks',
        y='Memory usage (%)',
        hue='Method',
        style='Method',
        markers=custom_markers,
        dashes=False,
        palette=memory_palette,
        linewidth=3.5,
        markersize=14,
        ax=axes[1]
    )
    axes[1].set_title("CIFAR-Superclass", fontsize=25)
    axes[1].set_ylabel("Memory usage (%)", fontsize=25)
    axes[1].set_xlabel("Number of tasks", fontsize=25)
    axes[1].tick_params(axis='both', labelsize=20)
    axes[1].get_legend().remove()

    # 第一行放前 3 个
    handles_top = handles[:3]
    labels_top = labels[:3]

    # 第二行放后 2 个
    handles_bottom = handles[3:]
    labels_bottom = labels[3:]

    # 添加第一行图例 (3列, 居中)
    l1 = fig.legend(
        handles=handles_top,
        labels=labels_top,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),  # 调整垂直位置，比原先稍高一点以容纳两行
        ncol=3,
        fontsize=21,
        columnspacing=1.5,
        handletextpad=0.6,
        frameon=False  # 去掉边框，使两行看起来像一个整体
    )

    # 添加第二行图例 (2列, 居中)
    l2 = fig.legend(
        handles=handles_bottom,
        labels=labels_bottom,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.96),  # 放在第一行下面
        ncol=2,  # 关键点：设置为2列，这样这两个元素就会居中对齐
        fontsize=21,
        columnspacing=1.5,
        handletextpad=0.6,
        frameon=False
    )

    # 去除上和右边框
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # 微调整体布局
    yticks = list(np.arange(0, 101, 10))
    xticks = list(np.arange(1, 10.01, 1))
    axes[0].set_yticks(yticks)
    axes[0].set_xticks(xticks)
    axes[0].grid(False)
    yticks = list(np.arange(0, 101, 10))
    xticks = list(np.arange(1, 20.01, 1))
    axes[1].set_yticks(yticks)
    axes[1].set_xticks(xticks)
    axes[1].grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig("./pdfs/figure_memory_usage.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    draw_line_memory_usage()
