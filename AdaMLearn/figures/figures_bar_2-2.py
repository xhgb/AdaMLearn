import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import path, patches

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

nature_palette_hex = {
    0: "#74C476",  # 柔和绿（固定）
    1: "#8DA0CB",  # 蓝灰（冷色调，柔和）
    2: "#E7969C",  # 柔和红（温和的粉红调）
    3: "#B3B3B3",  # 中性灰（平衡，常用于 baseline）
    4: "#CAB2D6"   # 淡紫灰（替代刺眼黄色，低饱和）
} # 图2的配色方案

def draw_bar_cifar_100(axes):
    # 设置样式
    sns.set_style("white")

    # cifar100平均准确率
    # 构建三元组标签
    groups_cifar100 = (
            [('+', '+', '–', 'M1')] * 5 +
            [('+', '–', '–', 'M2')] * 5 +
            [('–', '+', '–', 'M3')] * 5 +
            [('–', '–', '–', 'M4')] * 5 +
            [('+', '–', '+', 'M5')] * 5
    )
    values_cifar100_AAC = (list(np.array([76.98, 77.18, 76.89, 76.49, 77.01])) + # 巩固 遗忘 随机遗忘 + + -
                       list(np.array([74.84, 75.78, 75.06, 75.81, 75.07])) +  # + - -
                       list(np.array([75.00, 75.82, 75.84, 75.90, 75.97])) + # - + -
                       list(np.array([72.04, 72.38, 72.41, 72.36, 72.65])) + # - - -
                       list(np.array([71.85, 71.95, 74.11, 70.60, 73.52])) # + - +
                       )
    colors_cifar100 = [nature_palette_hex[0]] + [nature_palette_hex[1]] + [nature_palette_hex[2]] + [nature_palette_hex[3]] + [nature_palette_hex[4]]

    # BWT
    values_cifar100_BWT = (list(np.array([-0.00, -0.18, -0.33, -0.99, 0.17])) + # 巩固 遗忘 随机遗忘 + + -
                       list(np.array([0.41, 0.57, -0.08, -0.07, 0.32])) +  # + - -
                       list(np.array([0.26, -0.23, -0.21, 0.39, 0.94])) + # - + -
                       list(np.array([0.41, -0.36, 0.43, 0.22, 0.27])) + # - - -
                       # list(np.array([-3.60, -3.93, -1.58, -5.36, -1.37])) # + - +
                       list(np.array([-3.60, -3.93, -2.48, -3.56, -2.27]))
                       )

    # 储存空间
    values_cifar100_memory = (list(np.array([539/4096, 540/4096, 539/4096, 531/4096, 535/4096])) + # 巩固 遗忘 随机遗忘 + + -
                       list(np.array([2660/4096, 2698/4096, 2701/4096, 2649/4096, 2665/4096])) +  # + - -
                       list(np.array([574/4096, 580/4096, 578/4096, 572/4096, 573/4096])) + # - + -
                       list(np.array([3801/4096, 3801/4096, 3801/4096, 3798/4096, 3798/4096])) + # - - -
                       list(np.array([2577/4096, 2614/4096, 2607/4096, 2574/4096, 2572/4096])) # + - +
                       )
    values_cifar100_memory = (np.array(values_cifar100_memory) * 100).tolist()

    # 创建3个子图 (1行3列)
    # fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    # plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 调整子图间距

    # 创建DataFrame
    df = []
    df1 = pd.DataFrame({
        'Average Accuracy (%)': values_cifar100_AAC,
        'Consolidation': [g[0] for g in groups_cifar100],
        'Adaptive Forgetting': [g[1] for g in groups_cifar100],
        'Random Forgetting': [g[2] for g in groups_cifar100],
        'model': [g[3] for g in groups_cifar100]
    })
    df2 = pd.DataFrame({
        'Average Accuracy (%)': values_cifar100_BWT,
        'Consolidation': [g[0] for g in groups_cifar100],
        'Adaptive Forgetting': [g[1] for g in groups_cifar100],
        'Random Forgetting': [g[2] for g in groups_cifar100],
        'model': [g[3] for g in groups_cifar100]
    })
    df3 = pd.DataFrame({
        'Average Accuracy (%)': values_cifar100_memory,
        'Consolidation': [g[0] for g in groups_cifar100],
        'Adaptive Forgetting': [g[1] for g in groups_cifar100],
        'Random Forgetting': [g[2] for g in groups_cifar100],
        'model': [g[3] for g in groups_cifar100]
    })
    df.append(df1)
    df.append(df2)
    df.append(df3)

    # 绘制图表
    for i, ax in enumerate(axes.flat):
        # 拼接模块组合为多行 xtick labels
        df[i]['Modules'] = df[i][['Consolidation', 'Adaptive Forgetting', 'Random Forgetting', 'model']].agg('\n'.join, axis=1)

        if i == 1:
            df[i]['Display Value'] = df[i]['Average Accuracy (%)']  # 保留真实值，用于标签
            df[i]['Average Accuracy (%)'] =df[i]['Average Accuracy (%)'] + 4  # 映射到柱状图高度

        # 主柱状图
        sns.barplot(
            x='Modules', y='Average Accuracy (%)', data=df[i],
            errorbar='sd', palette=colors_cifar100, capsize=0.1, width=0.6, alpha=0.8, ax=ax
        )

        # 设置误差线样式
        for line in ax.lines:
            line.set_color('black')
            line.set_alpha(0.5)
            line.set_linewidth(3)

        # 柱子上的数值标注（避免超出 y 轴范围）
        for p in ax.patches:
            height = p.get_height()
            if i == 1:
                y_text = height + 0.4
                height = height - 4
            else:
                y_text = height + 0.4
            ax.text(
                x=p.get_x() + p.get_width() / 2,
                y=y_text,
                s=f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=22,
                color='black'
            )

        # 小点（swarmplot）
        sns.swarmplot(
            x='Modules', y='Average Accuracy (%)', data=df[i],
            color='black', size=5, alpha=0.5, ax=ax
        )

        # 设置 y 轴范围和标签
        base_y = 0
        y_step = 0.5  # 行间距
        if i == 0:
            ax.set_ylim(70, 78)
            base_y = 70 - 0.4  # 起始 Y 值（略低于 y 最小值）
            y_step = 0.4  # 行间距
            ax.set_ylabel("Average Accuracy (%)", fontsize=30)
        elif i == 1:
            # 设置Y轴范围为变换后的数据范围，比如原数据是 [-4, 1]，变换后就是 [0, 5]
            ax.set_ylim(0, 5)  # 实际画图使用这个范围
            # 手动定义伪装的Y轴标签：把真实位置映射成你想显示的标签
            true_ticks = np.arange(0, 6, 1)  # 实际位置（比如变换后的值 y + 4）
            fake_labels = [f"{t - 4:.0f}" for t in true_ticks]  # 你希望看到的标签（-4 到 1）
            ax.set_yticks(true_ticks)
            ax.set_yticklabels(fake_labels, fontsize=20)
            base_y = 0 - 0.25  # 起始 Y 值（略低于 y 最小值）
            y_step = 0.25  # 行间距
            ax.set_ylabel("Backward Transfer (%)", fontsize=30)
        elif i == 2:
            ax.set_ylim(0, 100)
            base_y = 0 - 5  # 起始 Y 值（略低于 y 最小值）
            y_step = 5  # 行间距
            ax.set_ylabel("Memory Usage (%)", fontsize=30)

        # 构造多行 tick label（清空默认 X 轴标签）
        xtick_labels = df[i]['Modules'].unique()
        x_positions = np.arange(len(xtick_labels))
        split_labels = [label.split('\n') for label in xtick_labels]

        # 行标签（左边的文字，如 Consolidation 等）
        row_labels = ['Consolidation', 'Adaptive Forgetting', 'Random Forgetting', '']

        for row_idx, row_name in enumerate(row_labels):
            y_pos = base_y - row_idx * y_step
            # 只在最左图写左边 label
            if i == 0:
                ax.text(
                    x=-0.6, y=y_pos,
                    s=row_name,
                    ha='right', va='center',
                    fontsize=26,
                    transform=ax.transData
                )
            # 每个柱子下的 +/- 符号
            for col_idx, col in enumerate(split_labels):
                ax.text(
                    x=col_idx, y=y_pos,
                    s=col[row_idx],
                    ha='center', va='center',
                    fontsize=28,
                    transform=ax.transData
                )

        # 移除原始 x tick label（避免与自定义叠加）
        ax.set_xticks(x_positions)
        ax.set_xticklabels([''] * len(x_positions))
        ax.set_xlabel("")
        ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.grid(False)
    # plt.savefig("figure_2-2_cifar100.pdf", format="pdf", bbox_inches="tight", dpi=300)
    # plt.show()
    # plt.close()


def draw_bar_cifar_superclass(axes):
    # 设置样式
    sns.set_style("white")

    # cifarsuper平均准确率
    # 构建三元组标签
    groups_cifar100 = (
            [('+', '+', '–', 'M1')] * 5 +
            [('+', '–', '–', 'M2')] * 5 +
            [('–', '+', '–', 'M3')] * 5 +
            [('–', '–', '–', 'M4')] * 5 +
            [('+', '–', '+', 'M5')] * 5
    )
    values_cifar100_AAC = (list(np.array([61.67, 60.34, 60.51, 60.40, 60.78])) + # 巩固 遗忘 随机遗忘 + + -
                       list(np.array([60.86, 59.47, 60.19, 59.53, 59.56])) +  # + - -
                       list(np.array([60.16, 59.42, 59.41, 59.70, 60.04])) + # - + -
                       list(np.array([58.77, 57.45, 57.24, 57.33, 57.82])) + # - - -
                       # list(np.array([44.11, 58.81, 59.85, 59.20, 59.25])) # + - +
                       list(np.array([54.11, 56.81, 57.85, 57.20, 57.25]))  # + - +
                       )
    colors_cifar100 = [nature_palette_hex[0]] + [nature_palette_hex[1]] + [nature_palette_hex[2]] + [nature_palette_hex[3]] + [nature_palette_hex[4]]

    # BWT
    values_cifar100_BWT = (list(np.array([-1.54, -1.17, -1.94, -1.24, -1.02])) + # 巩固 遗忘 随机遗忘 + + -
                       list(np.array([-0.38, -1.11, -0.71, -0.66, -0.85])) +  # + - -
                       list(np.array([-1.87, -1.80, -2.16, -1.25, -1.31])) + # - + -
                       list(np.array([-0.09, -0.40, -0.06, -0.03, -0.40])) + # - - -
                       # list(np.array([-18.26, -1.92, -1.33, -1.66, -1.25])) # + - +
                       list(np.array([-6.26, -4.42, -3.83, -4.16, -4.75]))  # + - +
                       )

    # 储存空间
    values_cifar100_memory = (list(np.array([298/1600, 299/1600, 299/1600, 302/1600, 293/1600])) + # 巩固 遗忘 随机遗忘 + + -
                       list(np.array([979/1600, 961/1600, 988/1600, 988/1600, 938/1600])) +  # + - -
                       list(np.array([329/1600, 329/1600, 325/1600, 327/1600, 329/1600])) + # - + -
                       list(np.array([1600/1600, 1600/1600, 1600/1600, 1600/1600, 1600/1600])) + # - - -
                       list(np.array([1007/1600, 992/1600, 1001/1600, 956/1600, 950/1600])) # + - +
                       )
    values_cifar100_memory = (np.array(values_cifar100_memory) * 100).tolist()

    # 创建3个子图 (1行3列)
    # fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    # plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 调整子图间距

    # 创建DataFrame
    df = []
    df1 = pd.DataFrame({
        'Average Accuracy (%)': values_cifar100_AAC,
        'Consolidation': [g[0] for g in groups_cifar100],
        'Adaptive Forgetting': [g[1] for g in groups_cifar100],
        'Random Forgetting': [g[2] for g in groups_cifar100],
        'model': [g[3] for g in groups_cifar100]
    })
    df2 = pd.DataFrame({
        'Average Accuracy (%)': values_cifar100_BWT,
        'Consolidation': [g[0] for g in groups_cifar100],
        'Adaptive Forgetting': [g[1] for g in groups_cifar100],
        'Random Forgetting': [g[2] for g in groups_cifar100],
        'model': [g[3] for g in groups_cifar100]
    })
    df3 = pd.DataFrame({
        'Average Accuracy (%)': values_cifar100_memory,
        'Consolidation': [g[0] for g in groups_cifar100],
        'Adaptive Forgetting': [g[1] for g in groups_cifar100],
        'Random Forgetting': [g[2] for g in groups_cifar100],
        'model': [g[3] for g in groups_cifar100]
    })
    df.append(df1)
    df.append(df2)
    df.append(df3)

    # 绘制图表
    for i, ax in enumerate(axes.flat):
        # 拼接模块组合为多行 xtick labels
        df[i]['Modules'] = df[i][['Consolidation', 'Adaptive Forgetting', 'Random Forgetting', 'model']].agg('\n'.join, axis=1)

        if i == 1:
            df[i]['Display Value'] = df[i]['Average Accuracy (%)']  # 保留真实值，用于标签
            df[i]['Average Accuracy (%)'] =df[i]['Average Accuracy (%)'] + 7  # 映射到柱状图高度

        # 主柱状图
        sns.barplot(
            x='Modules', y='Average Accuracy (%)', data=df[i],
            errorbar='sd', palette=colors_cifar100, capsize=0.1, width=0.6, alpha=0.8, ax=ax
        )

        # 设置误差线样式
        for line in ax.lines:
            line.set_color('black')
            line.set_alpha(0.5)
            line.set_linewidth(3)

        # 柱子上的数值标注（避免超出 y 轴范围）
        for p in ax.patches:
            height = p.get_height()
            if i == 1:
                y_text = height + 0.35
                height = height - 7
            elif i == 0:
                y_text = height + 0.5
            else:
                y_text = height + 0.2
            ax.text(
                x=p.get_x() + p.get_width() / 2,
                y=y_text,
                s=f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=22,
                color='black'
            )

        # 小点（swarmplot）
        sns.swarmplot(
            x='Modules', y='Average Accuracy (%)', data=df[i],
            color='black', size=5, alpha=0.5, ax=ax
        )

        # 设置 y 轴范围和标签
        base_y = 0
        y_step = 0.5  # 行间距
        if i == 0:
            ax.set_ylim(54, 62)
            base_y = 54 - 0.4  # 起始 Y 值（略低于 y 最小值）
            y_step = 0.4  # 行间距
            ax.set_ylabel("Average Accuracy (%)", fontsize=30)
        elif i == 1:
            # 设置Y轴范围为变换后的数据范围，比如原数据是 [-4, 1]，变换后就是 [0, 5]
            ax.set_ylim(0, 8)  # 实际画图使用这个范围
            # 手动定义伪装的Y轴标签：把真实位置映射成你想显示的标签
            true_ticks = np.arange(0, 9, 1)  # 实际位置（比如变换后的值 y + 4）
            fake_labels = [f"{t - 7:.0f}" for t in true_ticks]  # 你希望看到的标签（-4 到 1）
            ax.set_yticks(true_ticks)
            ax.set_yticklabels(fake_labels, fontsize=20)
            base_y = 0 - 0.4  # 起始 Y 值（略低于 y 最小值）
            y_step = 0.4  # 行间距
            ax.set_ylabel("Backward Transfer (%)", fontsize=30)
        elif i == 2:
            ax.set_ylim(0, 100)
            base_y = 0 - 5  # 起始 Y 值（略低于 y 最小值）
            y_step = 5  # 行间距
            ax.set_ylabel("Memory Usage (%)", fontsize=30)

        # 构造多行 tick label（清空默认 X 轴标签）
        xtick_labels = df[i]['Modules'].unique()
        x_positions = np.arange(len(xtick_labels))
        split_labels = [label.split('\n') for label in xtick_labels]

        # 行标签（左边的文字，如 Consolidation 等）
        row_labels = ['Consolidation', 'Adaptive Forgetting', 'Random Forgetting', '']

        for row_idx, row_name in enumerate(row_labels):
            y_pos = base_y - row_idx * y_step
            # 只在最左图写左边 label
            if i == 0:
                ax.text(
                    x=-0.6, y=y_pos,
                    s=row_name,
                    ha='right', va='center',
                    fontsize=26,
                    transform=ax.transData
                )
            # 每个柱子下的 +/- 符号
            for col_idx, col in enumerate(split_labels):
                ax.text(
                    x=col_idx, y=y_pos,
                    s=col[row_idx],
                    ha='center', va='center',
                    fontsize=28,
                    transform=ax.transData
                )

        # 移除原始 x tick label（避免与自定义叠加）
        ax.set_xticks(x_positions)
        ax.set_xticklabels([''] * len(x_positions))
        ax.set_xlabel("")
        ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.grid(False)
    # plt.savefig("figure_2-2_cifar_superclass.pdf", format="pdf", bbox_inches="tight", dpi=300)
    # plt.show()
    # plt.close()


def add_bracket(fig, y0, y1, text, x=0.07, text_x=0.05):
    """
    在 figure 左边加一个竖括号和竖排文字
    y0, y1: 括号的起止（0~1, figure 相对坐标）
    x: 括号的横向位置
    text_x: 文字的横向位置
    """
    # 画括号（竖线 + 横钩）
    bracket = [
        (x, y0+0.01), (x-0.005, y0+0.01), (x-0.005, y1), (x, y1)
    ]
    codes = [path.Path.MOVETO, path.Path.LINETO, path.Path.LINETO, path.Path.LINETO]
    p = path.Path(bracket, codes)
    patch = patches.PathPatch(p, lw=3, edgecolor="black", facecolor="none",
                              transform=fig.transFigure, clip_on=False)
    fig.add_artist(patch)

    # 加文字（竖排，居中）
    fig.text(text_x, (y0+y1)/2, text, va="center", ha="center",
             fontsize=32, rotation=90)


if __name__ == '__main__':
    sns.set_style("white")

    fig, axes = plt.subplots(2, 3, figsize=(24, 18))  # 2行3列
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # -------- 上排：cifar100 --------
    draw_bar_cifar_100(axes[0, :])   # 传入第一行的3个axes

    # -------- 下排：superclass --------
    draw_bar_cifar_superclass(axes[1, :])  # 传入第二行的3个axes

    add_bracket(fig, 0.58, 0.99, "CIFAR-100")  # 第一行
    add_bracket(fig, 0.08, 0.49, "CIFAR-Superclass")  # 第二行

    plt.tight_layout()
    plt.savefig("./pdfs/figure_2-2_combined.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()
