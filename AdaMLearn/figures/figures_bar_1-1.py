import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

nature_palette_hex = {
    0: "#74C476", # "#D62629",  # 红色 (0.84, 0.15, 0.16)
    1: "#FF800D",  # 橙色 (1.00, 0.50, 0.05)
    2: "#6BAED6", # "#2BA12B",  # 绿色 (0.17, 0.63, 0.17)
    3: "#1F78B4",  # 蓝色 (0.12, 0.47, 0.71)
    4: "#9467BD",  # 紫色 (0.58, 0.40, 0.74)
    5: "#9E9AC8", # "#8C564B",  # 棕色 (0.55, 0.34, 0.29)
    6: "#E377C2",  # 粉色 (0.89, 0.47, 0.76)
    7: "#7F7F7F",  # 灰色 (0.50, 0.50, 0.50)
    8: "#BDBD23",  # 黄色 (0.74, 0.74, 0.13)
    9: "#D9B38C", # "#17BFCB",  # 青色 (0.09, 0.75, 0.81)
}

def draw_bar():
    # 设置样式
    sns.set_style("white")
    plt.rcParams['font.family'] = 'sans-serif'  # 先指定通用字体族
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # 备选字体列表

    # cifar100
    groups_cifar100 = (['AdaMLearn'] * 5 + ['LUCIR'] * 5 + ['BIC'] * 5 + ['XDER'] * 5 +
                       ['AGEM'] * 5 + ['DER'] * 5 + ['DERpp'] * 5 + ['ER'] * 5 + ['EWC'] * 5 + ['LwF'] * 5 + ['SI'] * 5)
    values_cifar100 = (list(np.array([76.98, 77.18, 76.89, 76.49, 77.01])) + # EIM
                       list(np.array([69.51, 69.53, 67.58, 68.62, 68.68])) +  # LUCIR
                       list(np.array([74.38, 73.5, 74.26, 73.74, 73.89])) + # BIC
                       list(np.array([57.15, 56.28, 57.59, 66.36, 66.63])) + # XDER
                       list(np.array([40.55, 36.49, 39.69, 40.67, 40.3])) + # AGEM
                       list(np.array([55.94, 53.89, 55.56, 55.96, 53.95])) + # DER
                       list(np.array([53.6, 51.64, 55.33, 53.24, 52.24])) + # DERpp
                       list(np.array([50.99, 50.23, 50.91, 49.33, 49.14])) + # ER
                       list(np.array([27.7, 24.99, 26.45, 24.15, 25.74])) + # EWC
                       list(np.array([56.22, 52.79, 54.27, 55.23, 53.77])) + # LwF
                       list(np.array([66.2, 66.92, 66.12, 65.04, 64.97]))  # SI
                       )
    colors_cifar100 = [nature_palette_hex[0]] * 1 + [nature_palette_hex[2]] * 3 + [nature_palette_hex[9]] * 4 + [nature_palette_hex[5]] * 3

    # cifar_superclass
    groups_superclass = (['AdaMLearn'] * 5 + ['LUCIR'] * 5 + ['BIC'] * 5 + ['XDER'] * 5 +
                         ['AGEM'] * 5 + ['DER'] * 5 + ['DERpp'] * 5 + ['ER'] * 5 + ['EWC'] * 5 + ['LwF'] * 5 + ['SI'] * 5)
    values_superclass = (list(np.array([61.67, 60.34, 60.51, 60.4, 60.78])) +  # EIM
                         list(np.array([54.28, 54.48, 54.88, 54.31, 53.95])) +  # LUCIR
                         list(np.array([57.88, 56.05, 56.84, 58.37, 60.02])) +  # BIC
                         list(np.array([49.66, 48.6, 49.83, 49.69, 48.1])) +  # XDER
                         list(np.array([29.84, 29.52, 27.79, 29.24, 28.87])) +  # AGEM
                         list(np.array([41.53, 41.01, 42.31, 41.73, 41.04])) +  # DER
                         list(np.array([41.73, 41.6, 41.23, 41.73, 39.7])) +  # DERpp
                         list(np.array([44.99, 44.47, 43.58, 44.77, 45.28])) +  # ER
                         list(np.array([25.11, 25.02, 25.96, 24.23, 25.11])) +  # EWC
                         list(np.array([40.66, 39.13, 40.00, 40.39, 38.85])) +  # LwF
                         list(np.array([46.59, 47.17, 45.52, 46.91, 46.8]))  # SI
                         )
    colors_superclass = [nature_palette_hex[0]] * 1 + [nature_palette_hex[2]] * 3 + [nature_palette_hex[9]] * 4 + [nature_palette_hex[5]] * 3

    # five_datasets
    groups_five_datasets = (['AdaMLearn'] * 5 + ['XDER'] * 5 +
                            ['AGEM'] * 5 + ['DER'] * 5 + ['DERpp'] * 5 + ['ER'] * 5 + ['EWC'] * 5 + ['LwF'] * 5 + ['SI'] * 5)
    values_five_datasets = (list(np.array([91.99, 91.2, 91.6, 91.77, 91.27])) +  # EIM
                            list(np.array([87.48, 89.47, 89.33, 89.34, 89.52])) +  # XDER
                            list(np.array([48.43, 65.48, 54.79, 61.97, 55.61])) +  # AGEM
                            list(np.array([82.55, 82.46, 83.02, 81.61, 82.72])) +  # DER
                            list(np.array([79.53, 82.86, 81.99, 81.55, 80.64])) +  # DERpp
                            list(np.array([77.64, 77.5, 77.3, 76.97, 77.64])) +  # ER
                            list(np.array([45.75, 50.07, 52.55, 49.8, 58.2])) +  # EWC
                            list(np.array([57.65, 55.97, 59.82, 60.45, 57.77])) +  # LwF
                            list(np.array([39.56, 42.92, 36.95, 43.25, 37.32]))  # SI
                            )
    colors_five_datasets = [nature_palette_hex[0]] * 1 + [nature_palette_hex[2]] * 1 + [nature_palette_hex[9]] * 4 + [nature_palette_hex[5]] * 3

    # mini_imagenet
    groups_mini_imagenet = (['AdaMLearn'] * 5 + ['LUCIR'] * 5 + ['BIC'] * 5 + ['XDER'] * 5 +
                            ['AGEM'] * 5 + ['DER'] * 5 + ['DERpp'] * 5 + ['ER'] * 5 + ['LwF'] * 5 + ['SI'] * 5)
    values_mini_imagenet = (list(np.array([72.54, 71.1, 72.41, 72.4, 70.98])) +  # EIM
                            list(np.array([57.42, 58.86, 59.84, 56.69, 61.21])) +  # LUCIR
                            list(np.array([68.52, 71, 69.18, 72.94, 76.1])) +  # BIC
                            list(np.array([70.97, 73.73, 67.51, 72.12, 75.08])) +  # XDER
                            list(np.array([33.99, 32.43, 34.22, 40.22, 33.3])) +  # AGEM
                            list(np.array([56.92, 59.16, 53.06, 57.27, 56.59])) +  # DER
                            list(np.array([55.32, 59.92, 53.32, 58.72, 58.61])) +  # DERpp
                            list(np.array([59.72, 55.82, 50.79, 52.24, 55.09])) +  # ER
                            list(np.array([22.65, 25.26, 24.68, 29.12, 26.64])) +  # LwF
                            list(np.array([26.51, 26.12, 26.65, 25.55, 26.94]))  # SI
                            )
    colors_mini_imagenet = [nature_palette_hex[0]] * 1 + [nature_palette_hex[2]] * 3 + [nature_palette_hex[9]] * 4 + [nature_palette_hex[5]] * 2

    # 创建4个子图 (2行2列)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 调整子图间距

    # 创建DataFrame
    df = []
    df.append(pd.DataFrame({'Methods': groups_cifar100, 'Average Accuracy(%)': values_cifar100}))
    df.append(pd.DataFrame({'Methods': groups_superclass, 'Average Accuracy(%)': values_superclass}))
    df.append(pd.DataFrame({'Methods': groups_five_datasets, 'Average Accuracy(%)': values_five_datasets}))
    df.append(pd.DataFrame({'Methods': groups_mini_imagenet, 'Average Accuracy(%)': values_mini_imagenet}))

    colors = []
    colors.append(colors_cifar100)
    colors.append(colors_superclass)
    colors.append(colors_five_datasets)
    colors.append(colors_mini_imagenet)

    titles = []
    titles.append('CIFAR-100')
    titles.append('CIFAR-superclass')
    titles.append('FIVE')
    titles.append('Mini-imagenet')

    # 绘制图表
    for i, ax in enumerate(axes.flat):
        sns.barplot(x='Methods', y='Average Accuracy(%)', data=df[i], errorbar='sd', palette=colors[i], capsize=0.1, width=0.6, alpha=0.9, ax=ax)
        for line in ax.lines:
            line.set_color('black')  # 误差条颜色
            line.set_alpha(0.5)  # 透明度
            line.set_linewidth(3)  # 线宽
        for p in ax.patches:
            ax.text(
                x=p.get_x() + p.get_width() / 2,  # x坐标：柱子中心
                y=p.get_height() + 2.0,  # y坐标：柱子高度上方
                s=f"{p.get_height():.2f}",  # 显示的值（保留2位小数）
                ha='center',  # 水平居中
                va='bottom',  # 垂直底部对齐
                fontsize=16
            )
        sns.swarmplot(x='Methods', y='Average Accuracy(%)', data=df[i], color='black', size=5, alpha=0.5, ax=ax)
        # 添加标题/在右上角添加标题（坐标是相对轴的比例，范围0~1）
        # ax.set_title(titles[i], fontsize=14, pad=10)
        ax.text(
            x=0.95,  # x坐标（1.0为最右侧）
            y=0.95,  # y坐标（1.0为最顶部）
            s=titles[i],  # 标题文本
            ha="right",  # 水平对齐（右对齐）
            va="top",  # 垂直对齐（顶部对齐）
            transform=ax.transAxes,  # 使用轴坐标系
            fontsize=20,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")  # 背景框
        )
        ax.set_xlabel("")
        # 刻度字体大小
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # 方法名称刻度对齐
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        ax.set_xticks(np.arange(len(labels)) + 0.3)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Average Accuracy (%)", fontsize=24)
        # 去除上和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.grid(False)
    plt.savefig("./pdfs/figure_AAC_1-1.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    draw_bar()
