import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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
    10: "#74C476",
    11: "#FDAE6B",
    12: "#B3B3B3"
}

custom_palette = {
    'Ours': nature_palette_hex[0],
    'XDER': nature_palette_hex[5],
    'DER': nature_palette_hex[1],
    'DER++': nature_palette_hex[2],
    'ER': nature_palette_hex[3]
}

memory_palette = {
    'AdaMLearn': nature_palette_hex[10],
    'Disturb': nature_palette_hex[11],
    'SGD': nature_palette_hex[12]
}

def draw_line_retrieval():
    # 设置样式
    sns.set_style("white")

    # cifar100记忆占用率
    # 构建三元组标签
    groups_cifar100_t1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    values_cifar100_t1 = (
            list(np.array([78.3, 78.5, 76.9, 78.8, 79.5, 78.4, 78.8, 78.7, 79.9, 79.3])) +  # EN
            list(np.array([78.3, 75.2, 75.9, 73.5, 72.5, 71.2, 70.0, 70.2, 70.1, 69.0])) +  # disturb
            list(np.array([77.6, 1.0, 0.3, 0.2, 0.5, 0.2, 0.2, 0.5, 0.2, 0.2])) # SGD
    )

    groups_cifar100_t2 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

    values_cifar100_t2 = (
            list(np.array([70.9, 70.6, 70.7, 71.2, 70.9, 71.3, 72.6, 71.5, 71.9])) +  # EN
            list(np.array([70.9, 68.1, 69.4, 68.6, 66.5, 66.1, 65.5, 66.1, 64.9])) +  # disturb
            list(np.array([74.1, 1.3, 0.9, 0.7, 1.7, 2.1, 0.9, 1.0, 0.5]))  # SGD
    )

    groups_cifar100_t3 = np.array([3, 4, 5, 6, 7, 8, 9, 10])

    values_cifar100_t3 = (
            list(np.array([77.6, 77.5, 77.5, 77.9, 77.5, 77.7, 78.1, 75.7])) +  # EN
            list(np.array([77.8, 76.0, 75.8, 73.6, 76.2, 73.1, 71.4, 69.7])) +  # disturb
            list(np.array([75.4, 0.6, 0.4, 0.4, 0.5, 1.1, 0.0, 0.3]))  # SGD
    )

    groups_cifar100_t4 = np.array([4, 5, 6, 7, 8, 9, 10])

    values_cifar100_t4 = (
            list(np.array([72.7, 73.8, 72.9, 71.8, 74.1, 73.0, 73.7])) +  # EN
            list(np.array([75.1, 71.6, 69.4, 70.4, 68.6, 69.1, 69.1])) +  # disturb
            list(np.array([75.5, 0.7, 2.0, 0.5, 0.5, 0.2, 0.6]))  # SGD
    )

    groups_cifar100_t5 = np.array([5, 6, 7, 8, 9, 10])

    values_cifar100_t5 = (
            list(np.array([76.2, 75.7, 76.3, 76.2, 77.1, 76.3])) +  # EN
            list(np.array([79.6, 77.4, 75.6, 75.1, 75.2, 76.7])) +  # disturb
            list(np.array([77.5, 0.6, 0.2, 0.1, 1.0, 1.0]))  # SGD
    )

    groups_cifar100_t6 = np.array([6, 7, 8, 9, 10])

    values_cifar100_t6 = (
            list(np.array([75.2, 75.0, 75.6, 75.8, 76.1])) +  # EN
            list(np.array([78.9, 77.4, 75.6, 75.6, 74.1])) +  # disturb
            list(np.array([78.7, 0.9, 1.3, 0.8, 0.0]))  # SGD
    )

    all_groups = [groups_cifar100_t1, groups_cifar100_t2, groups_cifar100_t3, groups_cifar100_t4,
                  groups_cifar100_t5, groups_cifar100_t6]
    all_values = [values_cifar100_t1, values_cifar100_t2, values_cifar100_t3, values_cifar100_t4,
                  values_cifar100_t5, values_cifar100_t6]
    stage_labels = ['t1', 't2', 't3', 't4', 't5', 't6']
    methods = ['AdaMLearn', 'Disturb', 'SGD']

    df_all = pd.DataFrame()

    for stage, groups, values in zip(stage_labels, all_groups, all_values):
        group_len = len(groups)
        total_expected_len = group_len * len(methods)
        assert len(
            values) == total_expected_len, f"Mismatch in {stage}: Expected {total_expected_len}, got {len(values)}"

        # 切成每个 method 的一段
        values_split = [values[i * group_len: (i + 1) * group_len] for i in range(len(methods))]

        for method, vals in zip(methods, values_split):
            df = pd.DataFrame({
                'Number of tasks': groups,
                'Memory usage (%)': vals,
                'Method': [method] * group_len,
                'Stage': [stage] * group_len
            })
            df_all = pd.concat([df_all, df], ignore_index=True)

    # 设置样式
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=False)
    axes = axes.flatten()

    for i, stage in enumerate(stage_labels):
        ax = axes[i]
        data = df_all[df_all['Stage'] == stage]

        if data.empty:
            print(f"[Warning] Stage {stage} has no data!")
            continue

        plot = sns.lineplot(
            data=data,
            x='Number of tasks',
            y='Memory usage (%)',
            hue='Method',
            style='Method',
            markers=True,
            dashes=False,
            palette=memory_palette,
            ax=ax,
            legend=True,  # Changed from False to True
            linewidth=5,
            alpha=1.0,
            markersize=12
        )
        # Ensure per-axis legend is shown in the lower-right, nudged upward to avoid overlap
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title(None)
            ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.05), fontsize=18, frameon=True)

        ax.set_title(f"Context {stage}", fontsize=24)
        ax.set_ylabel("Average Accuracy (%)", fontsize=22)
        ax.set_xlabel("Number of tasks", fontsize=22)
        ax.tick_params(axis='both', labelsize=20)
        # ax.get_legend().remove()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        # ax.set_ylim(70, 80)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_ylim(0, 80)




    # Removed the shared legend block

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./pdfs/figure_retrieval_all_fixed.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    draw_line_retrieval()
