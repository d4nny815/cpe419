import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def load_and_sort_data(seq_path, para_path):
    seq_data = pd.read_csv(seq_path).sort_values(by="N")
    para_data = pd.read_csv(para_path).sort_values(by="N")
    return (seq_data, para_data)

def plot_results(seq_data, para_data, save_path="N_time_bar.png"):
    """Plots computation time as a bar chart, grouped by N for all four methods."""
    
    # Extract unique problem sizes (N)
    Ns = seq_data["N"].unique()

    # Prepare data for bar chart
    neighbor_seq = seq_data.set_index("N")[" tNeighbors"]
    stride_seq = seq_data.set_index("N")[" tStride"]
    neighbor_para = para_data.set_index("N")[" tNeighbors"]
    stride_para = para_data.set_index("N")[" tStride"]

    # Bar width and positions
    bar_width = 0.2
    x_indexes = np.arange(len(Ns))

    plt.figure(figsize=(10, 6))

    # Plot bars
    plt.bar(x_indexes - 1.5 * bar_width, neighbor_seq[Ns], bar_width, color="b", label="Neighbor Reduction (Sequential)")
    plt.bar(x_indexes - 0.5 * bar_width, stride_seq[Ns], bar_width, color="r", label="Stride Reduction (Sequential)")
    plt.bar(x_indexes + 0.5 * bar_width, neighbor_para[Ns], bar_width, color="g", label="Neighbor Reduction (Parallel)")
    plt.bar(x_indexes + 1.5 * bar_width, stride_para[Ns], bar_width, color="y", label="Stride Reduction (Parallel)")

    # Annotate bars with time values
    for i, N in enumerate(Ns):
        plt.text(x_indexes[i] - 1.5 * bar_width, neighbor_seq[N], f"{neighbor_seq[N]:.6f}", ha="center", va="bottom", fontsize=8, color="black")
        plt.text(x_indexes[i] - 0.5 * bar_width, stride_seq[N], f"{stride_seq[N]:.6f}", ha="center", va="bottom", fontsize=8, color="black")
        plt.text(x_indexes[i] + 0.5 * bar_width, neighbor_para[N], f"{neighbor_para[N]:.6f}", ha="center", va="bottom", fontsize=8, color="black")
        plt.text(x_indexes[i] + 1.5 * bar_width, stride_para[N], f"{stride_para[N]:.6f}", ha="center", va="bottom", fontsize=8, color="black")

    # Formatting
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xticks(x_indexes, [f"{N:.0e}" for N in Ns]) 
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time (Seconds)")
    plt.title("Computation Time for Different Reduction Methods")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_table_as_image(seq_data, para_data, save_path="N_time_table.png"):
    # Merge and sort data
    merged_data = pd.concat([seq_data.assign(Type="Sequential"), para_data.assign(Type="Parallel")])
    merged_data = merged_data.sort_values(by=["Type", "N"])
    
    # convert N into engineering notation
    merged_data["N"] = merged_data["N"].apply(lambda x: "{:.3e}".format(x))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")

    table_data = [merged_data.columns.tolist()] + merged_data.values.tolist()
    table = ax.table(cellText=table_data, colLabels=None, cellLoc="center", loc="center")

    table.auto_set_font_size(True)
    table.auto_set_column_width([i for i in range(len(merged_data.columns))])

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    seq_file = "seq/results.csv"
    para_file = "cuda/results.csv"

    seq_data, para_data = load_and_sort_data(seq_file, para_file)

    plot_results(seq_data, para_data, "N_time.png")
    # save_table_as_image(seq_data, para_data, "N_time_table.png")