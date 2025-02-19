import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_and_sort_data(seq_path, para_path):
    seq_data = pd.read_csv(seq_path).sort_values(by="N")
    para_data = pd.read_csv(para_path).sort_values(by="N")
    return (seq_data, para_data)

def plot_results(seq_data, para_data, save_path="N_time_bar.png"):
    # Extract unique problem sizes (N)
    Ns = seq_data["N"].unique()

    # Prepare data for bar chart
    neighbor_seq = seq_data.set_index("N")["tNeighbors"]
    stride_seq = seq_data.set_index("N")["tStride"]
    seg_scan_seq = seq_data.set_index("N")["tScan"]
    neighbor_para = para_data.set_index("N")["tNeighbors"]
    stride_para = para_data.set_index("N")["tStride"]
    seg_scan_para = para_data.set_index("N")["tScan"]

    # Bar width and positions
    bar_width = 0.1
    x_indexes = np.arange(len(Ns))

    plt.figure(figsize=(16, 8))

    # Plot bars with adjusted positions
    bars1 = plt.bar(x_indexes - 2.5 * bar_width, neighbor_seq[Ns], bar_width, color="b", label="Neighbor Reduction (Sequential)")
    bars2 = plt.bar(x_indexes - 1.5 * bar_width, neighbor_para[Ns], bar_width, color="g", label="Neighbor Reduction (Parallel)")
    bars3 = plt.bar(x_indexes - 0.5 * bar_width, stride_seq[Ns], bar_width, color="r", label="Stride Reduction (Sequential)")
    bars4 = plt.bar(x_indexes + 0.5 * bar_width, stride_para[Ns], bar_width, color="y", label="Stride Reduction (Parallel)")
    bars5 = plt.bar(x_indexes + 1.5 * bar_width, seg_scan_seq[Ns], bar_width, color="orange", label="SegScan Reduction (Sequential)")
    bars6 = plt.bar(x_indexes + 2.5 * bar_width, seg_scan_para[Ns], bar_width, color="purple", label="SegScan Reduction (Parallel)")

    # Function to annotate bars
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only annotate if height is greater than 0
                plt.text(bar.get_x() + bar.get_width() / 2, height * 1.05, 
                         f"{height:.6f}", ha="center", va="bottom", fontsize=8, color="black")

    # Annotate each set of bars
    for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
        annotate_bars(bars)
    
    # Formatting
    plt.yscale("log")
    plt.xticks(x_indexes, [f"{N}" for N in Ns])
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time (Seconds)")
    plt.title("N vs Computation Time for Reduce Methods")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_table_as_image(seq_data, para_data, save_path="N_time_table.png"):
    # Merge and sort data
    merged_data = pd.concat([seq_data.assign(Type="Sequential"), para_data.assign(Type="Parallel")])
    merged_data = merged_data.sort_values(by=["Type", "N"])
    
    # Convert N into engineering notation for the table
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
    save_table_as_image(seq_data, para_data, "N_time_table.png")
