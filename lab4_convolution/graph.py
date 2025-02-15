import matplotlib.pyplot as plt
import pandas as pd


def load_and_sort_data(seq_path, para_path):
    seq_data = pd.read_csv(seq_path).sort_values(by=" N")
    para_data = pd.read_csv(para_path).sort_values(by=" N")
    return (seq_data, para_data)


def plot_results(seq_data, para_data, save_path="N_time.png"):
    plt.figure()

    plt.scatter(seq_data[" N"], seq_data[" time (sec)"], color="b", marker="o", label="Sequential Points")
    plt.plot(seq_data[" N"], seq_data[" time (sec)"], linestyle="-", color="b", alpha=0.3)
    
    plt.scatter(para_data[" N"], para_data[" time (sec)"], color="r", marker="o", label="Parallel Points")
    plt.plot(para_data[" N"], para_data[" time (sec)"], linestyle="-", color="r", alpha=0.3)

    # each point with time values
    for i, row in seq_data.iterrows():
        plt.annotate(f"{row[' time (sec)']:.3f}", (row[" N"], row[" time (sec)"]), 
                     textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, color="blue")

    for i, row in para_data.iterrows():
        plt.annotate(f"{row[' time (sec)']:.3f}", (row[" N"], row[" time (sec)"]), 
                     textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, color="red")

    # Formatting
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("Time (Seconds)")
    plt.title("Computation Time vs. Problem Size (Lower is better)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_table_as_image(seq_data, para_data, save_path="N_time_table.png"):
    # Merge and sort data
    merged_data = pd.concat([seq_data.assign(Type="Sequential"), para_data.assign(Type="Parallel")])
    merged_data = merged_data.sort_values(by=["Type", " N"])
    
    # convert N into engineering notation
    merged_data[" N"] = merged_data[" N"].apply(lambda x: "{:.3e}".format(x))

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