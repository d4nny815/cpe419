import matplotlib.pyplot as plt
import pandas as pd

seq_data = pd.read_csv("seq/results.csv")
para_data = pd.read_csv("cuda/results.csv")

seq_data = seq_data.sort_values(by=" N")
para_data = para_data.sort_values(by=" N")

plt.figure()
plt.scatter(seq_data[' N'], seq_data[' time (sec)'], 
            color='b', marker='o', label='Seqeuntial Points') # plot the points
plt.plot(seq_data[' N'], seq_data[' time (sec)'], 
        linestyle='-', color='b', alpha=0.3) # plot the trendline

plt.scatter(para_data[' N'], para_data[' time (sec)'], 
            color='r', marker='o', label='Parallel Points') # plot the points
plt.plot(para_data[' N'], para_data[' time (sec)'], 
        linestyle='-', color='r', alpha=0.3) # plot the trendline

plt.xscale("log")
# plt.yscale("log")

plt.xlabel("N")
plt.ylabel("Time (Seconds)")
plt.title("Computation Time vs. Problem Size (Lower is better)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()