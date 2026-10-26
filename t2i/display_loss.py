import matplotlib.pyplot as plt
import sys

with open("loss_logs.txt") as f:
    content = f.read()

start = int(sys.argv[1])
loss_values = [float(x) for x in content.split("\n")[start:-1]]

plt.scatter(list(range(len(loss_values))), loss_values)
plt.savefig("loss.png")
plt.close()