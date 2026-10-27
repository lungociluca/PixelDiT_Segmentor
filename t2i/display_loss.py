import matplotlib.pyplot as plt
import sys


def read_loss_values(filename, start, end):
    with open(filename) as f:
        lines = f.read().splitlines()[start:end]
    return [float(x) for x in lines if x.strip()]


start = int(sys.argv[1])
end = int(sys.argv[2])

prev_loss_values = read_loss_values("prev_loss_logs.txt", start, end)
loss_values = read_loss_values("loss_logs.txt", start, end)

plt.scatter(range(len(prev_loss_values)), prev_loss_values, label="Previous loss")
plt.scatter(range(len(loss_values)), loss_values, label="Current loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.close()