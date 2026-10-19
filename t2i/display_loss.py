import matplotlib.pyplot as plt

with open("loss_logs.txt") as f:
    content = f.read()

loss_values = [float(x) for x in content.split("\n")[500:-1]]

plt.scatter(list(range(len(loss_values))), loss_values)
plt.savefig("loss.png")