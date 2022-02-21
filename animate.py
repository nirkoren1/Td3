import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xar = []
yar = []
plt.ion()
fig1 = plt.figure(figsize=(6, 4.5), dpi=100)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_ylim(0, 110)
ax1.set_title('Reward')
line, = ax1.plot(xar, yar, 'r', marker='o')


def update(y):
    yar.append(y)
    xar.append(len(yar))
    ax1.set_xlim(-5, len(xar) + 5)
    ax1.set_ylim(min(yar) - 10, max(yar) + 30)
    line.set_data(xar, yar)
    fig1.canvas.draw()
    fig1.canvas.flush_events()
