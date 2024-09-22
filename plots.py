import numpy as np
import matplotlib.pyplot as plt

def plot_losses(filename='plot_losses.npy', output='loss_plots.png'):
    # Load plot data
    plot_losses = np.load(filename)

    # Plot loss
    plt.figure(figsize=(12,8))
    plt.plot(plot_losses[:,0], plot_losses[:,1], color='b', linewidth=4)
    plt.plot(plot_losses[:,0], plot_losses[:,2], color='r', linewidth=4)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.grid()
    plt.legend(['training', 'validation'])
    plt.savefig(output)
    plt.close()

    print(f"Loss plot saved as '{output}'")

