import matplotlib.pyplot as plt
from IPython import display
import os

plt.ion()

def save(file_name='plot.png'):
    plot_folder_path = './figures'
    if not os.path.exists(plot_folder_path):
        os.makedirs(plot_folder_path)
        
    final_path = os.path.join(plot_folder_path, file_name)
    plt.savefig(final_path)

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Number of Objects vs. Simulations')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Objects Successfully Found')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    save()