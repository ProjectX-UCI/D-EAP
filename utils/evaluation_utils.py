import pandas as pd
import matplotlib.pyplot as plt

def plotDF(df,columns,title,ylabel,plot_figure=False,save_figure=False):
    df = pd.DataFrame(df, columns=columns)
    plot = df.plot(title=title)
    plot.set(xlabel="Time/Iteration", ylabel=ylabel)

    if plot_figure:
        plt.show()
    
    if save_figure:
        plt.savefig("./results/%s.jpg"%title)