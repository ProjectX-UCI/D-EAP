import pandas as pd

def plotDF(df,columns,title,ylabel):
    df = pd.DataFrame(df, columns=columns)
    plot = df.plot(title=title)
    plot.set(xlabel="Time/Iteration", ylabel=ylabel)