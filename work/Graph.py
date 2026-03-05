import matplotlib.pyplot as plt 
from typing import List
import os  
os.makedirs("DTA_BOT/Graphs", exist_ok=True)
# -------------------------------------
def line_graph(df, x_axis:str, y_axis:str):
    plt.plot(df[x_axis], df[y_axis], marker="o", markerfacecolor="black", color="blue")
    plt.title("Ploted graph")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.savefig("DTA_BOT/Graphs/line_graph.png")
# --------------------------------------
def bar_graph(df, x_axis:str, y_axis:str):
    plt.bar(df[x_axis], df[y_axis], color="blue")
    plt.title("Ploted bar graph")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.savefig("DTA_BOT/Graphs/bar_graph.png")

