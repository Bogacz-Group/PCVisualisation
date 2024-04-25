import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Matplotlib Animation in Tkinter")

        # Layout: create frames
        self.frame_animation = ttk.Frame(self.root)
        self.frame_animation.pack(side="left", fill="both", expand=True)

        self.frame_controls = ttk.Frame(self.root)
        self.frame_controls.pack(side="right", fill="y")


        # Matplotlib setup
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_animation)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-1.5, 1.5)

        # Controls
        self.speed_scale = ttk.Scale(self.frame_controls, from_=1, to=10, orient="horizontal")
        self.speed_scale.pack()

        self.button = ttk.Button(self.frame_controls, text="Start Animation", command=self.start_animation)
        self.button.pack()

        # Animation
        self.anim = None

    def start_animation(self):
        if self.anim is None:
            self.anim = FuncAnimation(self.fig, self.update_plot, frames=np.linspace(0, 2*np.pi, 128),
                                      init_func=self.plot_init, blit=True, interval=100)

    def plot_init(self):
        self.line.set_data([], [])
        return self.line,

    def update_plot(self, i):
        x = np.linspace(0, 2 * np.pi, 1000)
        y = np.sin(x + i * float(self.speed_scale.get()))
        self.line.set_data(x, y)
        return self.line,

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
