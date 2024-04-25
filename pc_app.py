import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt



def x_update_pc(x, w, b, idx, dt):
    return x[idx] - dt * ((x[idx] - w[idx]*x[idx+1] - b[idx]) - w[idx-1]*(x[idx-1] - w[idx-1]*x[idx] - b[idx-1]))

def x_update_mcpc(x, w, b, idx, dt):
    return x[idx] - dt * ((x[idx] - w[idx]*x[idx+1] - b[idx]) - w[idx-1]*(x[idx-1] - w[idx-1]*x[idx] - b[idx-1])) + np.sqrt(2*dt)*np.random.randn()

def x_update_post(x, w, b, idx, dt):
    return (w[idx]*x[idx+1] + b[idx] + w[idx-1]*(x[idx-1] - b[idx-1]))/(w[idx-1]**2 + 1)

def x_update_root(x, w, b, idx, dt):
    return w[idx]*x[idx+1] + b[idx] + w[idx-1]*(x[idx-1] - w[idx-1]*x[idx] - b[idx-1])

def w_update_pc(x, w, b, idx, dtw, decay, bound):
    update =  w[idx] + dtw * ((x[idx] - x[idx+1]*w[idx] - b[idx]) * x[idx+1] - decay*w[idx])
    if bound is None:
        return update
    else:
        return max(-np.abs(bound), min(np.abs(bound), update))

def b_update_pc(x, w, b, idx, dtw, decay, bound):
    update = b[idx] + dtw * (x[idx] - x[idx+1]*w[idx] - b[idx] - decay*b[idx])
    if bound is None:
        return update
    else:
        return max(-np.abs(bound), min(np.abs(bound), update))



def generate_spring(x1, x2, y1, num_coils=10):
    """ Generate points that simulate a spring between two points. """
    # Linear interpolation of points along the spring
    t = np.linspace(0, 1, 100)
    x_line = x1 + (x2 - x1) * t
    y_line = y1 * np.ones_like(t)
    amplitude = 0.1  # Amplitude proportional to the vertical distance
    spring_x = x_line 
    spring_y = y_line + amplitude * np.sin(num_coils * np.pi * t)
    return spring_x, spring_y

def measure_error(x_data, w, b):
    return round(sum([(x_data[i] - x_data[i+1]*w[i] - b[i])**2 for i in range(len(w))]), 2)


class SimulationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Predictive coidng simulator")

        # Creating frames for layout
        self.frame_controls = ttk.Frame(self.master)
        self.frame_controls.pack(side="right", fill="y")

        self.frame_animation = ttk.Frame(self.master)
        self.frame_animation.pack(side="left", fill="both", expand=True)

        # Parameters and variables initialization
        self.num_time_steps = 5
        self.dt = 0.1
        self.dtw = 5.0
        self.x_data = [-0.2, np.random.randn(), np.random.randn(), 0.1]
        self.w = [np.random.rand()+ 0.5 for i in range(len(self.x_data)-1)]
        self.b = [0. for i in range(len(self.x_data)-1)]

        self.p_decay = 0
        self.p_bound = None

        self.layer_update = x_update_pc

        self.x_update = [
            lambda x, w, b, idx, dt: x[idx],
            lambda x, w, b, idx, dt: self.layer_update(x, w, b, idx, dt),
            lambda x, w, b, idx, dt: self.layer_update(x, w, b, idx, dt),
            lambda x, w, b, idx, dt: x[idx]
        ]

        self.weight_update = w_update_pc
        self.w_update = [
            lambda x, w, b, idx, dtw, decay, bound: self.weight_update(x, w, b, idx, dtw, decay, bound),
            lambda x, w, b, idx, dtw, decay, bound: self.weight_update(x, w, b, idx, dtw, decay, bound),
            lambda x, w, b, idx, dtw, decay, bound: self.weight_update(x, w, b, idx, dtw, decay, bound)
        ]

        self.bias_update = b_update_pc
        self.b_update = [
            lambda x, w, b, idx, dtw, decay, bound: self.bias_update(x, w, b, idx, dtw, decay, bound),
            lambda x, w, b, idx, dtw, decay, bound: self.bias_update(x, w, b, idx, dtw, decay, bound),
            lambda x, w, b, idx, dtw, decay, bound: self.bias_update(x, w, b, idx, dtw, decay, bound)
        ]

        # assess imputs from user
        assert len(self.x_update) == len(self.x_data)  # Check that the number of functions matches the number of points
        assert len(self.w)+1 == len(self.x_data)  # Check that the number of weights matches the number of points
        assert len(self.w_update) == len(self.w)  # Check that the number of functions matches the number of weights
        self.y_positions = np.arange(1, len(self.x_data)+1)  # Vertical positions of the points

        # Create matplotlib figure and axes
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim((-3, 3))
        self.ax.set_ylim((0, len(self.x_data)+1))
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.set_xlabel('x')

        # Adding horizontal dashed lines
        for y in self.y_positions:
            self.ax.axhline(y=y, color='gray', linestyle='dashed', linewidth=1)
        self.ax.axvline(x = 0, color='gray', linestyle='dashed', linewidth=1)
        self.points = [self.ax.plot([], [], 'ko')[0] for _ in self.y_positions]
        self.lines = [self.ax.plot([], [], 'k-')[0] for _ in range(len(self.y_positions)-1)]  # Red lines connecting the points
        self.springs = [self.ax.plot([], [], 'b-')[0] for _ in range(len(self.y_positions)-1)]  # Red lines connecting the points
        self.train_number_text = self.ax.text(0.05, 0.95, '', transform= self.ax.transAxes, fontsize=12)  # Positioning the text in the top left
        self.energy_text = self.ax.text(0.75, 0.95, '', transform=self.ax.transAxes, fontsize=12)  # Positioning the text in the top left

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_animation)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Controls for the parameters
        self.option_label = ttk.Label(self.frame_controls, text="Neuron dynamics")
        self.option_label.pack()
        self.option_combobox = ttk.Combobox(self.frame_controls, values=["pc", "mcpc", "post", "root"])
        self.option_combobox.pack()
        self.option_combobox.current(0)  # Default to the first option "pc"

        self.input_label = ttk.Label(self.frame_controls, text="Input")
        self.input_label.pack()
        self.input_entry = ttk.Entry(self.frame_controls)
        self.input_entry.pack()
        self.input_entry.insert(0, "0.2")

        self.latent_label = ttk.Label(self.frame_controls, text="Latent")
        self.latent_label.pack()
        self.latent_entry = ttk.Entry(self.frame_controls)
        self.latent_entry.pack()
        self.latent_entry.insert(0, "1.0")

        self.num_steps_label = ttk.Label(self.frame_controls, text="Number of steps")
        self.num_steps_label.pack()
        self.num_steps_entry = ttk.Entry(self.frame_controls)
        self.num_steps_entry.pack()
        self.num_steps_entry.insert(0, "10")

        self.dt_label = ttk.Label(self.frame_controls, text="dt")
        self.dt_label.pack()
        self.dt_entry = ttk.Entry(self.frame_controls)
        self.dt_entry.pack()
        self.dt_entry.insert(0, "0.1")  # Default value for dt

        self.dtw_label = ttk.Label(self.frame_controls, text="dtp")
        self.dtw_label.pack()
        self.dtw_entry = ttk.Entry(self.frame_controls)
        self.dtw_entry.pack()
        self.dtw_entry.insert(0, "1.0")  # Default value for dtw

        self.p_d_label = ttk.Label(self.frame_controls, text="parameter decay")
        self.p_d_label.pack()
        self.p_d_entry = ttk.Entry(self.frame_controls)
        self.p_d_entry.pack()
        self.p_d_entry.insert(0, "0.0")  # Default value for dtw

        self.p_b_label = ttk.Label(self.frame_controls, text="parameter bound")
        self.p_b_label.pack()
        self.p_b_entry = ttk.Entry(self.frame_controls)
        self.p_b_entry.pack()
        self.p_b_entry.insert(0, "None")  # Default value for dtw

        self.start_button = ttk.Button(self.frame_controls, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack()
        self.stop_button = ttk.Button(self.frame_controls, text="Stop Simulation", command=self.stop_simulation)
        self.stop_button.pack()

        # Initialize animation
        self.ani = None

    def update_dt(self, val):
        assert float(val) >= 0
        self.dt = float(val)

    def update_dtw(self, val):
        assert float(val) >= 0
        self.dtw = float(val)

    def update_p_decay(self,val):
        assert float(val) >= 0
        self.p_decay = float(val)

    def update_p_bound(self, val):
        if val == "None":
            self.p_bound = None
        else:
            assert float(val) > 0
            self.p_bound = float(val)

    def update_num_time_steps(self, val):
        assert int(val) > 0
        self.num_time_steps = int(val)

    def update_option(self, val):
        self.selected_option = val
        if val == "pc":
            self.layer_update = x_update_pc
        elif val =="mcpc":
            self.layer_update = x_update_mcpc
        elif val == "post":
            self.layer_update = x_update_post
        elif val == "root":
            self.layer_update = x_update_root
        else:
            raise ValueError(f"Invalid option: {val}")

    def update_input(self, val):
        self.x_data[0] = float(val)

    def update_latent(self, val):
        self.x_data[-1] = float(val)

    def sample_hidden(self):
        self.x_data[1] = np.random.randn()*4
        self.x_data[2] = np.random.randn()*4

    def sample_weights(self):
        for idx in range(len(self.w)):
            self.w[idx] = 1.0 # np.random.rand() + 0.5
    
    def sample_bias(self):
        for idx in range(len(self.b)):
            self.b[idx] = 0.

    def start_simulation(self):
        try:
            self.update_dt(self.dt_entry.get())  # Update dt from the entry widget
            self.update_dtw(self.dtw_entry.get())  # Update dtw from the entry widget
            self.update_num_time_steps(self.num_steps_entry.get())  # Update num_time_steps from the entry widget
            self.update_option(self.option_combobox.get())  # Update the selected option
            self.update_input(self.input_entry.get())  # Update the input value
            self.update_latent(self.latent_entry.get())  # Update the latent value
            self.update_p_decay(self.p_d_entry.get())
            self.update_p_bound(self.p_b_entry.get())
            self.sample_hidden()
            self.sample_weights()
            self.sample_bias()
        except ValueError:
            return    

        print(f"Selected option: {self.selected_option}")  # Display or use this value in your simulation logic
        print(f"dt: {self.dt}")
        print(f"dtw: {self.dtw}")
        print(f"decay: {self.p_decay}")
        print(f"bound: {self.p_bound}")
        print(f"Number of time steps: {self.num_time_steps}")

        self.ani = FuncAnimation(self.fig, self.update, frames=self.num_time_steps, init_func=self.init, blit=True)

    def stop_simulation(self):
        if self.ani is not None:
            self.ani.event_source.stop()

    def init(self):
        for point in self.points:
            point.set_data([], [])
        for line in self.lines:
            line.set_data([], [])
        for spring in self.springs:
            spring.set_data([], [])
        self.train_number_text.set_text('')
        self.energy_text.set_text('')
        return self.points + self.lines + self.springs + [self.train_number_text, self.energy_text]

    def update(self, frame):
        for i in range(len(self.x_data)):
            self.x_data[i] = self.x_update[i](self.x_data, self.w, self.b, i, self.dt)
        for idx, point in enumerate(self.points):
            point.set_data(self.x_data[idx], self.y_positions[idx])
        for idx, line in enumerate(self.lines):
            line.set_data([self.x_data[idx+1], self.x_data[idx+1]*self.w[idx] + self.b[idx]], [self.y_positions[idx+1], self.y_positions[idx]])
        for idx, spring in enumerate(self.springs):
            spring_x, spring_y = generate_spring(self.x_data[idx+1]*self.w[idx] + self.b[idx], self.x_data[idx], self.y_positions[idx])
            spring.set_data(spring_x, spring_y)
        self.train_number_text.set_text(f'n = {frame}')
        self.energy_text.set_text(f'E = {measure_error(self.x_data, self.w, self.b)}')

        # update weights
        if frame == self.num_time_steps - 1:
            for idx in range(len(self.w)):
                self.w[idx] = self.w_update[idx](self.x_data, self.w, self.b, idx, self.dtw, self.p_decay, self.p_bound)
                self.b[idx] = self.b_update[idx](self.x_data, self.w, self.b, idx, self.dtw, self.p_decay, self.p_bound)
            print(self.w)
            print(self.b)
            self.sample_hidden()
        return self.points + self.lines + self.springs + [self.train_number_text, self.energy_text]

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()
