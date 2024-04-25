import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def x_update_pc(x, w, idx, dt):
    return x[idx] - dt * ((x[idx] - w[idx]*x[idx+1]) - w[idx-1]*(x[idx-1] - w[idx-1]*x[idx]))

def x_update_post(x, w, idx, dt):
    return (w[idx]*x[idx+1] + w[idx-1]*x[idx-1])/(w[idx-1]**2 + 1)

def x_update_root(x, w, idx, dt):
    return w[idx]*x[idx+1] + w[idx-1]*(x[idx-1] - w[idx-1]*x[idx])

def w_update_pc(x, w, idx, dtw):
    return w[idx] + dtw * (x[idx] - x[idx+1]*w[idx]) * x[idx+1]


# Parameters/options to be set by the user
num_time_steps = 5
dt = 0.1
dtw = 5.
x_data = [-0.2, np.random.randn(), np.random.randn(), 0.1]
w = [0.1, 0.1, 0.1]

x_update = [
    lambda x, w, idx, dt: x[idx],                          
    lambda x, w, idx, dt: x_update_post(x, w, idx, dt),        
    lambda x, w, idx, dt: x_update_post(x, w, idx, dt),        
    lambda x, w, idx, dt: x[idx]                           
]

w_update = [
    lambda x, w, idx, dtw: w_update_pc(x, w, idx, dtw),
    lambda x, w, idx, dtw: w_update_pc(x, w, idx, dtw),
    lambda x, w, idx, dtw: w_update_pc(x, w, idx, dtw)
]


# assess imputs from user
assert len(x_update) == len(x_data)  # Check that the number of functions matches the number of points
assert len(w)+1 == len(x_data)  # Check that the number of weights matches the number of points
assert len(w_update) == len(w)  # Check that the number of functions matches the number of weights

# run the animation
y_positions = np.arange(1, len(x_data)+1)  # Vertical positions of the points
fig, ax = plt.subplots()
ax.set_xlim((-1, 1))
ax.set_ylim((0, len(x_data)+1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_visible(False)
ax.xaxis.set_ticks_position('bottom')  # Only show bottom spine as the x-axis
ax.set_xlabel('x')

# Adding horizontal dashed lines
for y in y_positions:
    ax.axhline(y=y, color='gray', linestyle='dashed', linewidth=1)

points = [ax.plot([], [], 'ko')[0] for _ in y_positions]
lines = [ax.plot([], [], 'k-')[0] for _ in range(len(y_positions)-1)]  # Red lines connecting the points
springs = [ax.plot([], [], 'b-')[0] for _ in range(len(y_positions)-1)]  # Red lines connecting the points
train_number_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)  # Positioning the text in the top left
energy_text = ax.text(0.75, 0.95, '', transform=ax.transAxes, fontsize=12)  # Positioning the text in the top left

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

def measure_error(x_data, w):
    return round(sum([(x_data[i] - x_data[i+1]*w[i])**2 for i in range(len(w))]), 2)

def init():
    for point in points:
        point.set_data([], [])
    for line in lines:
        line.set_data([], [])
    for spring in springs:
        spring.set_data([], [])
    train_number_text.set_text('')
    energy_text.set_text('')
    return points + lines + springs + [train_number_text] + [energy_text]

def update(frame):
    global x_data  # use global to modify the x_data defined at the top level
    x_data = [x_update[i](x_data, w, i, dt) for i in range(len(x_data))]
    for i, point in enumerate(points):
        point.set_data(x_data[i], y_positions[i])
    for i, line in enumerate(lines):
        line.set_data([x_data[i+1], x_data[i+1]*w[i]], [y_positions[i+1], y_positions[i]])
    for i in range(len(springs)):
        spring_x, spring_y = generate_spring(x_data[i+1]*w[i], x_data[i], y_positions[i])
        springs[i].set_data(spring_x, spring_y)
    train_number_text.set_text(f'n = {frame}')
    energy_text.set_text(f'E = {measure_error(x_data, w)}')

    # update weights
    if frame == num_time_steps - 1:
        for i in range(len(w)):
            w[i] = w_update[i](x_data, w, i, dtw)
        # print(w)

    return points + lines + springs + [train_number_text] + [energy_text]

ani = animation.FuncAnimation(fig, update, frames=num_time_steps, init_func=init, blit=True)
plt.show()
