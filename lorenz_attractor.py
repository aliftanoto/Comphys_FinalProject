import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- Lorenz system parameters ---
sigma = 10
beta = 8/3
rho = 28

# --- Lorenz system equations ---
def lorenz_system(t, vector, sigma, beta, rho):
    x, y, z = vector
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Initial conditions
position_0_1 = [5.5, 6.6, 7.7]
position_0_2 = [12.1 , 8.4 , 20.6]

# Time span and evaluation points
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# --- Solve the system ---
sol_1 = solve_ivp(lorenz_system, t_span, position_0_1, args=(sigma, beta, rho), t_eval=t_eval, method='RK45')
sol_2 = solve_ivp(lorenz_system, t_span, position_0_2, args=(sigma, beta, rho), t_eval=t_eval, method='RK45')

x_sol_1, y_sol_1, z_sol_1 = sol_1.y
x_sol_2, y_sol_2, z_sol_2 = sol_2.y

# --- Plotting setup ---
fig = plt.figure(figsize=(12, 8))
ax3d = fig.add_subplot(2, 2, 1, projection='3d')
ax_xt = fig.add_subplot(2, 2, 2)
ax_yt = fig.add_subplot(2, 2, 3)
ax_zt = fig.add_subplot(2, 2, 4)
# Add axis labels for the time plots
ax_xt.set_xlabel('Time (s)')
ax_xt.set_ylabel('x')

ax_yt.set_xlabel('Time (s)')
ax_yt.set_ylabel('y')

ax_zt.set_xlabel('Time (s)')
ax_zt.set_ylabel('z')


lorenz_plt_1, = ax3d.plot([], [], [], 'red', label=f'Initial: {position_0_1}', lw=0.5)
lorenz_plt_2, = ax3d.plot([], [], [], 'blue', label=f'Initial: {position_0_2}', lw=0.5)

line_x1, = ax_xt.plot([], [], 'r-', label='x1(t)')
line_x2, = ax_xt.plot([], [], 'b-', label='x2(t)')
line_y1, = ax_yt.plot([], [], 'r-', label='y1(t)')
line_y2, = ax_yt.plot([], [], 'b-', label='y2(t)')
line_z1, = ax_zt.plot([], [], 'r-', label='z1(t)')
line_z2, = ax_zt.plot([], [], 'b-', label='z2(t)')

time_text = ax3d.text2D(0, 0.95, '', transform=ax3d.transAxes, fontsize=12)

# Axes limits and labels
ax3d.set_title('3D Lorenz Attractor')
ax3d.set_xlim(-30, 30); ax3d.set_ylim(-30, 30); ax3d.set_zlim(0, 50)
ax3d.legend()

ax_xt.set_title('x over time'); ax_xt.set_xlim(t_span); ax_xt.set_ylim(-30, 30); ax_xt.legend()
ax_yt.set_title('y over time'); ax_yt.set_xlim(t_span); ax_yt.set_ylim(-30, 30); ax_yt.legend()
ax_zt.set_title('z over time'); ax_zt.set_xlim(t_span); ax_zt.set_ylim(0, 50); ax_zt.legend()

# --- Pause control ---
paused = [False]  # Use mutable type to allow modification inside event handler

def toggle_pause(event):
    if event.key == ' ':  # Spacebar to pause/resume
        paused[0] = not paused[0]

fig.canvas.mpl_connect('key_press_event', toggle_pause)

# --- Animation update function ---
frame_index = [0]  # mutable index to hold the current frame

def update(_):
    if paused[0]:
        return []  # return nothing if paused

    frame = frame_index[0]
    lower_lim = max(0, frame - 100)

    lorenz_plt_1.set_data(x_sol_1[lower_lim:frame+1], y_sol_1[lower_lim:frame+1])
    lorenz_plt_1.set_3d_properties(z_sol_1[lower_lim:frame+1])

    lorenz_plt_2.set_data(x_sol_2[lower_lim:frame+1], y_sol_2[lower_lim:frame+1])
    lorenz_plt_2.set_3d_properties(z_sol_2[lower_lim:frame+1])

    line_x1.set_data(t_eval[:frame+1], x_sol_1[:frame+1])
    line_x2.set_data(t_eval[:frame+1], x_sol_2[:frame+1])
    line_y1.set_data(t_eval[:frame+1], y_sol_1[:frame+1])
    line_y2.set_data(t_eval[:frame+1], y_sol_2[:frame+1])
    line_z1.set_data(t_eval[:frame+1], z_sol_1[:frame+1])
    line_z2.set_data(t_eval[:frame+1], z_sol_2[:frame+1])

    time_text.set_text(f'Time: {t_eval[frame]:.2f} s')

    frame_index[0] = min(frame + 1, len(t_eval) - 1)

    return (lorenz_plt_1, lorenz_plt_2, line_x1, line_x2, line_y1, line_y2, line_z1, line_z2, time_text)

# --- Run the animation ---
animation = FuncAnimation(fig, update, interval=16, blit=False, cache_frame_data= False)

plt.tight_layout()
plt.show()
