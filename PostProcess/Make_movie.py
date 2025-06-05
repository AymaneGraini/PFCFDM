import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotxdmf import *

file = "./out/benchmark/PFmixedtest_squaretohex.h5"

def create_animation_directly(output_file='animation.mp4', num_frames=200, fps=10, dpi=100):
    fig, ax = plt.subplots(figsize=(4, 4))
    X, Y, psi, t = get_im_data(file, "Psi", 0)
    contour = ax.contourf(X, Y, psi, cmap="seismic", levels=200)
    title = ax.set_title(f"Time {t}")
    
    def update(frame):
        ax.clear()
        X, Y, psi, t = get_im_data(file, "Psi", frame)
        contour = ax.contourf(X, Y, psi, cmap="seismic", levels=200)
        ax.set_title(f"Time {np.round(t,2)}")
        return contour.collections

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False, repeat=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_file, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to {output_file}")

create_animation_directly('animation_method2.mp4', num_frames=200)
