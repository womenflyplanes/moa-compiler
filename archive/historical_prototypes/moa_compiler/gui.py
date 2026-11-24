import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

root = tk.Tk()
root.title("MoA Universal IR — Interactive GUI")
root.geometry("900x700")

d_var = tk.IntVar(value=12)
k_var = tk.IntVar(value=64)
n_var = tk.IntVar(value=1024)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

info = tk.Label(root, text="", font=("Courier", 10), justify=tk.LEFT)
info.pack()

def update(*args):
    d = d_var.get(); k = k_var.get(); n = n_var.get()
    F = d * k * n; M = d * k
    devices = {'Delta A100':0.004,'MI100':0.028,'Max1550':0.004,'V100':0.004,'H100':0.028,'UA A40':0.006}
    ax1.clear(); ax2.clear()
    for name, alpha in devices.items():
        beta = 0.028 if alpha>0.01 else 0.004
        d_range = np.arange(4,37)
        T_range = alpha*(d_range*k*n)/1e6 + beta*(d_range*k)/1e3
        ax1.plot(d_range, T_range, label=f"{name}")
    ax1.axhline(y=36, color='g', linestyle='--', label='M1exp:36 reached(d) ✓')
    ax1.set_xlabel('d depth'); ax1.set_ylabel('T ms T≈αF+βM')
    ax1.set_title(f'MoA: d={d} k={k} n={n} Same DNF Many ONFs')
    ax1.legend(fontsize=6, loc='upper left'); ax1.grid(True, alpha=0.3)
    T = 0.004*F/1e6+0.004*M/1e3
    bytes_cost=F*4; dollar_cost=T*0.0001; energy=T*0.3; heat=energy*3.6
    costs=[F/1e6, bytes_cost/1e6, T, dollar_cost*1000, energy, heat]
    labels=['FLOPs M','Bytes MB','Time ms','$ m$','Energy Wh','Heat kJ']
    ax2.bar(labels, costs); ax2.set_title('Live: $, bytes, times, energy, heat')
    ax2.tick_params(axis='x', rotation=15)
    reached="✓ reached(d)" if d>=9 else "✗ Xd>0"
    info.config(text=f"DNF FIXED ivec ψ A ≡ (rav A)[γ(ivec,ρA)] | d={d} k={k} n={n} F={F:,} {reached}\nCIS261342 Delta A100/MI100 Stampede3 Max1550 Bridges-2 V100/H100 + CHI-261724 + UA A40\n2.00x→2.5x 535x NUMA vs <3x 3.17x/3.35x C vs Fortran")
    canvas.draw()

frame = tk.Frame(root); frame.pack(fill=tk.X)
tk.Label(frame, text="d:").pack(side=tk.LEFT)
tk.Scale(frame, from_=4, to=36, orient=tk.HORIZONTAL, variable=d_var, command=update, length=150).pack(side=tk.LEFT)
tk.Label(frame, text="k:").pack(side=tk.LEFT)
tk.Scale(frame, from_=16, to=256, orient=tk.HORIZONTAL, variable=k_var, command=update, length=150).pack(side=tk.LEFT)
tk.Label(frame, text="n:").pack(side=tk.LEFT)
tk.Scale(frame, from_=128, to=4096, orient=tk.HORIZONTAL, variable=n_var, command=update, length=150).pack(side=tk.LEFT)
update()
root.mainloop()
