import fastf1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache("/Users/aakashbaskar/.cache/fastf1")

# ── Load session ──────────────────────────────────────────────
print("Loading session...")
session = fastf1.get_testing_session(2026, 1, 1)
session.load()

russell_laps = session.laps.pick_drivers("63")
fastest      = russell_laps.pick_fastest()
tel          = fastest.get_telemetry().reset_index(drop=True)

lap_time = str(fastest["LapTime"]).split()[-1][:11]
compound = fastest["Compound"]
lap_num  = int(fastest["LapNumber"])
print(f"Lap {lap_num}  |  {lap_time}  |  {compound}  |  {len(tel)} telemetry points")

# ── Compute cumulative distance for x-axis ────────────────────
tel["dt"]       = tel["Time"].dt.total_seconds().diff().fillna(0)
tel["distance"] = (tel["Speed"] / 3.6 * tel["dt"]).cumsum()

x  = tel["X"].values
y  = tel["Y"].values
N  = len(tel)

# ── Style ─────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1a1d27"
SILVER   = "#c0c4cc"
MERCEDES = "#27F4D2"
STEP     = 3   # animate every Nth point — lower = smoother but slower to save

# ── Figure layout ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle(
    f"GEORGE RUSSELL  |  BAHRAIN PRE-SEASON TEST 2026  |  LAP {lap_num}  |  {lap_time}  |  {compound}",
    color="white", fontsize=12, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(
    4, 2,
    figure=fig,
    left=0.06, right=0.97,
    top=0.93,  bottom=0.07,
    wspace=0.35, hspace=0.55
)

ax_map      = fig.add_subplot(gs[:, 0])
ax_speed    = fig.add_subplot(gs[0, 1])
ax_throttle = fig.add_subplot(gs[1, 1])
ax_brake    = fig.add_subplot(gs[2, 1])
ax_gear     = fig.add_subplot(gs[3, 1])

telemetry_axes = [
    (ax_speed,    "Speed",    "Speed (km/h)", "#8844ff",  MERCEDES),
    (ax_throttle, "Throttle", "Throttle (%)", "#00cc44",  "#00cc44"),
    (ax_brake,    "Brake",    "Brake",        "#ff4444",  "#ff4444"),
    (ax_gear,     "nGear",    "Gear",         "#aa88ff",  "#aa88ff"),
]

dist  = tel["distance"].values
x_max = dist[-1]

# ── Static track base ─────────────────────────────────────────
ax_map.set_facecolor(PANEL)
ax_map.set_aspect("equal")
ax_map.axis("off")
ax_map.set_title("TRACK MAP", color="white", fontsize=10,
                 fontweight="bold", pad=6)

# Speed-coloured track outline (dim background)
points   = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm     = Normalize(vmin=tel["Speed"].min(), vmax=tel["Speed"].max())
lc       = LineCollection(segments, cmap="plasma", norm=norm,
                          linewidth=4, zorder=1, capstyle="round", alpha=0.4)
lc.set_array(tel["Speed"].values[:-1])
ax_map.add_collection(lc)
ax_map.autoscale()

# S/F marker
ax_map.scatter(x[0], y[0], color="white", s=100, zorder=5, marker="D")
ax_map.annotate("S/F", (x[0], y[0]), color="white", fontsize=7,
                xytext=(6, 6), textcoords="offset points")

# Colorbar
sm   = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
cbar = fig.colorbar(sm, ax=ax_map, fraction=0.025, pad=0.02)
cbar.set_label("Speed (km/h)", color=SILVER, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=SILVER, labelcolor=SILVER)

# Moving dot + trail
pointer, = ax_map.plot([], [], "o", color="white", markersize=10,
                        zorder=10, markeredgecolor=MERCEDES, markeredgewidth=2)
trail,   = ax_map.plot([], [], "-", color=MERCEDES, linewidth=2,
                        zorder=9, alpha=0.8)

# ── Telemetry panels ──────────────────────────────────────────
tel_lines  = []
tel_vlines = []

for ax, col, ylabel, color, lcolor in telemetry_axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SILVER, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.set_xlim(0, x_max)
    ax.set_ylim(tel[col].min() - 1, tel[col].max() + 1)
    ax.set_ylabel(ylabel, color=SILVER, fontsize=8)
    ax.set_xlabel("Distance (m)", color=SILVER, fontsize=7)
    ax.grid(color="#2a2d3a", linewidth=0.5, linestyle="--")

    # dim full-lap ghost trace
    ax.plot(dist, tel[col].values, color=color, linewidth=1, alpha=0.2)

    # live trace
    line, = ax.plot([], [], color=lcolor, linewidth=1.8, zorder=3)
    tel_lines.append((ax, col, line))

    # playhead
    vl = ax.axvline(0, color="white", linewidth=1,
                    linestyle="--", alpha=0.7, zorder=4)
    tel_vlines.append(vl)

# Live readout text
progress_text = fig.text(
    0.5, 0.005, "0%",
    ha="center", color=MERCEDES, fontsize=10, fontweight="bold"
)

# ── Animation ─────────────────────────────────────────────────
frames = list(range(1, N, STEP)) + [N - 1]

def animate(i):
    # pointer on map
    pointer.set_data([x[i]], [y[i]])

    # trail (last 80 points)
    t0 = max(0, i - 80)
    trail.set_data(x[t0:i+1], y[t0:i+1])

    # telemetry traces + vlines
    for (ax, col, line), vl in zip(tel_lines, tel_vlines):
        line.set_data(dist[:i+1], tel[col].values[:i+1])
        vl.set_xdata([dist[i]])

    # live readout
    pct  = int(i / (N - 1) * 100)
    spd  = tel["Speed"].values[i]
    gear = int(tel["nGear"].values[i])
    thr  = tel["Throttle"].values[i]
    progress_text.set_text(
        f"Progress: {pct}%   |   {spd:.0f} km/h   |   Gear {gear}   |   Throttle {thr:.0f}%"
    )

    return [pointer, trail, progress_text] + \
           [line for _, _, line in tel_lines] + tel_vlines

print("Building animation...")
anim = FuncAnimation(
    fig, animate,
    frames=frames,
    interval=30,
    blit=True
)

# ── Save as MP4 ───────────────────────────────────────────────
print("Saving to russell_animated.mp4 (takes ~30s)...")
anim.save(
    "russell_animated.mp4",
    fps=30,
    dpi=120,
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
)
print("✅ Done! Open russell_animated.mp4")
plt.show()