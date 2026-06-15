"""Renderer for fire rescue demos."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

GridPos = Tuple[int, int]


class FireRescueRenderer:
    """Render frames with a city-disaster visual style.

    The visual language follows the reference project's map-first rendering:
    discrete terrain colors, translucent smoke overlay, thin paths, and distinct
    UAV/UGV markers. Only presentation style is reused, not algorithms.
    """

    COLORS: Dict[str, str] = {
        "empty": "#f8fafc",
        "building": "#30343b",
        "road": "#d4d7dc",
        "debris": "#8a6f4d",
        "fire": "#e53935",
        "smoke": "#8d8aa6",
        "water": "#1e88e5",
        "survivor": "#ffd54f",
        "base": "#43a047",
        "uav": "#1565c0",
        "ugv": "#2e7d32",
        "link": "#64b5f6",
        "text": "#111827",
        "panel": "#ffffff",
    }

    def __init__(self, cell_inches: float = 0.42) -> None:
        self.cell_inches = cell_inches

    def render(self, state: Dict) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        layout = state["layout"]
        height = len(layout)
        width = len(layout[0])
        fig_w = max(7.4, width * self.cell_inches + 1.8)
        fig_h = max(7.1, height * self.cell_inches + 2.0)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=140, facecolor="#f4f6f8")
        gs = fig.add_gridspec(3, 1, height_ratios=[0.72, height, 0.55], hspace=0.05)
        status_ax = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
        legend_ax = fig.add_subplot(gs[2])

        terrain = self._terrain_grid(layout, state)
        cmap = mcolors.ListedColormap(
            [
                self.COLORS["empty"],
                self.COLORS["building"],
                self.COLORS["road"],
                self.COLORS["debris"],
                self.COLORS["fire"],
                self.COLORS["smoke"],
                self.COLORS["water"],
                self.COLORS["survivor"],
                self.COLORS["base"],
            ]
        )
        norm = mcolors.BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
        ax.imshow(terrain, cmap=cmap, norm=norm, interpolation="nearest")

        smoke_alpha = np.zeros((height, width), dtype=float)
        for row, col in state["smoke_cells"]:
            if 0 <= row < height and 0 <= col < width:
                smoke_alpha[row, col] = 0.34
        if smoke_alpha.any():
            ax.imshow(smoke_alpha, cmap="Purples", alpha=smoke_alpha, interpolation="nearest", vmin=0.0, vmax=0.60)

        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.set_xticks(np.arange(-0.5, width, 5))
        ax.set_yticks(np.arange(-0.5, height, 5))
        ax.grid(which="minor", color="#ffffff", linewidth=0.28, alpha=0.38)
        ax.grid(which="major", color="#ffffff", linewidth=0.62, alpha=0.72)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(self.COLORS["empty"])

        uav_xy, ugv_xy = self._agent_display_xy(state["uav_pos"], state["ugv_pos"])
        self._draw_uav_vision(ax, uav_xy, state.get("view_radius", 3), width, height)
        self._draw_path(ax, state.get("uav_path", []), self.COLORS["uav"], linewidth=1.35)
        self._draw_path(ax, state.get("ugv_path", []), self.COLORS["ugv"], linewidth=1.45)
        self._draw_communication(ax, uav_xy, ugv_xy)
        self._draw_survivor(ax, state)
        self._draw_base(ax, state["base_pos"])
        self._draw_uav(ax, uav_xy)
        self._draw_ugv(ax, ugv_xy, state.get("ugv_path", []), carrying=state["survivor_picked"])
        self._draw_status_panel(status_ax, state)
        self._draw_legend(legend_ax)

        fig.subplots_adjust(left=0.035, right=0.965, top=0.965, bottom=0.035)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    def _terrain_grid(self, layout, state: Dict) -> np.ndarray:
        grid = np.zeros((len(layout), len(layout[0])), dtype=np.int16)
        for r, row in enumerate(layout):
            for c, char in enumerate(row):
                if char == "#":
                    grid[r, c] = 1
                elif char in {".", "B", "S", "F"}:
                    grid[r, c] = 2
                elif char == "X":
                    grid[r, c] = 3
                elif char == "W":
                    grid[r, c] = 6
        for r, c in state["smoke_cells"]:
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] not in {1, 3}:
                grid[r, c] = 5
        for r, c in state["fire_cells"]:
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                grid[r, c] = 4
        br, bc = state["base_pos"]
        grid[br, bc] = 8
        sr, sc = state["survivor_pos"]
        if not state["survivor_picked"]:
            grid[sr, sc] = 7
        return grid

    def _draw_status_panel(self, ax, state: Dict) -> None:
        from matplotlib.patches import FancyBboxPatch

        ax.set_axis_off()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        panel = FancyBboxPatch(
            (0.012, 0.12),
            0.976,
            0.76,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            linewidth=0.0,
            facecolor=self.COLORS["panel"],
            alpha=0.98,
        )
        ax.add_patch(panel)
        stage = str(state["stage"])
        stage_color = {
            "search survivor": self.COLORS["uav"],
            "ground rescue": "#c47f14",
            "return to base": self.COLORS["ugv"],
        }.get(stage, self.COLORS["text"])
        ax.text(0.035, 0.61, "Dynamic Fire Rescue", fontsize=10.4, weight="bold", color=self.COLORS["text"])
        ax.text(0.035, 0.35, "UAV search + UGV evacuation", fontsize=7.4, color="#4b5563")

        stats = [
            (0.34, "Step", f"{state['step_count']:03d}", "#111827"),
            (0.45, "Stage", stage, stage_color),
            (0.66, "Reward", f"{state['episode_reward']:.1f}", "#0f766e"),
            (0.77, "Risk", f"{state['risk_exposure']:.1f}", "#b91c1c"),
        ]
        for x, label, value, color in stats:
            ax.text(x, 0.65, label.upper(), fontsize=5.9, color="#6b7280", weight="bold")
            ax.text(x, 0.34, value, fontsize=8.4, color=color, weight="bold")

        badges = [
            ("found", state["survivor_discovered"], self.COLORS["uav"], 0.875),
            ("picked", state["survivor_picked"], self.COLORS["ugv"], 0.950),
        ]
        for label, active, color, x_center in badges:
            face = color if active else "#e5e7eb"
            text_color = "white" if active else "#374151"
            badge = FancyBboxPatch(
                (x_center - 0.034, 0.30),
                0.068,
                0.36,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=0.0,
                facecolor=face,
            )
            ax.add_patch(badge)
            ax.text(x_center, 0.48, label, ha="center", va="center", fontsize=6.1, color=text_color, weight="bold")

    def _draw_legend(self, ax) -> None:
        import matplotlib.pyplot as plt

        ax.set_axis_off()
        items = [
            ("road", self.COLORS["road"]),
            ("debris", self.COLORS["debris"]),
            ("fire", self.COLORS["fire"]),
            ("smoke", self.COLORS["smoke"]),
            ("base", self.COLORS["base"]),
            ("survivor", self.COLORS["survivor"]),
        ]
        handles = [
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=7, label=label)
            for label, color in items
        ]
        handles.extend(
            [
                plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=self.COLORS["uav"], markeredgecolor="white", markersize=8, label="UAV"),
                plt.Line2D([0], [0], color=self.COLORS["ugv"], linewidth=1.6, label="UGV path"),
                plt.Line2D([0], [0], color=self.COLORS["link"], linewidth=1.0, linestyle="--", label="link"),
            ]
        )
        ax.legend(
            handles=handles,
            loc="center",
            ncol=9,
            frameon=False,
            fontsize=7.0,
            handletextpad=0.25,
            columnspacing=0.65,
        )

    def _draw_base(self, ax, pos: GridPos) -> None:
        row, col = pos
        ax.scatter(col, row, marker="s", s=155, color=self.COLORS["base"], edgecolors="white", linewidths=1.0, zorder=7)
        ax.text(col, row + 0.02, "B", ha="center", va="center", fontsize=8.0, color="white", weight="bold", zorder=8)

    def _draw_survivor(self, ax, state: Dict) -> None:
        if state["survivor_picked"]:
            return
        row, col = state["survivor_pos"]
        label = "S" if state["survivor_discovered"] else "?"
        ax.scatter(col, row, marker="*", s=225, color=self.COLORS["survivor"], edgecolors="#111827", linewidths=0.75, zorder=7)
        ax.text(col, row + 0.03, label, ha="center", va="center", fontsize=8.0, color="#111827", weight="bold", zorder=8)

    def _draw_uav(self, ax, xy: tuple[float, float]) -> None:
        x, y = xy
        ax.scatter(x, y, marker="^", s=165, color=self.COLORS["uav"], edgecolors="white", linewidths=1.0, zorder=10)
        ax.text(
            x + 0.25,
            y - 0.24,
            "UAV",
            fontsize=6.2,
            color=self.COLORS["text"],
            weight="bold",
            zorder=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.35},
        )

    def _draw_ugv(self, ax, xy: tuple[float, float], path, carrying: bool) -> None:
        from matplotlib.patches import Circle, Polygon

        x, y = xy
        body_color = "#1b5e20" if carrying else self.COLORS["ugv"]
        forward = self._last_motion_unit(path)
        side = np.array([-forward[1], forward[0]], dtype=float)
        center = np.array([x, y], dtype=float)
        length = 0.78
        width = 0.46
        corners = [
            center + forward * length / 2 + side * width / 2,
            center + forward * length / 2 - side * width / 2,
            center - forward * length / 2 - side * width / 2,
            center - forward * length / 2 + side * width / 2,
        ]
        ax.add_patch(Polygon(corners, closed=True, facecolor=body_color, edgecolor="white", linewidth=1.0, zorder=10))

        cabin_center = center + forward * 0.08
        cabin_len = 0.34
        cabin_w = 0.28
        cabin_color = "#facc15" if carrying else "#86efac"
        cabin = [
            cabin_center + forward * cabin_len / 2 + side * cabin_w / 2,
            cabin_center + forward * cabin_len / 2 - side * cabin_w / 2,
            cabin_center - forward * cabin_len / 2 - side * cabin_w / 2,
            cabin_center - forward * cabin_len / 2 + side * cabin_w / 2,
        ]
        ax.add_patch(Polygon(cabin, closed=True, facecolor=cabin_color, edgecolor="none", alpha=0.95, zorder=11))

        wheel_color = "#111827"
        for front_back in (-0.24, 0.24):
            for left_right in (-0.29, 0.29):
                wheel = center + forward * front_back + side * left_right
                ax.add_patch(Circle(wheel, radius=0.065, facecolor=wheel_color, edgecolor="none", zorder=12))
        for left_right in (-0.12, 0.12):
            light = center + forward * 0.42 + side * left_right
            ax.add_patch(Circle(light, radius=0.035, facecolor="#fef3c7", edgecolor="none", zorder=12))
        ax.text(
            x + 0.31,
            y - 0.30,
            "UGV",
            fontsize=6.2,
            color=self.COLORS["text"],
            weight="bold",
            zorder=13,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.35},
        )

    def _draw_communication(self, ax, uav_xy: tuple[float, float], ugv_xy: tuple[float, float]) -> None:
        ax.plot(
            [uav_xy[0], ugv_xy[0]],
            [uav_xy[1], ugv_xy[1]],
            color=self.COLORS["link"],
            linewidth=1.0,
            alpha=0.66,
            linestyle="--",
            zorder=5,
        )

    def _draw_path(self, ax, path, color: str, linewidth: float) -> None:
        if len(path) < 2:
            return
        xs = [col for _, col in path]
        ys = [row for row, _ in path]
        ax.plot(xs, ys, color=color, linewidth=linewidth + 1.0, alpha=0.18, zorder=5)
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.78, zorder=6)

    def _draw_uav_vision(
        self,
        ax,
        uav_xy: tuple[float, float],
        radius: int,
        width: int,
        height: int,
    ) -> None:
        from matplotlib.patches import Rectangle

        radius = max(1, int(radius))
        x, y = uav_xy
        left = max(-0.5, x - radius - 0.5)
        top = max(-0.5, y - radius - 0.5)
        right = min(width - 0.5, x + radius + 0.5)
        bottom = min(height - 0.5, y + radius + 0.5)
        ax.add_patch(
            Rectangle(
                (left, top),
                right - left,
                bottom - top,
                facecolor=self.COLORS["uav"],
                edgecolor=self.COLORS["uav"],
                linewidth=0.8,
                linestyle=":",
                alpha=0.075,
                zorder=4,
            )
        )

    def _agent_display_xy(self, uav_pos: GridPos, ugv_pos: GridPos) -> tuple[tuple[float, float], tuple[float, float]]:
        uav_xy = (float(uav_pos[1]), float(uav_pos[0]))
        ugv_xy = (float(ugv_pos[1]), float(ugv_pos[0]))
        if uav_pos == ugv_pos:
            return (uav_xy[0] + 0.20, uav_xy[1] - 0.18), (ugv_xy[0] - 0.20, ugv_xy[1] + 0.18)
        return uav_xy, ugv_xy

    def _last_motion_unit(self, path) -> np.ndarray:
        for prev, current in zip(reversed(path[:-1]), reversed(path[1:])):
            dr = current[0] - prev[0]
            dc = current[1] - prev[1]
            if dr or dc:
                vec = np.array([float(dc), float(dr)], dtype=float)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    return vec / norm
        return np.array([1.0, 0.0], dtype=float)
