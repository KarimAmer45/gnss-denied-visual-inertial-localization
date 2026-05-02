"""Pillow-based plots for reproducible README screenshots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ekf import YAW, wrap_angle


Palette = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "grid": "#d9e2ec",
    "axis": "#334e68",
    "truth": "#111827",
    "fused": "#0f766e",
    "inertial": "#c2410c",
    "visual": "#7c3aed",
    "gnss": "#2563eb",
    "dropout": "#fee2e2",
    "dropout_line": "#dc2626",
    "bg": "#ffffff",
}


def render_all_plots(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_trajectory(result, output_dir / "trajectory_overview.png")
    render_position_error(result, output_dir / "position_error.png")
    render_sensor_timeline(result, output_dir / "sensor_timeline.png")
    render_yaw_error(result, output_dir / "yaw_error.png")


def render_trajectory(result, output_path: Path) -> None:
    width, height = 1280, 820
    image = Image.new("RGB", (width, height), Palette["bg"])
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    plot = _PlotArea(96, 96, width - 58, height - 110)

    truth = result.truth.pose[:, :2]
    fused = result.fused_state[:, :2]
    inertial = result.inertial_odom_state[:, :2]
    visual = result.measurements.visual_pose[:, :2]
    finite_visual = np.isfinite(visual[:, 0])
    gnss = result.measurements.gnss_position
    finite_gnss = result.measurements.gnss_available

    xy = np.vstack((truth, fused, inertial, visual[finite_visual], gnss[finite_gnss]))
    mapper = plot.mapper(xy[:, 0], xy[:, 1], pad_fraction=0.08)
    _draw_grid(draw, plot, mapper, "x [m]", "y [m]", fonts)

    dropout = result.measurements.dropout_mask
    _draw_polyline(draw, mapper(truth[:, 0], truth[:, 1]), Palette["truth"], 4)
    _draw_polyline(draw, mapper(fused[:, 0], fused[:, 1]), Palette["fused"], 4)
    _draw_polyline(draw, mapper(inertial[:, 0], inertial[:, 1]), Palette["inertial"], 3)
    _draw_polyline(draw, mapper(visual[finite_visual, 0], visual[finite_visual, 1]), Palette["visual"], 2)
    _draw_polyline(draw, mapper(truth[dropout, 0], truth[dropout, 1]), Palette["dropout_line"], 7)

    for px, py in mapper(gnss[finite_gnss, 0], gnss[finite_gnss, 1])[::2]:
        r = 4
        draw.ellipse((px - r, py - r, px + r, py + r), fill=Palette["gnss"])

    _title(draw, "GNSS-denied visual-inertial EKF trajectory", fonts)
    _subtitle(
        draw,
        f"GNSS outage: {result.config.dropout_start_s:.0f}s to {result.config.dropout_end_s():.0f}s. "
        f"Dropout RMSE improves {result.metrics['dropout_improvement_percent']:.1f}% vs inertial/odom only.",
        fonts,
    )
    _legend(
        draw,
        [
            ("Truth", Palette["truth"]),
            ("EKF fused", Palette["fused"]),
            ("IMU + odom only", Palette["inertial"]),
            ("Visual odometry", Palette["visual"]),
            ("GNSS fixes", Palette["gnss"]),
            ("Outage segment", Palette["dropout_line"]),
        ],
        width - 330,
        116,
        fonts,
    )
    image.save(output_path)


def render_position_error(result, output_path: Path) -> None:
    width, height = 1280, 760
    image = Image.new("RGB", (width, height), Palette["bg"])
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    plot = _PlotArea(94, 102, width - 58, height - 112)

    t = result.truth.t
    truth = result.truth.pose[:, :2]
    fused_error = np.linalg.norm(result.fused_state[:, :2] - truth, axis=1)
    inertial_error = np.linalg.norm(result.inertial_odom_state[:, :2] - truth, axis=1)
    visual = result.measurements.visual_pose[:, :2]
    visual_error = np.linalg.norm(visual - truth, axis=1)
    visual_mask = np.isfinite(visual_error)

    max_y = float(np.nanmax([fused_error.max(), inertial_error.max(), np.nanmax(visual_error)]))
    mapper = plot.mapper(t, np.array([0.0, max_y * 1.08]), y_min_override=0.0)
    _shade_dropout(draw, plot, mapper, result)
    _draw_grid(draw, plot, mapper, "time [s]", "position error [m]", fonts)

    _draw_polyline(draw, mapper(t, fused_error), Palette["fused"], 4)
    _draw_polyline(draw, mapper(t, inertial_error), Palette["inertial"], 3)
    _draw_polyline(draw, mapper(t[visual_mask], visual_error[visual_mask]), Palette["visual"], 2)

    _title(draw, "Position error through a simulated GNSS outage", fonts)
    _subtitle(
        draw,
        f"Fused dropout RMSE {result.metrics['dropout_fused_position_rmse_m']:.2f} m; "
        f"IMU/odom-only dropout RMSE {result.metrics['dropout_inertial_odom_position_rmse_m']:.2f} m.",
        fonts,
    )
    _legend(
        draw,
        [
            ("EKF fused", Palette["fused"]),
            ("IMU + odom only", Palette["inertial"]),
            ("Visual odometry", Palette["visual"]),
            ("GNSS unavailable", Palette["dropout_line"]),
        ],
        width - 310,
        122,
        fonts,
    )
    image.save(output_path)


def render_yaw_error(result, output_path: Path) -> None:
    width, height = 1280, 700
    image = Image.new("RGB", (width, height), Palette["bg"])
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    plot = _PlotArea(94, 102, width - 58, height - 112)

    t = result.truth.t
    fused_error = np.abs(wrap_angle(result.fused_state[:, YAW] - result.truth.pose[:, 2])) * 180.0 / np.pi
    inertial_error = np.abs(wrap_angle(result.inertial_odom_state[:, YAW] - result.truth.pose[:, 2])) * 180.0 / np.pi
    max_y = float(max(fused_error.max(), inertial_error.max()) * 1.12)
    mapper = plot.mapper(t, np.array([0.0, max_y]), y_min_override=0.0)
    _shade_dropout(draw, plot, mapper, result)
    _draw_grid(draw, plot, mapper, "time [s]", "yaw error [deg]", fonts)
    _draw_polyline(draw, mapper(t, fused_error), Palette["fused"], 4)
    _draw_polyline(draw, mapper(t, inertial_error), Palette["inertial"], 3)

    _title(draw, "Yaw error from IMU prediction and visual correction", fonts)
    _subtitle(
        draw,
        f"Camera yaw updates hold heading drift during the {result.config.dropout_duration_s:.0f}s GNSS outage.",
        fonts,
    )
    _legend(
        draw,
        [
            ("EKF fused", Palette["fused"]),
            ("IMU + odom only", Palette["inertial"]),
            ("GNSS unavailable", Palette["dropout_line"]),
        ],
        width - 280,
        122,
        fonts,
    )
    image.save(output_path)


def render_sensor_timeline(result, output_path: Path) -> None:
    width, height = 1280, 560
    image = Image.new("RGB", (width, height), Palette["bg"])
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    left, right = 180, width - 70
    top = 150
    row_h = 70
    t = result.truth.t
    t_min, t_max = float(t[0]), float(t[-1])

    def x_at(time_s: float) -> int:
        return int(left + (time_s - t_min) / (t_max - t_min) * (right - left))

    _title(draw, "Sensor availability timeline", fonts)
    _subtitle(draw, "The EKF keeps receiving IMU, odometry, and visual updates while GNSS fixes are withheld.", fonts)

    for i, (name, mask, color) in enumerate(
        [
            ("IMU prediction", np.ones_like(t, dtype=bool), Palette["truth"]),
            ("Wheel odom", result.measurements.odom_available, Palette["inertial"]),
            ("Visual odom", result.measurements.visual_available, Palette["visual"]),
            ("GNSS fixes", result.measurements.gnss_available, Palette["gnss"]),
        ]
    ):
        y = top + i * row_h
        draw.text((52, y - 12), name, fill=Palette["ink"], font=fonts["body"])
        draw.line((left, y, right, y), fill=Palette["grid"], width=8)
        segments = _availability_segments(t, mask)
        for start, end in segments:
            draw.line((x_at(start), y, x_at(end), y), fill=color, width=12)
        for time_s in t[mask][:: max(1, int(mask.sum() / 80))]:
            x = x_at(float(time_s))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    dropout_left = x_at(result.config.dropout_start_s)
    dropout_right = x_at(result.config.dropout_end_s())
    draw.rectangle((dropout_left, top - 42, dropout_right, top + 3 * row_h + 42), outline=Palette["dropout_line"], width=3)
    draw.text((dropout_left + 10, top - 70), "GNSS dropout window", fill=Palette["dropout_line"], font=fonts["small_bold"])

    for tick in np.linspace(0, result.config.duration_s, 7):
        x = x_at(float(tick))
        draw.line((x, top + 3 * row_h + 40, x, top + 3 * row_h + 50), fill=Palette["axis"], width=2)
        _center_text(draw, (x, top + 3 * row_h + 58), f"{tick:.0f}s", fonts["small"], Palette["muted"])

    image.save(output_path)


class _PlotArea:
    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def mapper(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        pad_fraction: float = 0.04,
        y_min_override: float | None = None,
    ):
        x_min = float(np.nanmin(xs))
        x_max = float(np.nanmax(xs))
        y_min = float(np.nanmin(ys)) if y_min_override is None else y_min_override
        y_max = float(np.nanmax(ys))
        x_pad = max((x_max - x_min) * pad_fraction, 1e-6)
        y_pad = max((y_max - y_min) * pad_fraction, 1e-6)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad if y_min_override is None else 0.0
        y_max += y_pad

        def map_points(x_values, y_values):
            x_values = np.asarray(x_values, dtype=float)
            y_values = np.asarray(y_values, dtype=float)
            px = self.left + (x_values - x_min) / (x_max - x_min) * (self.right - self.left)
            py = self.bottom - (y_values - y_min) / (y_max - y_min) * (self.bottom - self.top)
            return list(zip(px.astype(int), py.astype(int)))

        map_points.x_min = x_min
        map_points.x_max = x_max
        map_points.y_min = y_min
        map_points.y_max = y_max
        return map_points


def _draw_grid(draw: ImageDraw.ImageDraw, plot: _PlotArea, mapper, x_label: str, y_label: str, fonts) -> None:
    for frac in np.linspace(0.0, 1.0, 6):
        x = int(plot.left + frac * (plot.right - plot.left))
        y = int(plot.bottom - frac * (plot.bottom - plot.top))
        draw.line((x, plot.top, x, plot.bottom), fill=Palette["grid"], width=1)
        draw.line((plot.left, y, plot.right, y), fill=Palette["grid"], width=1)
        x_value = mapper.x_min + frac * (mapper.x_max - mapper.x_min)
        y_value = mapper.y_min + frac * (mapper.y_max - mapper.y_min)
        _center_text(draw, (x, plot.bottom + 22), _tick_label(x_value), fonts["small"], Palette["muted"])
        draw.text((plot.left - 72, y - 8), _tick_label(y_value), fill=Palette["muted"], font=fonts["small"])
    draw.rectangle((plot.left, plot.top, plot.right, plot.bottom), outline=Palette["axis"], width=2)
    _center_text(draw, ((plot.left + plot.right) // 2, plot.bottom + 56), x_label, fonts["body"], Palette["axis"])
    draw.text((18, (plot.top + plot.bottom) // 2), y_label, fill=Palette["axis"], font=fonts["body"])


def _shade_dropout(draw: ImageDraw.ImageDraw, plot: _PlotArea, mapper, result) -> None:
    points = mapper(
        np.array([result.config.dropout_start_s, result.config.dropout_end_s()]),
        np.array([mapper.y_min, mapper.y_min]),
    )
    left, right = points[0][0], points[1][0]
    draw.rectangle((left, plot.top, right, plot.bottom), fill=Palette["dropout"])
    draw.line((left, plot.top, left, plot.bottom), fill=Palette["dropout_line"], width=2)
    draw.line((right, plot.top, right, plot.bottom), fill=Palette["dropout_line"], width=2)


def _draw_polyline(draw: ImageDraw.ImageDraw, points: Iterable[tuple[int, int]], color: str, width: int) -> None:
    pts = list(points)
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def _legend(draw: ImageDraw.ImageDraw, entries: list[tuple[str, str]], x: int, y: int, fonts) -> None:
    pad = 12
    line_h = 28
    max_width = max(draw.textlength(label, font=fonts["small"]) for label, _ in entries)
    draw.rounded_rectangle(
        (x - pad, y - pad, x + int(max_width) + 64, y + line_h * len(entries) + pad),
        radius=6,
        fill="#f8fafc",
        outline="#cbd5e1",
    )
    for i, (label, color) in enumerate(entries):
        yy = y + i * line_h
        draw.line((x, yy + 10, x + 28, yy + 10), fill=color, width=5)
        draw.text((x + 40, yy), label, fill=Palette["ink"], font=fonts["small"])


def _title(draw: ImageDraw.ImageDraw, text: str, fonts) -> None:
    draw.text((54, 30), text, fill=Palette["ink"], font=fonts["title"])


def _subtitle(draw: ImageDraw.ImageDraw, text: str, fonts) -> None:
    draw.text((56, 68), text, fill=Palette["muted"], font=fonts["body"])


def _availability_segments(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    start: float | None = None
    last = float(t[0])
    for time_s, available in zip(t, mask):
        if available and start is None:
            start = float(time_s)
        if not available and start is not None:
            segments.append((start, last))
            start = None
        last = float(time_s)
    if start is not None:
        segments.append((start, float(t[-1])))
    return segments


def _center_text(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str, font, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((center[0] - width // 2, center[1]), text, fill=fill, font=font)


def _tick_label(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fonts() -> dict[str, ImageFont.ImageFont]:
    return {
        "title": _load_font(28, bold=True),
        "body": _load_font(17),
        "small": _load_font(14),
        "small_bold": _load_font(14, bold=True),
    }


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()

