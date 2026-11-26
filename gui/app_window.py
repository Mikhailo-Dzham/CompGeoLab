import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.optimizer import find_min_area_star_polygon
from utils.point_generator import generate_uniform, generate_normal, generate_ring
from utils.benchmark import run_benchmarks


class StarPolygonApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Генерація зіркових многокутників — Задача №37 (ОМЗМ Г)")
        self.root.geometry("1100x720")

        self.points: list = []
        self.variants: list = []
        self.variant_idx: int = 0

        self._build_ui()
        self._redraw_canvas()

    def _build_ui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Кількість точок (N):").pack(anchor=tk.W)
        self.n_var = tk.StringVar(value="15")
        ttk.Entry(control_frame, textvariable=self.n_var, width=18).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(control_frame, text="Розподіл:").pack(anchor=tk.W)
        self.dist_var = tk.StringVar(value="Рівномірний")
        ttk.Combobox(
            control_frame,
            textvariable=self.dist_var,
            values=["Рівномірний", "Нормальний", "Кільце"],
            state="readonly",
            width=16,
        ).pack(fill=tk.X, pady=(2, 10))

        ttk.Button(control_frame, text="Згенерувати точки", command=self._on_generate).pack(fill=tk.X, pady=3)
        ttk.Button(control_frame, text="Знайти мінімальний", command=self._on_solve).pack(fill=tk.X, pady=3)
        ttk.Button(control_frame, text="Очистити", command=self._on_clear).pack(fill=tk.X, pady=3)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(control_frame, text="Перегляд варіантів (N ≤ 20):").pack(anchor=tk.W)
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=4)
        ttk.Button(nav_frame, text="← Попер.", command=self._on_prev, width=9).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(nav_frame, text="Наст. →", command=self._on_next, width=9).pack(side=tk.RIGHT)

        self.variant_label = ttk.Label(control_frame, text="Варіант: - / -")
        self.variant_label.pack(anchor=tk.W, pady=(2, 10))

        ttk.Button(control_frame, text="Графік складності T(N)", command=self._on_benchmark).pack(fill=tk.X, pady=4)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        self.info_var = tk.StringVar(value="Клікніть по полю для додавання точок\nабо згенеруйте множину.")
        ttk.Label(control_frame, textvariable=self.info_var, wraplength=210, justify=tk.LEFT).pack(anchor=tk.W)

        plot_frame = ttk.Frame(self.root, padding=5)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _on_canvas_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if len(self.points) >= 100:
            messagebox.showinfo("Обмеження", "Для ручного введення рекомендовано до 100 точок. Для більших N використайте генератор.")
            return
        self.points.append([event.xdata, event.ydata])
        self.variants = []
        self.variant_idx = 0
        self.info_var.set(f"Точок: {len(self.points)}\nНатисніть «Знайти мінімальний».")
        self._redraw_canvas()

    def _on_generate(self):
        try:
            n = int(self.n_var.get())
            if n < 3 or n > 10000:
                raise ValueError
        except ValueError:
            messagebox.showerror("Помилка", "Введіть ціле число N від 3 до 10000.")
            return

        dist = self.dist_var.get()
        if dist == "Рівномірний":
            pts = generate_uniform(n)
        elif dist == "Нормальний":
            pts = generate_normal(n)
        else:
            pts = generate_ring(n)

        self.points = pts.tolist()
        self.variants = []
        self.variant_idx = 0
        self.variant_label.set("Варіант: - / -") if hasattr(self.variant_label, "set") else self.variant_label.config(text="Варіант: - / -")
        self.info_var.set(f"Згенеровано {n} точок.\nНатисніть «Знайти мінімальний».")
        self._redraw_canvas()

    def _on_solve(self):
        if len(self.points) < 3:
            messagebox.showwarning("Недостатньо точок", "Додайте щонайменше 3 точки.")
            return

        pts = np.array(self.points, dtype=float)
        res = find_min_area_star_polygon(pts)
        if res.best_polygon is None:
            messagebox.showerror("Помилка", "Не вдалося побудувати зірковий многокутник (усі точки колінеарні).")
            return

        self.variants = res.variants
        self.variant_idx = 0
        self._update_variant_view(res.checked_count, res.elapsed_ms)

    def _update_variant_view(self, checked: int = None, elapsed_ms: float = None):
        poly, area, z = self.variants[self.variant_idx]
        self.variant_label.config(text=f"Варіант: {self.variant_idx + 1} / {len(self.variants)}")
        extra = ""
        if checked is not None and elapsed_ms is not None:
            extra = f"\nПеревірено конфігурацій: {checked}\nЧас пошуку: {elapsed_ms:.2f} мс"
        self.info_var.set(
            f"Кількість точок N: {len(self.points)}\n"
            f"Площа S: {area:.2f}\n"
            f"Ядро z: ({z[0]:.2f}, {z[1]:.2f})"
            f"{extra}"
        )
        self._redraw_canvas(poly, z)

    def _on_prev(self):
        if not self.variants:
            return
        self.variant_idx = (self.variant_idx - 1) % len(self.variants)
        self._update_variant_view()

    def _on_next(self):
        if not self.variants:
            return
        self.variant_idx = (self.variant_idx + 1) % len(self.variants)
        self._update_variant_view()

    def _on_clear(self):
        self.points = []
        self.variants = []
        self.variant_idx = 0
        self.variant_label.config(text="Варіант: - / -")
        self.info_var.set("Полотно очищено.")
        self._redraw_canvas()

    def _on_benchmark(self):
        data = run_benchmarks()
        ns = [row[0] for row in data]
        t_single = [row[1] for row in data]
        t_opt = [row[2] for row in data]

        win = tk.Toplevel(self.root)
        win.title("Графік часової складності T(N)")
        fig = Figure(figsize=(6, 4.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(ns, t_single, marker="o", label="1 побудова O(N log N)")
        ax.plot(ns, t_opt, marker="s", label="Пошук мінімальної площі")
        ax.set_xlabel("Кількість точок N")
        ax.set_ylabel("Час виконання, мс")
        ax.set_title("Експериментальна залежність T(N)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def _redraw_canvas(self, polygon: np.ndarray = None, z: np.ndarray = None):
        self.ax.clear()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", alpha=0.5)
        self.ax.set_title("Робоче поле (клік ЛКМ додає точку)")

        if polygon is not None:
            closed = np.vstack([polygon, polygon[0]])
            lw = 0.6 if len(polygon) > 500 else 1.5
            self.ax.fill(closed[:, 0], closed[:, 1], color="#4a90e2", alpha=0.22)
            self.ax.plot(closed[:, 0], closed[:, 1], color="#1f4e79", linewidth=lw)

        if self.points:
            pts = np.array(self.points)
            ms = 2 if len(pts) > 500 else 18
            self.ax.scatter(pts[:, 0], pts[:, 1], color="#222222", s=ms, zorder=3)

        if z is not None:
            self.ax.scatter([z[0]], [z[1]], color="#d9534f", s=55, marker="X", zorder=4, label="Точка ядра z")
            self.ax.legend(loc="upper right")

        self.canvas.draw()