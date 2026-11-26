import tkinter as tk
from gui.app_window import StarPolygonApp


def main():
    root = tk.Tk()
    StarPolygonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()