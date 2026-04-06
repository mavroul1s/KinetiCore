import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading

# Clamp helper
def clamp(value):
    return max(0, min(255, int(value)))

# Dithering algorithms

def floyd_steinberg(img, threshold, color=False):
    mode = "RGB" if color else "L"
    img = img.convert(mode)
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            old = pixels[x, y]
            if color:
                new = tuple(255 if c > threshold else 0 for c in old)
                err = tuple(old[i] - new[i] for i in range(3))
            else:
                new = 255 if old > threshold else 0
                err = old - new
            pixels[x, y] = new
            for dx, dy, f in ((1,0,7/16),(-1,1,3/16),(0,1,5/16),(1,1,1/16)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if color:
                        r, g, b = pixels[nx, ny]
                        pixels[nx, ny] = (
                            clamp(r + err[0] * f),
                            clamp(g + err[1] * f),
                            clamp(b + err[2] * f)
                        )
                    else:
                        pixels[nx, ny] = clamp(pixels[nx, ny] + err * f)
    return img.convert("RGB")

# Example of Jarvis-Judice-Ninke (JJN)
def jarvis_judice_ninke(img, threshold):
    img = img.convert("L")
    pixels = img.load()
    w, h = img.size
    kernel = [
        (1, 0, 7/48), (2, 0, 5/48),
        (-2, 1, 3/48), (-1, 1, 5/48), (0, 1, 7/48), (1, 1, 5/48), (2, 1, 3/48),
        (-2, 2, 1/48), (-1, 2, 3/48), (0, 2, 5/48), (1, 2, 3/48), (2, 2, 1/48)
    ]
    for y in range(h):
        for x in range(w):
            old = pixels[x, y]
            new = 255 if old > threshold else 0
            err = old - new
            pixels[x, y] = new
            for dx, dy, f in kernel:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    pixels[nx, ny] = clamp(pixels[nx, ny] + err * f)
    return img.convert("RGB")

# Atkinson dithering

def atkinson(img, threshold):
    img = img.convert("L")
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            old = pixels[x, y]
            new = 255 if old > threshold else 0
            err = old - new
            pixels[x, y] = new
            for dx, dy in ((1,0),(2,0),( -1,1),(0,1),(1,1),(0,2)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    pixels[nx, ny] = clamp(pixels[nx, ny] + err * 1/8)
    return img.convert("RGB")

# GUI
class DitherApp:
    def __init__(self, root):
        self.root = root
        root.title("Modern Dither Tool")
        root.geometry("1000x700")
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Segoe UI',10))
        style.configure('TButton', background='#3c3f41', foreground='white', font=('Segoe UI',10))
        style.map('TButton', background=[('active','#5c5f61')])

        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)
        self.setup_controls(ctrl)

        self.canvas = tk.Canvas(main, bg='#1e1e1e', bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e:self.refresh())

        self.image_path = None
        self.original = None
        self.preview_img = None
        self.zoom_f = 1.0

    def setup_controls(self, frame):
        ttk.Button(frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)

        ttk.Label(frame, text="Algorithm:").pack(pady=(15,2))
        self.alg = ttk.Combobox(frame, values=["Floyd-Steinberg","JJN","Atkinson"], state='readonly')
        self.alg.current(0); self.alg.pack(fill=tk.X)

        ttk.Label(frame, text="Threshold:").pack(pady=(15,2))
        self.thresh = tk.Scale(frame, from_=0, to=255, orient=tk.HORIZONTAL, bg='#2b2b2b', fg='white', troughcolor='#555', highlightthickness=0)
        self.thresh.set(127); self.thresh.pack(fill=tk.X)

        ttk.Button(frame, text="Preview", command=self.start_preview).pack(fill=tk.X, pady=10)
        ttk.Button(frame, text="Zoom +", command=lambda:self.zoom(1.2)).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Zoom -", command=lambda:self.zoom(0.8)).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Save", command=self.save_full).pack(fill=tk.X, pady=10)

        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.bmp *.gif")])
        if not path: return
        self.image_path = path
        self.original = Image.open(path)
        self.preview_img = None
        self.zoom_f = 1.0
        self.show(self.original)

    def start_preview(self):
        if not self.original:
            messagebox.showwarning("Warning","Load an image first.")
            return
        self.progress.start(8)
        self.preview_img = None
        def task():
            img = Image.open(self.image_path)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            img.thumbnail((cw, ch), Image.LANCZOS)
            thr = self.thresh.get()
            if self.alg.get() == "Floyd-Steinberg":
                result = floyd_steinberg(img, thr, color=(self.alg.get()=='Atkinson'))
            elif self.alg.get() == "JJN":
                result = jarvis_judice_ninke(img, thr)
            else:
                result = atkinson(img, thr)
            self.preview_img = result
            self.root.after(0, self.on_preview_done)
        threading.Thread(target=task, daemon=True).start()

    def on_preview_done(self):
        self.progress.stop()
        self.zoom_f = 1.0
        self.show(self.preview_img)

    def zoom(self, f):
        if not (self.preview_img or self.original): return
        self.zoom_f *= f
        self.show(self.preview_img or self.original)

    def save_full(self):
        if not self.image_path: return
        img = Image.open(self.image_path)
        thr = self.thresh.get()
        if self.alg.get() == "Floyd-Steinberg":
            full = floyd_steinberg(img, thr, color=(self.alg.get()=='Atkinson'))
        elif self.alg.get() == "JJN":
            full = jarvis_judice_ninke(img, thr)
        else:
            full = atkinson(img, thr)
        path = filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png")])
        if path: full.save(path)

    def show(self, img):
        self.canvas.delete('all'); self.canvas.update_idletasks()
        w, h = int(img.width*self.zoom_f), int(img.height*self.zoom_f)
        disp = img.resize((w, h), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(disp)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_image(cw//2, ch//2, image=self.tkimg, anchor=tk.CENTER)

    def refresh(self):
        self.show(self.preview_img or self.original)

if __name__ == "__main__":
    root = tk.Tk()
    app = DitherApp(root)
    root.mainloop()
