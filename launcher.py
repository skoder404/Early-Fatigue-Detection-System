import os
import subprocess
import tkinter as tk
from tkinter import messagebox

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VENV_PYTHON = r"venv\Scripts\python.exe"

HYBRID_MAIN = os.path.join(BASE_DIR, "hybrid_model", "main.py")
THRESHOLD_MAIN = os.path.join(BASE_DIR, "threshold_model", "main.py")

HYBRID_DIR = os.path.join(BASE_DIR, "hybrid_model")
THRESHOLD_DIR = os.path.join(BASE_DIR, "threshold_model")

BG_COLOR = "#0f172a"
CARD_COLOR = "#111827"
TITLE_COLOR = "#22d3ee"
TEXT_COLOR = "#f8fafc"
SUBTEXT_COLOR = "#cbd5e1"
HYBRID_COLOR = "#16a34a"
HYBRID_HOVER = "#15803d"
THRESHOLD_COLOR = "#2563eb"
THRESHOLD_HOVER = "#1d4ed8"
EXIT_COLOR = "#dc2626"
EXIT_HOVER = "#b91c1c"

# =========================================================
# FUNCTIONS
# =========================================================
def run_hybrid():
    if not os.path.exists(VENV_PYTHON):
        messagebox.showerror("Error", f"Python interpreter not found:\n{VENV_PYTHON}")
        return

    if not os.path.exists(HYBRID_MAIN):
        messagebox.showerror("Error", f"File not found:\n{HYBRID_MAIN}")
        return

    subprocess.Popen(
        [VENV_PYTHON, HYBRID_MAIN],
        cwd=HYBRID_DIR
    )


def run_threshold():
    if not os.path.exists(VENV_PYTHON):
        messagebox.showerror("Error", f"Python interpreter not found:\n{VENV_PYTHON}")
        return

    if not os.path.exists(THRESHOLD_MAIN):
        messagebox.showerror("Error", f"File not found:\n{THRESHOLD_MAIN}")
        return

    subprocess.Popen(
        [VENV_PYTHON, THRESHOLD_MAIN],
        cwd=THRESHOLD_DIR
    )


def exit_app():
    root.destroy()


def bind_hover(button, normal_color, hover_color):
    button.bind("<Enter>", lambda e: button.config(bg=hover_color))
    button.bind("<Leave>", lambda e: button.config(bg=normal_color))


def disable_fullscreen(event=None):
    root.attributes("-fullscreen", False)


def enable_fullscreen(event=None):
    root.attributes("-fullscreen", True)


# =========================================================
# ROOT WINDOW
# =========================================================
root = tk.Tk()
root.title("Early Fatigue Detection System")
root.configure(bg=BG_COLOR)
root.attributes("-fullscreen", True)

root.bind("<Escape>", disable_fullscreen)
root.bind("<F11>", enable_fullscreen)

# =========================================================
# MAIN CONTAINER
# =========================================================
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

card = tk.Frame(
    main_frame,
    bg=CARD_COLOR,
    bd=0,
    highlightthickness=0
)
card.pack(ipadx=60, ipady=45)

# =========================================================
# TITLE
# =========================================================
title_label = tk.Label(
    card,
    text="Welcome to Early Fatigue Detection System",
    font=("Arial", 34, "bold"),
    fg=TITLE_COLOR,
    bg=CARD_COLOR
)
title_label.pack(pady=(25, 15))

subtitle_label = tk.Label(
    card,
    text="Choose a detection mode to continue",
    font=("Arial", 20),
    fg=SUBTEXT_COLOR,
    bg=CARD_COLOR
)
subtitle_label.pack(pady=(0, 25))

desc_label = tk.Label(
    card,
    text="This application provides two fatigue detection approaches:\n"
         "Threshold-Based Detection and Hybrid Model Detection.",
    font=("Arial", 16),
    fg=TEXT_COLOR,
    bg=CARD_COLOR,
    justify="center"
)
desc_label.pack(pady=(0, 35))

# =========================================================
# BUTTONS
# =========================================================
button_frame = tk.Frame(card, bg=CARD_COLOR)
button_frame.pack(pady=15)

hybrid_btn = tk.Button(
    button_frame,
    text="Hybrid Model",
    font=("Arial", 20, "bold"),
    width=20,
    height=2,
    bg=HYBRID_COLOR,
    fg="white",
    activeforeground="white",
    activebackground=HYBRID_HOVER,
    relief="flat",
    bd=0,
    cursor="hand2",
    command=run_hybrid
)
hybrid_btn.grid(row=0, column=0, padx=25, pady=20)

threshold_btn = tk.Button(
    button_frame,
    text="Threshold Model",
    font=("Arial", 20, "bold"),
    width=20,
    height=2,
    bg=THRESHOLD_COLOR,
    fg="white",
    activeforeground="white",
    activebackground=THRESHOLD_HOVER,
    relief="flat",
    bd=0,
    cursor="hand2",
    command=run_threshold
)
threshold_btn.grid(row=0, column=1, padx=25, pady=20)

bind_hover(hybrid_btn, HYBRID_COLOR, HYBRID_HOVER)
bind_hover(threshold_btn, THRESHOLD_COLOR, THRESHOLD_HOVER)

# =========================================================
# EXTRA INFO
# =========================================================
info_label = tk.Label(
    card,
    text="Press ESC to exit full screen",
    font=("Arial", 15),
    fg=SUBTEXT_COLOR,
    bg=CARD_COLOR
)
info_label.pack(pady=(25, 15))

exit_btn = tk.Button(
    card,
    text="Exit",
    font=("Arial", 16, "bold"),
    width=14,
    height=1,
    bg=EXIT_COLOR,
    fg="white",
    activeforeground="white",
    activebackground=EXIT_HOVER,
    relief="flat",
    bd=0,
    cursor="hand2",
    command=exit_app
)
exit_btn.pack(pady=(8, 25))

bind_hover(exit_btn, EXIT_COLOR, EXIT_HOVER)

# =========================================================
# FOOTER
# =========================================================
footer_label = tk.Label(
    root,
    text="Early Fatigue Detection Project",
    font=("Arial", 15),
    fg=SUBTEXT_COLOR,
    bg=BG_COLOR
)
footer_label.place(relx=0.5, rely=0.97, anchor="center")

root.mainloop()   