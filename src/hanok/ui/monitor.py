"""Optional Tkinter training monitor retained from v4."""
import logging, os, queue, threading, tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, Entry, Button, Frame, Label
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from ..model.architecture import KoreanLLM
from ..inference.generation import generate
from ..training.checkpoint import find_latest_checkpoint
logger=logging.getLogger(__name__)
loss_history=[]

class TrainingMonitorGUI:
    def __init__(self, tokenizer, device, model_config: dict, checkpoint_dir: str,
                 stage: str = "sft"):
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model_config = model_config
        self.checkpoint_dir = checkpoint_dir
        self.stage = stage
        self.chat_model = None
        self.current_ckpt_path = None
        self.running = True
        self.msg_queue = queue.Queue()
        self._generation_busy = False

        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        except RuntimeError:
            pass

        self.root = tk.Tk()
        self.root.title("Hanok-LLM · Training Monitor")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 720)
        self.root.configure(bg="#eef2f7")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        left_frame = Frame(self.root, width=900, bg="#eef2f7")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        Label(left_frame, text="학습 진행 상황", bg="#eef2f7", fg="#172033",
              font=("Segoe UI", 17, "bold"), anchor="w").pack(fill=tk.X)
        Label(left_frame, text="Pretraining과 SFT를 별도 그래프로 표시합니다. 아래 툴바에서 확대·이동할 수 있습니다.",
              bg="#eef2f7", fg="#5b667a", font=("Segoe UI", 9), anchor="w").pack(fill=tk.X, pady=(2, 8))

        self.fig, (self.ax_pretrain, self.ax_sft) = plt.subplots(2, 1, figsize=(8, 7), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self._style_axis(self.ax_pretrain, "Pretraining", "#3976e8")
        self._style_axis(self.ax_sft, "SFT", "#e58a17")
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().configure(bg="#ffffff", highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, left_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.configure(background="#eef2f7")
        toolbar.pack(fill=tk.X, pady=(6, 0))

        right_frame = Frame(self.root, width=420, bg="#172033", padx=14, pady=14)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=0, pady=0)
        right_frame.pack_propagate(False)

        Label(right_frame, text="Hanok-LLM Chat", bg="#172033", fg="#ffffff",
              font=("Segoe UI", 17, "bold"), anchor="w").pack(fill=tk.X)
        self.ckpt_label = Label(right_frame, text="체크포인트: 아직 로드되지 않음",
                                bg="#172033", fg="#aebbd0", font=("Segoe UI", 9), anchor="w")
        self.ckpt_label.pack(fill=tk.X, pady=(3, 10))

        self.chat_display = scrolledtext.ScrolledText(right_frame, state='disabled', wrap=tk.WORD,
            font=("Segoe UI", 10), bg="#202c42", fg="#edf2fb", insertbackground="#ffffff",
            relief=tk.FLAT, padx=10, pady=10)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.tag_configure("user", foreground="#8ec5ff", spacing1=5)
        self.chat_display.tag_configure("model", foreground="#ffffff", spacing1=5)
        self.chat_display.tag_configure("system", foreground="#aebbd0", spacing1=3)

        options = Frame(right_frame, bg="#172033")
        options.pack(fill=tk.X, pady=(0, 7))
        Label(options, text="최대 토큰", bg="#172033", fg="#aebbd0",
              font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.max_tokens_var = tk.IntVar(value=128)
        tk.Spinbox(options, from_=32, to=512, increment=32, width=6,
                   textvariable=self.max_tokens_var, bg="#273650", fg="#ffffff",
                   buttonbackground="#273650", relief=tk.FLAT).pack(side=tk.LEFT, padx=(7, 0))
        Label(options, text="CPU / greedy", bg="#172033", fg="#7dd3a8",
              font=("Segoe UI", 9)).pack(side=tk.RIGHT)

        input_frame = Frame(right_frame, bg="#172033")
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = Entry(input_frame, font=("Segoe UI", 11), bg="#273650", fg="#ffffff",
                                insertbackground="#ffffff", relief=tk.FLAT)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7, padx=(0, 7))
        self.user_input.bind("<Return>", self.send_message)

        self.send_btn = Button(input_frame, text="전송", command=self.send_message, width=7,
                               bg="#3976e8", fg="#ffffff", activebackground="#5b91f2",
                               activeforeground="#ffffff", relief=tk.FLAT)
        self.send_btn.pack(side=tk.RIGHT, ipady=5)

        buttons = Frame(right_frame, bg="#172033")
        buttons.pack(fill=tk.X)
        for text, command in (("최신 체크포인트 로드", self.load_latest_checkpoint),
                              ("채팅 비우기", self.clear_chat)):
            Button(buttons, text=text, command=command, bg="#273650", fg="#ffffff",
                   activebackground="#34496d", activeforeground="#ffffff",
                   relief=tk.FLAT, padx=8, pady=6).pack(side=tk.LEFT, expand=True,
                                                        fill=tk.X, padx=(0, 5))

        self.root.after(250, self.update_gui)

    @staticmethod
    def _style_axis(axis, title, color):
        axis.set_title(title, loc="left", color=color, fontweight="bold")
        axis.set_xlabel("Step")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.22)
        axis.set_facecolor("#fbfcfe")

    def on_close(self):
        self.running = False
        self.root.destroy()

    def update_gui(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg["type"] == "loss":
                    self._update_plot()
                elif msg["type"] == "ckpt":
                    self.current_ckpt_path = msg["path"]
                    self.ckpt_label.config(text=f"현재 체크포인트: {Path(msg['path']).name}")
                elif msg["type"] == "log":
                    self._append_chat(f"[시스템] {msg['text']}\n", "system")
                elif msg["type"] == "chat_result":
                    self._append_chat(f"모델: {msg['text']}\n\n", "model")
                    self._generation_busy = False
                    self.send_btn.config(state=tk.NORMAL)
                elif msg["type"] == "chat_error":
                    self._append_chat(f"[시스템] 생성 오류: {msg['text']}\n", "system")
                    self._generation_busy = False
                    self.send_btn.config(state=tk.NORMAL)
        except queue.Empty:
            pass

        if self.running:
            self.root.after(250, self.update_gui)

    def _update_plot(self):
        if not loss_history:
            return
        histories = {"pretrain": [], "sft": []}
        for item in loss_history:
            stage = item.get("stage") or self.stage
            stage = "pretrain" if stage == "pretrain" else "sft"
            histories[stage].append(item)

        for axis, stage, color in ((self.ax_pretrain, "pretrain", "#3976e8"),
                                   (self.ax_sft, "sft", "#e58a17")):
            axis.clear()
            self._style_axis(axis, "Pretraining" if stage == "pretrain" else "SFT", color)
            rows = histories[stage]
            if rows:
                axis.plot([row["step"] for row in rows], [row["loss"] for row in rows],
                          color=color, linewidth=1.4, label="Train")
                val_rows = [row for row in rows if row.get("validation_loss") is not None]
                if val_rows:
                    axis.plot([row["step"] for row in val_rows],
                              [row["validation_loss"] for row in val_rows],
                              color="#7c3aed", linestyle="--", marker="o", markersize=3,
                              label="Validation")
                axis.legend(loc="upper right")
            else:
                axis.text(0.5, 0.5, "아직 기록 없음", transform=axis.transAxes,
                          ha="center", va="center", color="#94a3b8")
            axis.relim()
            axis.autoscale_view()
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _append_chat(self, text: str, tag: str = "user"):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text, tag)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def clear_chat(self):
        self.chat_display.config(state='normal')
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state='disabled')

    def load_latest_checkpoint(self):
        ckpt = find_latest_checkpoint(self.checkpoint_dir)
        if not ckpt:
            self._append_chat("[시스템] 체크포인트가 아직 없습니다.\n", "system")
            return

        try:
            self._append_chat(f"[시스템] 체크포인트 로딩 중(CPU): {Path(ckpt).name} ...\n", "system")
            self.root.update()

            if self.chat_model is None:
                self.chat_model = KoreanLLM(**self.model_config).to(self.device)

            checkpoint = torch.load(ckpt, map_location=self.device)
            model_state = dict(checkpoint['model_state_dict'])
            lora_keys = [key for key in model_state if key.endswith('.lora_A')]
            if lora_keys:
                self.chat_model.enable_lora(
                    model_state[lora_keys[0]].shape[0], checkpoint.get('lora_alpha')
                )
            for name in ('f_cos', 'f_sin'):
                if name in model_state:
                    model_state[name] = getattr(self.chat_model, name)
            self.chat_model.load_state_dict(model_state)
            self.chat_model.reset_rope_cache()
            self.chat_model.eval()

            self.current_ckpt_path = ckpt
            self.ckpt_label.config(text=f"현재 체크포인트: {Path(ckpt).name}")
            self._append_chat(f"[시스템] 로드 완료(CPU)! 이제 대화할 수 있어요.\n", "system")
        except Exception as e:
            self._append_chat(f"[시스템] 로드 실패: {e}\n", "system")

    def send_message(self, event=None):
        prompt = self.user_input.get().strip()
        if not prompt or self._generation_busy:
            return "break" if event else None

        self.user_input.delete(0, tk.END)
        self._append_chat(f"나: {prompt}\n", "user")

        if self.chat_model is None:
            self._append_chat("[시스템] 먼저 최신 체크포인트를 로드해주세요.\n", "system")
            return "break" if event else None

        try:
            max_tokens = max(32, min(512, int(self.max_tokens_var.get())))
        except (TypeError, ValueError, tk.TclError):
            max_tokens = 128

        self._generation_busy = True
        self.send_btn.config(state=tk.DISABLED)
        self._append_chat("[시스템] 생성 중...\n", "system")
        threading.Thread(target=self._generate_on_cpu,
                         args=(prompt, max_tokens), daemon=True).start()
        return "break" if event else None

    def _generate_on_cpu(self, prompt: str, max_tokens: int):
        try:
            response = generate(
                self.chat_model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=1.0,
                do_sample=False,
                device=self.device,
            )
            self.msg_queue.put({"type": "chat_result", "text": response})
        except Exception as exc:
            logger.exception("CPU chat generation failed")
            self.msg_queue.put({"type": "chat_error", "text": str(exc)})

    def notify_loss(self):
        self.msg_queue.put({"type": "loss"})

    def notify_checkpoint(self, path: str):
        self.msg_queue.put({"type": "ckpt", "path": path})

    def notify_log(self, text: str):
        self.msg_queue.put({"type": "log", "text": text})

    def run(self):
        self.root.mainloop()

