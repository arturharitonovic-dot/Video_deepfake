import customtkinter as ctk
import cv2
import threading
import os
import tkinter.filedialog as fd
from PIL import Image
from detector import DeepfakeDetector
from analyzer import TemporalAnalyzer

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DeepfakeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Deepfake Detection and Temporal Consistency Analysis System")
        self.geometry("1100x750")

        weights_path = "weights/resnet18_deepfake.pth"
        if not os.path.exists(weights_path):
            print(f"[!] Файл весов {weights_path} отсутствует. Инференс запустится на случайных весах.")
            self.detector = DeepfakeDetector(model_weights_path=None)
        else:
            self.detector = DeepfakeDetector(weights_path)

        self.analyzer = TemporalAnalyzer(window_size=30)
        self.is_processing = False
        self.cap = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=3)  # Окно видео
        self.main_container.grid_columnconfigure(1, weight=1)  # Окно аналитики
        self.main_container.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self.main_container)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.video_display = ctk.CTkLabel(self.left_panel, text="Ожидание загрузки видеофайла...", font=("Arial", 16))
        self.video_display.pack(expand=True, fill="both", padx=10, pady=10)

        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.btn_select = ctk.CTkButton(self.right_panel, text="Открыть видеофайл", command=self.start_analysis_flow,
                                        font=("Arial", 14), height=40)
        self.btn_select.pack(pady=20, padx=15, fill="x")

        self.btn_interrupt = ctk.CTkButton(self.right_panel, text="Прервать анализ", command=self.terminate_analysis,
                                           state="disabled", font=("Arial", 14), height=40, fg_color="#D32F2F",
                                           hover_color="#B71C1C")
        self.btn_interrupt.pack(pady=5, padx=15, fill="x")

        self.divider = ctk.CTkFrame(self.right_panel, height=2, fg_color="gray30")
        self.divider.pack(pady=25, padx=15, fill="x")

        self.lbl_metrics_title = ctk.CTkLabel(self.right_panel, text="МОНИТОРИНГ ПРИЗНАКОВ", font=("Arial", 14, "bold"))
        self.lbl_metrics_title.pack(pady=10)

        self.val_prob = ctk.CTkLabel(self.right_panel, text="Spatial Probability: --", font=("Arial", 13))
        self.val_prob.pack(pady=8, anchor="w", padx=20)

        self.val_var = ctk.CTkLabel(self.right_panel, text="Temporal Variance: --", font=("Arial", 13))
        self.val_var.pack(pady=8, anchor="w", padx=20)

        self.val_flicker = ctk.CTkLabel(self.right_panel, text="Inter-frame Flickering: --", font=("Arial", 13))
        self.val_flicker.pack(pady=8, anchor="w", padx=20)

        self.verdict_panel = ctk.CTkFrame(self.right_panel, height=80, fg_color="gray20")
        self.verdict_panel.pack(pady=40, padx=15, fill="x")
        self.verdict_panel.pack_propagate(False)

        self.val_verdict = ctk.CTkLabel(self.verdict_panel, text="ВЕРДИКТ: ОЖИДАНИЕ", font=("Arial", 16, "bold"),
                                        text_color="gray60")
        self.val_verdict.pack(expand=True)

    def start_analysis_flow(self):
        target_path = fd.askopenfilename(filetypes=[("Медиаконтейнеры", "*.mp4 *.avi *.mov *.mkv")])
        if target_path:
            self.terminate_analysis()
            self.analyzer.reset()

            self.cap = cv2.VideoCapture(target_path)
            self.is_processing = True

            self.btn_select.configure(state="disabled")
            self.btn_interrupt.configure(state="normal")

            worker = threading.Thread(target=self.video_processing_loop, daemon=True)
            worker.start()

    def terminate_analysis(self):
        self.is_processing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_select.configure(state="normal")
        self.btn_interrupt.configure(state="disabled")
        self.video_display.configure(image=None, text="Анализ завершен. Ожидание нового файла.")
        self.val_verdict.configure(text="ВЕРДИКТ: ОСТАНОВЛЕН", text_color="gray60")

    def video_processing_loop(self):
        while self.is_processing and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            orig_h, orig_w, _ = frame.shape
            scale_factor = 640.0 / max(orig_h, orig_w)
            disp_w, disp_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
            display_frame = cv2.resize(frame, (disp_w, disp_h))

            prob, bbox = self.detector.process_frame(frame)

            if prob is not None:
                metrics = self.analyzer.update(prob)

                x_ratio = disp_w / orig_w
                y_ratio = disp_h / orig_h
                x1, y1, x2, y2 = bbox
                x1_d, y1_d = int(x1 * x_ratio), int(y1 * y_ratio)
                x2_d, y2_d = int(x2 * x_ratio), int(y2 * y_ratio)

                color = (0, 0, 255) if metrics["decision"] == "FAKE" else (0, 255, 0)
                cv2.rectangle(display_frame, (x1_d, y1_d), (x2_d, y2_d), color, 2)

                self.after(0, self.sync_ui_metrics, metrics)
            else:
                self.after(0, self.sync_ui_no_face)

            rgb_disp = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_disp)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

            self.after(0, self.sync_video_frame, ctk_img)

            cv2.waitKey(1)

        self.after(0, self.terminate_analysis)

    def sync_video_frame(self, ctk_img):
        if self.is_processing:
            self.video_display.configure(image=ctk_img, text="")
            self.video_display._image_tracker = ctk_img

    def sync_ui_metrics(self, metrics):
        self.val_prob.configure(text=f"Spatial Probability: {metrics['smoothed_probability']:.4f}")
        self.val_var.configure(text=f"Temporal Variance: {metrics['variance']:.6f}")

        flicker_status = "ОБНАРУЖЕНО" if metrics['is_flickering'] else "СТАБИЛЬНО"
        flicker_color = "#E53935" if metrics['is_flickering'] else "#4CAF50"
        self.val_flicker.configure(text=f"Inter-frame Flickering: {flicker_status}", text_color=flicker_color)

        if metrics['decision'] == "FAKE":
            self.val_verdict.configure(text="ВЕРДИКТ: ПОДДЕЛКА (FAKE)", text_color="#E53935")
        else:
            self.val_verdict.configure(text="ВЕРДИКТ: ОРИГИНАЛ (REAL)", text_color="#4CAF50")

    def sync_ui_no_face(self):
        self.val_prob.configure(text="Spatial Probability: Лицо не найдено")


if __name__ == "__main__":
    app = DeepfakeApp()
    app.mainloop()