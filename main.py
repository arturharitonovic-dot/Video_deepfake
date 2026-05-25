import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms  # Добавили transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

import customtkinter as ctk
from tkinter import filedialog
import threading

# Настройки темы
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class DeepFakeDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Система детекции ИИ-видео (DeepFake Detector)")
        self.geometry("700x550")
        self.resizable(False, False)

        self.video_path = ""

        self.mtcnn = MTCNN(keep_all=False, device='cpu')

        self.classifier = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(num_ftrs, 2)

        try:
            self.classifier.load_state_dict(
                torch.load(r"C:\Work\model_weights.pth", map_location=torch.device('cpu'), weights_only=False))
            print("УСПЕХ: Обученная модель успешно загружена!")
        except Exception as e:
            print(f"ВНИМАНИЕ: Ошибка загрузки весов: {e}")

        self.classifier.eval()

        self.face_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.init_ui()

    def init_ui(self):
        self.title_label = ctk.CTkLabel(self, text="АНАЛИЗАТОР ВИДЕОПОТОКА", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=20)

        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(pady=10, fill="x", padx=20)

        self.btn_select = ctk.CTkButton(self.file_frame, text="Выбрать видео", command=self.select_video)
        self.btn_select.pack(side="left", padx=10, pady=10)

        self.lbl_file_path = ctk.CTkLabel(self.file_frame, text="Файл не выбран...", text_color="gray")
        self.lbl_file_path.pack(side="left", padx=10, fill="x", expand=True)

        self.btn_start = ctk.CTkButton(self, text="Запустить анализ", state="disabled",
                                       command=self.start_analysis_thread, fg_color="green", hover_color="darkgreen")
        self.btn_start.pack(pady=15)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.set(0)

        self.txt_log = ctk.CTkTextbox(self, height=180, width=660)
        self.txt_log.pack(pady=10, padx=20)
        self.txt_log.configure(state="disabled")

        self.verdict_frame = ctk.CTkFrame(self, height=60)
        self.verdict_frame.pack(pady=15, fill="x", padx=20)
        self.verdict_frame.pack_propagate(False)

        self.lbl_verdict = ctk.CTkLabel(self.verdict_frame, text="ОЖИДАНИЕ ЗАГРУЗКИ ФАЙЛА",
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_verdict.pack(expand=True)

    def log(self, message):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", message + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def select_video(self):
        file_types = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        self.video_path = filedialog.askopenfilename(title="Выберите видеофайл", filetypes=file_types)

        if self.video_path:
            self.lbl_file_path.configure(text=self.video_path, text_color="white")
            self.btn_start.configure(state="normal")
            self.log(f"Загружен файл: {self.video_path}")
            self.lbl_verdict.configure(text="ГОТОВ К АНАЛИЗУ", text_color="white")
            self.progress_bar.set(0)

    def start_analysis_thread(self):
        self.btn_start.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")
        self.lbl_verdict.configure(text="ИДЕТ СКАНИРОВАНИЕ...", text_color="orange")

        analysis_thread = threading.Thread(target=self.analyze_video_logic)
        analysis_thread.daemon = True
        analysis_thread.start()

    def analyze_video_logic(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0

        if total_frames == 0:
            self.log("Ошибка: не удалось прочитать видеофайл.")
            self.reset_ui_state()
            return

        frame_count = 0
        fake_frames = 0
        analyzed_frames = 0

        CONFIDENCE_THRESHOLD = 0.85
        SHARPNESS_THRESHOLD = 100.0

        self.log(f"Всего кадров: {total_frames}. Обработка каждого 30-го кадра...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 30 == 0:
                progress = frame_count / total_frames
                self.progress_bar.set(progress)

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image_rgb)

                boxes, _ = self.mtcnn.detect(pil_img)

                if boxes is not None:
                    analyzed_frames += 1
                    box = boxes[0].astype(int)
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])

                    face_crop = image_rgb[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    is_blurry = sharpness < SHARPNESS_THRESHOLD

                    face_pil = Image.fromarray(face_crop)
                    input_tensor = self.face_transforms(face_pil).unsqueeze(0)

                    with torch.no_grad():
                        outputs = self.classifier(input_tensor)
                        probabilities = F.softmax(outputs, dim=1)[0]
                        fake_prob = probabilities[0].item()  # Индекс 0 = Фейк

                        if fake_prob > CONFIDENCE_THRESHOLD or (fake_prob > 0.5 and is_blurry):
                            fake_frames += 1
                            self.log(
                                f"Кадр {frame_count}: [ФЕЙК] Вероятность: {fake_prob:.1%}, Резкость: {sharpness:.1f}")
                        else:
                            self.log(
                                f"Кадр {frame_count}: [РЕАЛ] Вероятность: {(1 - fake_prob):.1%}, Резкость: {sharpness:.1f}")
                else:
                    self.log(f"Кадр {frame_count}: Лицо не обнаружено")

        cap.release()
        self.progress_bar.set(1.0)

        if analyzed_frames > 0:
            fake_ratio = fake_frames / analyzed_frames
            self.log(f"\nАнализ завершен. Аномальных кадров: {fake_frames} из {analyzed_frames}")

            if fake_ratio > 0.35:
                self.lbl_verdict.configure(text=f"ВЕРДИКТ: ОБНАРУЖЕН ДИПФЕЙК ({fake_ratio:.1%})", text_color="red")
            else:
                self.lbl_verdict.configure(text=f"ВЕРДИКТ: ВИДЕО ПОДЛИННОЕ (Фейков: {fake_ratio:.1%})",
                                           text_color="green")
        else:
            self.lbl_verdict.configure(text="ОШИБКА: НЕТ ЛИЦ ДЛЯ АНАЛИЗА", text_color="gray")
            self.log("Не удалось найти лица на протяжении всего видео.")

        self.reset_ui_state()

    def reset_ui_state(self):
        self.btn_start.configure(state="normal")
        self.btn_select.configure(state="normal")


if __name__ == "__main__":
    app = DeepFakeDetectorApp()
    app.mainloop()