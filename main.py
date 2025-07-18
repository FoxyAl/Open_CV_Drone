import cv2
import numpy as np
import threading
import queue
import time


class VideoCaptureThread(threading.Thread):
    """Поток для захвата кадров с камеры"""

    def __init__(self, src=0, buffer_size=2):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError("Ошибка открытия камеры")
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame.copy())
        self.cap.release()

    def get_frame(self, timeout=0.5):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        time.sleep(0.1)
        while not self.frame_queue.empty():
            self.frame_queue.get()


def apply_heavy_salt_pepper_noise(image, noise_prob=0.1):
    """Добавление сильного шума 'соль и перец'"""
    noisy = np.copy(image)
    height, width = image.shape[:2]
    num_pixels = int(height * width * noise_prob)

    # Случайные координаты для "соли" (белые точки)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in [height, width]]
    noisy[coords[0], coords[1]] = 255

    # Случайные координаты для "перца" (черные точки)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in [height, width]]
    noisy[coords[0], coords[1]] = 0
    return noisy


def filter_image(noisy_frame):
    if not hasattr(filter_image, "prev_frame"):
        filter_image.prev_frame = noisy_frame

    # Простое временное сглаживание
    alpha = 0.8  # Коэффициент сглаживания
    filtered = cv2.addWeighted(noisy_frame, alpha, filter_image.prev_frame, 1 - alpha, 0)
    filter_image.prev_frame = filtered.copy()

    return filtered


def main():
    # Инициализация видеопотока в отдельном потоке
    capture_thread = VideoCaptureThread()
    capture_thread.start()

    while True:
        frame = capture_thread.get_frame()
        if frame is None:
            continue

        # 1. Оригинальный кадр
        original = frame.copy()

        # 2. Кадр с сильным шумом "соль-перец"
        noisy = apply_heavy_salt_pepper_noise(frame, noise_prob=0.2)

        # 3. Отфильтрованный кадр
        denoised = cv2.medianBlur(noisy, 5)
        filtered = filter_image(denoised)

        # Масштабируем кадры для отображения
        scale_percent = 70
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        original_resized = cv2.resize(original, dim)
        noisy_resized = cv2.resize(noisy, dim)
        filtered_resized = cv2.resize(filtered, dim)

        # Добавляем подписи
        cv2.putText(original_resized, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(noisy_resized, "Salt & Pepper Noise", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(filtered_resized, "Filter", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(filtered_resized, "Press 'E' to quit", (10, 330),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Объединяем все три кадра в одно окно
        top_row = np.hstack((original_resized, noisy_resized))
        combined = np.vstack((top_row, cv2.resize(filtered_resized, (width * 2, height))))

        cv2.imshow("Video Processing: Original | Noisy | Filtered", combined)

        # Завершение работы по нажатию 'E'
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Корректное завершение
    capture_thread.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
