import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import torch
import os


def process_frame(model, frame):
    """Детекция и отслеживание людей в кадре.

    Args:
        model: Экземпляр модели YOLO.
        frame: Входной кадр ndarray(массив numpy).

    Returns:
        boxes: Обнаруженные рамки с координатами и уверенностью.
    """
    results = model.track(frame, persist=True, classes=[0], conf=0.4, iou=0.5, imgsz=640)
    return results[0].boxes


def process_video():
    """Обработка видео,создание видео и метрик.

    Returns:
        results: Словарь с FPS, Avg People, People Std.
        people_per_frame: Словарь с количеством людей по кадрам.
        confidences: Словарь со средней уверенностью по кадрам.
        heatmap: Словарь с тепловыми картами плотности толпы.
        max_diff_frame: Словарь с кадром, разницей и номером кадра.
    """
    logger = logging.getLogger(__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используемое устройство: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Пути
    video_path = os.path.join(os.path.dirname(__file__), "..", "data", "crowd.mp4")
    output_paths = {"YOLOv8x": "output/output_yolov8x.mp4", "YOLO11x": "output/output_yolo11x.mp4"}
    comparison_output = "output/output_comparison.mp4"
    os.makedirs("output", exist_ok=True)

    # Модели
    # Стало
    models = {
        "YOLOv8x": YOLO("yolov8x.pt", device=device),
        "YOLO11x": YOLO("yolo11x.pt", device=device)
    }
    logger.info("Модели загружены")

    # Результаты
    results = {"Model": [], "FPS": [], "Avg People": [], "People Std": []}
    people_per_frame = {name: [] for name in models}
    confidences = {name: [] for name in models}
    max_diff_frame = {"frame": None, "diff": 0, "number": 0}

    # Видео
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Тепловые карты
    heatmap = {name: np.zeros((frame_height, frame_width)) for name in models}

    # Видеопотоки
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writers = {name: cv2.VideoWriter(output_paths[name], fourcc, fps, (frame_width, frame_height)) for name in models}
    comp_writer = cv2.VideoWriter(comparison_output, fourcc, fps, (frame_width * 2, frame_height))

    # Обработка
    frame_count = 0
    model_times = {name: 0 for name in models}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        logger.info(f"Кадр {frame_count}")

        comp_frame = np.zeros((frame_height, frame_width * 2, 3), dtype=np.uint8)
        people_counts = {}

        for idx, (model_name, model) in enumerate(models.items()):
            start_time = time.time()
            frame_copy = frame.copy()

            # Инференс
            boxes = process_frame(model, frame_copy)

            # Отрисовка
            people_in_frame = 0
            frame_confidences = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = f"Person ({conf:.2f})"
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                people_in_frame += 1
                frame_confidences.append(float(conf))
                heatmap[model_name][y1:y2, x1:x2] += 1

            people_per_frame[model_name].append(people_in_frame)
            confidences[model_name].append(np.mean(frame_confidences) if frame_confidences else 0)
            people_counts[model_name] = people_in_frame
            video_writers[model_name].write(frame_copy)

            # Сравнительный кадр
            comp_frame[:, idx * frame_width:(idx + 1) * frame_width] = frame_copy
            cv2.putText(comp_frame, model_name, (idx * frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            model_times[model_name] += time.time() - start_time

        # Ключевой кадр
        diff = abs(people_counts["YOLOv8x"] - people_counts["YOLO11x"])
        if diff > max_diff_frame["diff"]:
            max_diff_frame = {"frame": comp_frame.copy(), "diff": diff, "number": frame_count}

        comp_writer.write(comp_frame)

    cap.release()
    for writer in video_writers.values():
        writer.release()
    comp_writer.release()

    # Сохранение ключевого кадра
    cv2.imwrite(f"output/max_diff_frame_{max_diff_frame['number']}.png", max_diff_frame["frame"])
    logger.info(f"Ключевой кадр сохранен: max_diff_frame_{max_diff_frame['number']}.png")

    # Метрики
    for model_name in models:
        avg_fps = frame_count / model_times[model_name]
        avg_people = np.mean(people_per_frame[model_name])
        people_std = np.std(people_per_frame[model_name])
        results["Model"].append(model_name)
        results["FPS"].append(avg_fps)
        results["Avg People"].append(avg_people)
        results["People Std"].append(people_std)
        logger.info(f"{model_name}: Люди: {avg_people:.2f}, Std: {people_std:.2f}, FPS: {avg_fps:.2f}")


    return results, people_per_frame, confidences, heatmap, max_diff_frame