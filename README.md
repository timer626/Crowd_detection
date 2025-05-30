# Детекция толпы

# Обзор
Этот проект использует модели YOLOv8x и YOLO11x для детекции людей в видео с толпой. 

Оценка проводится по следующим метрикам:

FPS: Скорость обработки.

Среднее количество людей (Avg People): Среднее число обнаруженных людей в кадре.

Стандартное отклонение числа людей (People Std): Стабильность детекции.

# Выходные данные

Индивидуальные видео: output_yolov8x.mp4, output_yolo11x.mp4 с метками "Person" и значениями уверенности.

Сравнительное видео: output_comparison.mp4.

Метрики: model_comparison.csv (FPS, Avg People, People Std).

Интерактивный график: people_count_comparison.html с аннотацией ключевого кадра и всплывающими подсказками (кадр, число людей, уверенность).

Тепловые карты: heatmaps_comparison.png, показывающие плотность толпы.

Ключевой кадр: max_diff_frame_*.png с максимальной разницей в числе людей.

# Установка

1) git clone https://github.com/timer626/Crowd_detection

2) cd Сrowd_detection

3) python -m venv venv

4) source venv/bin/activate  # На Windows: venv\Scripts\activate

5) pip install -r requirements.txt

6) mkdir data

7) Поместите входное видео (Важно название: crowd.mp4) в папку data/.

# Использование
Запустите основной скрипт:

python src/main.py

# Папка с результатами
Результаты сохраняются в output/:

Видео: output_yolov8x.mp4, output_yolo11x.mp4, output_comparison.mp4.

Метрики: model_comparison.csv.

Визуализации: people_count_comparison.html, heatmaps_comparison.png, max_diff_frame_*.png.

Откройте people_count_comparison.html в браузере для просмотра интерактивного графика.

# Методология

Модели: YOLOv8x и YOLO11x от Ultralytics.
Обработка: Детекция и трекинг людей (класс 0) с уверенностью > 0.4, IoU > 0.5, размер изображения 640.
Инструменты: Ultralytics, OpenCV, Plotly, Pandas.

# Результаты

YOLO11x обычно показывает более высокий FPS (~3 против ~2.5 на CPU) и стабильную детекцию.
Интерактивный график выделяет кадр с максимальной разницей в числе людей.
Тепловые карты визуализируют плотность толпы для анализа безопасности.
Ключевой кадр (max_diff_frame_*.png) показывает расхождения моделей.

Рекомендация: YOLO11x для приложений реального времени благодаря скорости и точности.
Примечания

Обработка на CPU: ~12 минут для 60-секундного видео (30 FPS). На GPU — ~1 минута.
Для ускорения обработки измените imgsz=320 в process.py.

