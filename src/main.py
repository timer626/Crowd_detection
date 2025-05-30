import logging
from process import process_video
from visualize import visualize_results


def main():
    """Точка входа для детекции толпы."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Начало детекции")

    # Обработка видео и получение результатов
    results, people_per_frame, confidences, heatmap, max_diff_frame = process_video()

    # Визуализация результатов
    visualize_results(results, people_per_frame, confidences, heatmap, max_diff_frame)

    logger.info("Обработка завершена. Результаты сохранены в output/")


if __name__ == "__main__":
    main()