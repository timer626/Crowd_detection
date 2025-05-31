import matplotlib.pyplot as plt
import plotly.express as px
import logging


def visualize_results(results, people_per_frame, confidences, heatmap, max_diff_frame):
    """Визуализация результатов с интерактивным графиком и тепловыми картами.

    Args:
        results: Словарь с FPS, Avg People, People Std.
        people_per_frame: Словарь с количеством людей по кадрам.
        confidences: Словарь со средней уверенностью по кадрам.
        heatmap: Словарь с тепловыми картами плотности толпы.
        max_diff_frame: Словарь с кадром, разницей и номером кадра.
    """
    logger = logging.getLogger(__name__)

    # Интерактивный график
    fig = px.line(title="Сравнение количества людей")
    for model_name in people_per_frame:
        fig.add_scatter(
            x=list(range(1, len(people_per_frame[model_name]) + 1)),
            y=people_per_frame[model_name],
            name=model_name,
            mode='lines',
            hovertemplate=f"Кадр: %{{x}}<br>Люди: %{{y}}<br>Модель: {model_name}<br>Средняя уверенность: %{{customdata:.2f}}",
            customdata=confidences[model_name]
        )
    fig.update_layout(xaxis_title="Кадр", yaxis_title="Люди")
    fig.add_vline(x=max_diff_frame["number"], line_dash="dash", annotation_text="Макс. разница", annotation_position="top")
    fig.write_html("output/people_count_comparison.html")
    #fig.show()
    logger.info("График сохранен в people_count_comparison.html")

    # Тепловые карты
    plt.figure(figsize=(12, 5))
    for idx, model_name in enumerate(people_per_frame, 1):
        plt.subplot(1, 2, idx)
        plt.imshow(heatmap[model_name], cmap="hot", interpolation="nearest")
        plt.colorbar(label="Плотность")
        plt.title(f"Тепловая карта ({model_name})")
    plt.tight_layout()
    plt.savefig("output/heatmaps_comparison.png")
    #plt.show()
    logger.info("Тепловые карты сохранены в heatmaps_comparison.png")

    # Вывод таблицы в консоль
    print("\nРезультаты:")
    print("Модель\tFPS\tСреднее кол-во людей\tСтд. отклонение")
    print("-" * 50)
    for row in results:
        print(f"{row['Model']}\t{row['FPS']:.2f}\t{row['Avg People']:.2f}\t\t{row['People Std']:.2f}")