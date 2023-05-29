from data_collection import collect_data
from data_processing import process_data
from data_analysis import analyze_data
from data_visualization import visualize_data

def main():
    # Prikupljanje podataka
    data = collect_data()

    # Obrada podataka
    processed_data = process_data(data)

    # Analiza podataka
    analyzed_data = analyze_data(processed_data)

    # Vizualizacija podataka
    visualize_data(analyzed_data)

if __name__ == "__main__":
    main()
