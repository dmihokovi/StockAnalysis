import matplotlib.pyplot as plt

def visualize_data(data):
    # Vizualizacija cijene dionice i pokretnog prosjeka
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Cijena dionice')
    plt.plot(data['50_MA'], label='50-dnevni pokretni prosjek')
    plt.title('Apple Inc. (AAPL) Cijena dionice i 50-dnevni pokretni prosjek')
    plt.xlabel('Datum')
    plt.ylabel('Cijena ($)')
    plt.legend()
    plt.show()
