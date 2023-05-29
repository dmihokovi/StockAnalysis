def process_data(data):
    # Izračunavanje 50-dnevnog pokretnog prosjeka
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    return data
