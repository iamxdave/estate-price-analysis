translation_dict = {
    'Cena': 'Price',
    'Liczba pokoi': 'Rooms',
    'Powierzchnia': 'Area',
    'Piętro': 'Floor',
    'Specyfikacja': 'Specification',
    'Lokalizacja': 'Location',
    'Link': 'Link',
    'Tekst': 'Text',
    'Cena za metr kwadratowy': 'Price per square meter',
}

def translate_keys(data):
    translated_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = translate_keys(v)  # Recursively translate keys of nested dictionaries
        translated_data[translation_dict.get(k, k)] = v
    return translated_data