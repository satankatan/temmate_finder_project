from preprocessing.text_processor import preprocessor

# Тестовые примеры
test_texts = [
    "Я агро игрок, люблю давить на врага и контролить карту!",
    "Ну я типа саппорт, просто стою и хиллю тиммейтов",
    "Агрисьный мидер, каррю игры, фармлю и ганкаю линии",
    "Люблю играть керри, фармлю и несу игру"
]

print("🧪 Testing FIXED Text Preprocessor:")
for i, text in enumerate(test_texts, 1):
    cleaned = preprocessor.clean_text(text)
    print(f"Пример {i}:")
    print(f"📝 Original: {text}")
    print(f"🧹 Cleaned: {cleaned}")
    print("---")