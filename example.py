from transformers import pipeline

from functions_QA import get_detailed_answer, calculate_f1_for_json_dataset

#Пример использования
if __name__ == "__main__":
    # Инициализируем модель
    model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    context = "Leo Tolstoy wrote the war and Peace"
    question = "Who wrote the War and Peace?"
    
    result = get_detailed_answer(model, question, context, return_metadata=True)
    
    print("Детали ответа:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Файл
    original_file = "dataset.json"

    # Оцениваем производительность на отфильтрованном датасете
    print("\nEvaluating on dataset...")
    f1, accuracy = calculate_f1_for_json_dataset(original_file, model)

    print(f"\n=== Results ===")
    print(f"Dataset - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")