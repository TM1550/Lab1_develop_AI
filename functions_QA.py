import json
import re
import random
from collections import Counter

from tqdm import tqdm

def normalize_text(text):
    """Нормализация текста для сравнения"""
    return re.sub(r'\W+', ' ', text.lower()).strip()

def calculate_f1_score(predicted, true):
    """Вычисляет F1-score между предсказанным и правильным ответом"""
    pred_tokens = normalize_text(predicted).split()
    true_tokens = normalize_text(true).split()

    if not pred_tokens or not true_tokens:
        return 0

    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def is_answer_correct(predicted, true, threshold=0.8):
    """Проверяет, правильный ли ответ на основе F1-score"""
    f1 = calculate_f1_score(predicted, true)
    return f1 >= threshold


def calculate_f1_for_json_dataset(json_file, model):
    """Вычисляет F1 на JSON датасете"""
    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_f1 = 0
    correct_count = 0

    for item in tqdm(dataset, desc="Calculating F1"):
        try:
            prediction = model(question=item["question"], context=item["context"])
            f1 = calculate_f1_score(prediction["answer"], item["answer"])
            total_f1 += f1

            if f1 >= 0.8:
                correct_count += 1
        except Exception as e:
            print(f"Error: {e}")
            continue

    accuracy = correct_count / len(dataset) if dataset else 0
    avg_f1 = total_f1 / len(dataset) if dataset else 0

    return avg_f1, accuracy


def get_detailed_answer(model, question, context, return_metadata=False):
    """
    Расширенная версия функции с дополнительной информацией
    
    Args:
        model: модель для question-answering
        question (str): вопрос
        context (str): контекст
        return_metadata (bool): возвращать ли метаданные
    
    Returns:
        dict: словарь с ответом, score и дополнительной информацией
    """
    try:
        prediction = model(question=question, context=context)
        
        result = {
            'answer': prediction['answer'],
            'score': prediction['score'],
            'start': prediction.get('start', 0),
            'end': prediction.get('end', 0)
        }
        
        if return_metadata:
            result.update({
                'question': question,
                'context_snippet': context[max(0, prediction.get('start', 0)-50):prediction.get('end', 0)+50],
                'timestamp': prediction.get('timestamp', None)
            })
        
        return result
        
    except Exception as e:
        print(f"Error in get_detailed_answer: {e}")
        return {
            'answer': '',
            'score': 0.0,
            'start': 0,
            'end': 0
        }
