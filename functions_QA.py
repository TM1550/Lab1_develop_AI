import json
import re
import random
import logging
from collections import Counter
from typing import Dict, List, Tuple, Callable, Any, Union, Optional
from pathlib import Path

from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Нормализация текста для сравнения
    
    Args:
        text (str): Исходный текст для нормализации
        
    Returns:
        str: Нормализованный текст (нижний регистр, только слова и пробелы)
    """
    if not isinstance(text, str):
        logger.warning(f"Ожидается строка, получен {type(text)}")
        return ""
    
    try:
        normalized = re.sub(r'\W+', ' ', text.lower()).strip()
        return normalized
    except Exception as e:
        logger.warning(f"Ошибка при нормализации текста: {e}")
        return ""


def validate_model_output(prediction: Dict[str, Any]) -> bool:
    """
    Валидация выходных данных модели
    
    Args:
        prediction (Dict): Выходные данные модели
        
    Returns:
        bool: True если данные валидны
    """
    required_keys = ['answer']
    
    if not isinstance(prediction, dict):
        logger.warning("Выходные данные модели должны быть словарем")
        return False
    
    for key in required_keys:
        if key not in prediction:
            logger.warning(f"Отсутствует обязательный ключ '{key}' в выходных данных модели")
            return False
    
    # Валидация типов данных
    if not isinstance(prediction['answer'], str):
        logger.warning(f"Ответ должен быть строкой, получен {type(prediction['answer'])}")
        return False
    
    if 'score' in prediction and not isinstance(prediction['score'], (int, float)):
        logger.warning(f"Score должен быть числом, получен {type(prediction['score'])}")
        return False
    
    if 'start' in prediction and not isinstance(prediction['start'], int):
        logger.warning(f"Start должен быть целым числом, получен {type(prediction['start'])}")
        return False
    
    if 'end' in prediction and not isinstance(prediction['end'], int):
        logger.warning(f"End должен быть целым числом, получен {type(prediction['end'])}")
        return False
    
    return True


def calculate_f1_score(predicted: str, true: str) -> float:
    """
    Вычисляет F1-score между предсказанным и правильным ответом
    
    Args:
        predicted (str): Предсказанный ответ
        true (str): Правильный ответ
        
    Returns:
        float: F1-score между 0 и 1
    """
    if not isinstance(predicted, str):
        logger.warning(f"Предсказанный ответ должен быть строкой, получен {type(predicted)}")
        return 0.0
    
    if not isinstance(true, str):
        logger.warning(f"Правильный ответ должен быть строкой, получен {type(true)}")
        return 0.0
    
    try:
        pred_tokens = normalize_text(predicted).split()
        true_tokens = normalize_text(true).split()

        if not pred_tokens or not true_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(true_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(true_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
        
    except Exception as e:
        logger.warning(f"Ошибка при вычислении F1-score: {e}")
        return 0.0


def is_answer_correct(predicted: str, true: str, threshold: float = 0.8) -> bool:
    """
    Проверяет, правильный ли ответ на основе F1-score
    
    Args:
        predicted (str): Предсказанный ответ
        true (str): Правильный ответ
        threshold (float, optional): Порог F1-score для определения корректности. 
                                   По умолчанию 0.8
                                   
    Returns:
        bool: True если F1-score >= threshold, иначе False
    """
    if not 0 <= threshold <= 1:
        logger.warning(f"Threshold должен быть в диапазоне [0, 1], получен {threshold}")
        return False
    
    try:
        f1 = calculate_f1_score(predicted, true)
        return f1 >= threshold
    except Exception as e:
        logger.warning(f"Ошибка при проверке корректности ответа: {e}")
        return False


def validate_dataset_item(item: Dict[str, Any]) -> bool:
    """
    Валидация элемента датасета
    
    Args:
        item (Dict): Элемент датасета
        
    Returns:
        bool: True если элемент валиден
    """
    required_keys = ['question', 'context', 'answer']
    
    if not isinstance(item, dict):
        logger.warning("Элемент датасета должен быть словарем")
        return False
    
    for key in required_keys:
        if key not in item:
            logger.warning(f"Отсутствует обязательный ключ '{key}' в элементе датасета")
            return False
        
        if not isinstance(item[key], str):
            logger.warning(f"Значение ключа '{key}' должно быть строкой")
            return False
            
        if not item[key].strip():
            logger.warning(f"Значение ключа '{key}' не должно быть пустой строкой")
            return False
    
    return True


def calculate_f1_for_json_dataset(json_file: str, model: Callable) -> Tuple[float, float]:
    """
    Вычисляет F1-score и accuracy на JSON датасете
    
    Args:
        json_file (str): Путь к JSON файлу с датасетом. 
                        Ожидается список словарей с ключами "question", "context", "answer"
        model (Callable): Модель для question-answering. Должна принимать параметры 
                         question и context и возвращать словарь с ключом "answer"
                         
    Returns:
        Tuple[float, float]: Средний F1-score и accuracy по датасету
    """
    try:
        json_path = Path(json_file)
        if not json_path.exists():
            logger.error(f"Файл не найден: {json_file}")
            return 0.0, 0.0
        
        if json_path.stat().st_size == 0:
            logger.error(f"Файл пуст: {json_file}")
            return 0.0, 0.0
        
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON в файле {json_file}: {e}")
        return 0.0, 0.0
    except UnicodeDecodeError as e:
        logger.error(f"Ошибка декодирования текста в файле {json_file}: {e}")
        return 0.0, 0.0
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке файла {json_file}: {e}")
        return 0.0, 0.0
    
    if not isinstance(dataset, list):
        logger.error(f"Датасет должен быть списком, получен {type(dataset)}")
        return 0.0, 0.0
    
    if not dataset:
        logger.error("Датасет пуст")
        return 0.0, 0.0
    
    total_f1 = 0.0
    correct_count = 0
    processed_count = 0
    invalid_items = 0

    for i, item in enumerate(tqdm(dataset, desc="Calculating F1")):
        try:
            # Валидация элемента датасета
            if not validate_dataset_item(item):
                invalid_items += 1
                continue
            
            # Проверка что модель callable
            if not callable(model):
                logger.warning("Модель должна быть callable")
                continue
            
            prediction = model(question=item["question"], context=item["context"])
            
            # Валидация выходных данных модели
            if not validate_model_output(prediction):
                invalid_items += 1
                continue
            
            f1 = calculate_f1_score(prediction["answer"], item["answer"])
            total_f1 += f1

            if f1 >= 0.8:
                correct_count += 1
                
            processed_count += 1
            
        except Exception as e:
            logger.warning(f"Ошибка при обработке элемента {i}: {e}")
            continue

    if invalid_items > 0:
        logger.info(f"Пропущено {invalid_items} невалидных элементов")

    if processed_count == 0:
        logger.error("Не удалось обработать ни одного элемента датасета")
        return 0.0, 0.0

    accuracy = correct_count / processed_count
    avg_f1 = total_f1 / processed_count

    logger.info(f"Обработано элементов: {processed_count}/{len(dataset)}")
    logger.info(f"F1-score: {avg_f1:.4f}, Accuracy: {accuracy:.4f}")

    return avg_f1, accuracy


def get_detailed_answer(
    model: Callable, 
    question: str, 
    context: str, 
    return_metadata: bool = False,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Расширенная версия функции с дополнительной информацией
    
    Args:
        model (Callable): Модель для question-answering. Должна принимать параметры 
                         question и context и возвращать словарь с ключами:
                         "answer", "score", "start", "end"
        question (str): Вопрос для модели
        context (str): Контекст для поиска ответа
        return_metadata (bool, optional): Возвращать ли дополнительную метаинформацию. 
                                         По умолчанию False
        max_retries (int): Максимальное количество попыток вызова модели
                                         
    Returns:
        Dict[str, Any]: Словарь с результатами, содержащий ключи:
            - answer (str): найденный ответ
            - score (float): уверенность модели
            - start (int): начальная позиция ответа в контексте
            - end (int): конечная позиция ответа в контексте
            - question (str, optional): исходный вопрос (только если return_metadata=True)
            - context_snippet (str, optional): сниппет контекста вокруг ответа 
                                              (только если return_metadata=True)
            - timestamp (Any, optional): временная метка (только если return_metadata=True)
            - error (str, optional): сообщение об ошибке (только при ошибке)
    """
    # Валидация входных параметров
    if not callable(model):
        logger.error("Модель должна быть callable")
        return _create_error_result(question, context, return_metadata, "Модель должна быть callable")
    
    if not isinstance(question, str) or not question.strip():
        logger.error("Вопрос должен быть непустой строкой")
        return _create_error_result(question, context, return_metadata, "Вопрос должен быть непустой строкой")
    
    if not isinstance(context, str) or not context.strip():
        logger.error("Контекст должен быть непустой строкой")
        return _create_error_result(question, context, return_metadata, "Контекст должен быть непустой строкой")
    
    if not isinstance(return_metadata, bool):
        logger.warning("return_metadata должен быть булевым значением, используется значение по умолчанию False")
        return_metadata = False
    
    try:
        # Безопасный вызов модели
        prediction = model(question=question, context=context)
        
        result = {
            'answer': prediction.get('answer', ''),
            'score': prediction.get('score', 0.0),
            'start': prediction.get('start', 0),
            'end': prediction.get('end', 0)
        }
        
        if return_metadata:
            start_pos = prediction.get('start', 0)
            end_pos = prediction.get('end', 0)
            
            # Безопасное извлечение сниппета контекста
            context_snippet = ""
            try:
                snippet_start = max(0, start_pos - 50)
                snippet_end = min(len(context), end_pos + 50)
                context_snippet = context[snippet_start:snippet_end]
            except Exception as e:
                logger.warning(f"Ошибка при извлечении сниппета контекста: {e}")
            
            result.update({
                'question': question,
                'context_snippet': context_snippet,
                'timestamp': prediction.get('timestamp', None)
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Критическая ошибка в get_detailed_answer: {e}")
        return _create_error_result(question, context, return_metadata, str(e))


def _create_error_result(question: str, context: str, return_metadata: bool, error_msg: str) -> Dict[str, Any]:
    """
    Создает результат с ошибкой
    
    Args:
        question (str): Вопрос
        context (str): Контекст
        return_metadata (bool): Возвращать ли метаданные
        error_msg (str): Сообщение об ошибке
        
    Returns:
        Dict[str, Any]: Результат с ошибкой
    """
    error_result = {
        'answer': '',
        'score': 0.0,
        'start': 0,
        'end': 0,
        'error': error_msg
    }
    
    if return_metadata:
        error_result.update({
            'question': question,
            'context_snippet': context[:100] if context else "",
            'timestamp': None
        })
    
    return error_result