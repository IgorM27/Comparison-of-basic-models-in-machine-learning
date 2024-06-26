# Comparison of Basic Models in Machine Learning

## Введение:
В данном учебном проекте я сравнивал производительность различных базовых алгоритмов машинного обучения на классическом датасете Kaggle "Титаник". Целью является определение модели, наиболее подходящей для прогнозирования выживания пассажиров в условиях кораблекрушения.

## Используемые методы:
- Support Vector Machines (SVM): Алгоритм, ищущий гиперплоскость, максимально отделяющую классы друг от друга.
- Decision Tree: Дерево решений, построенное на основе рекурсивного деления данных по признакам.
- Bagging with Decision Tree: Метод ансамблевого обучения, комбинирующий предсказания нескольких деревьев решений для повышения точности.
- Random Forest: Алгоритм, основанный на множестве случайных деревьев решений.
- Boosting with Decision Tree: Метод ансамблевого обучения, последовательно обучающий деревья решений с фокусом на misclassified examples.

## Обучение и оценка:
- **Данные:** "Титаник" датасет из Kaggle, очищенный и подготовленный к работе.
- Модели обучались с использованием библиотеки scikit-learn.
- Оценка производительности осуществлялась с помощью метрики точности (accuracy score).

## Результаты:
| Модель                      | Точность |
|-----------------------------|----------|
| SVM                         | 0.82     |
| Decision Tree               | 0.86     |
| Bagging with Decision Tree  | 0.84     |
| Random Forest               | 0.87     |
| Boosting with Decision Tree | 0.84     |

## Вывод:
Из представленных результатов видно, что Random Forest демонстрирует наилучшую производительность среди базовых моделей, достигая точности 0.87.

## Дальнейшие направления:
- Тонкая настройка параметров: Оптимизация гиперпараметров каждой модели может привести к дальнейшему улучшению их точности.
- Исследование ансамблевых методов: Комбинирование различных моделей может привести к созданию еще более эффективного решения.
- Анализ ошибок: Изучение ошибок каждой модели может помочь в выявлении областей, требующих доработки.

## Заключение:
Данный проект демонстрирует возможности различных базовых алгоритмов машинного обучения на примере классической задачи классификации.
