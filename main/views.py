import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from django.shortcuts import render
from scipy import stats
from scipy.stats import kstest, chi2, norm, chisquare
from django.http import HttpResponse

# Устанавливаем высокое разрешение
plt.gcf().set_dpi(600)

df = pd.DataFrame()


def index(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            file = request.FILES['file']
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, header=None)

                # Удаление нечисловых символов из DataFrame
                df.replace(r'[а-яА-Я%]', '0', regex=True, inplace=True)
                df.apply(pd.to_numeric)

                # Удаление пустых столбцов
                df.dropna(how='all', axis=1, inplace=True)

                # Заполнение ячеек со значением NaN
                df = df.fillna(0)

                # Преобразование значений в целочисленный тип
                df = df.astype(int)

                # Преобразование DataFrame в одномерный массив данных
                data = df.values.flatten()

                # Исходные данные
                data_orig = df.copy()
                data_orig = data_orig.reset_index(drop=True)
                data_orig.index = data_orig.index + 1
                # Преобразование данных для отображения на шаблонах
                data_orig = data_orig.to_html(
                    classes='table table-striped table-hover table-bordered table-sm table-responsive text-center nowrap',
                    index_names=False,
                    justify='center',
                    header=False)

                # Создание новой фигуры для графика
                fig, ax = plt.subplots()

                # Задание уровня значимости
                alpha = 0.05

                # Визуализация распределения данных с помощью KDE plot
                sns.kdeplot(data, ax=ax, fill=True, alpha=0.1, label="Data")

                fig.savefig('plot.svg')

                # Стандартизация данных
                standardized_data = (data - np.mean(data)) / np.std(data)

                # Выполнение теста Колмогорова-Смирнова
                test_statistic, p_value = kstest(standardized_data, 'norm')

                # Выводим результаты теста
                print('Статистика теста:', test_statistic)
                print('p-значение:', p_value)
                print('Уровень значимости: 0,05')
                if p_value < alpha:
                    vivod1 = 'Вывод: распределение не является нормальным, так как p-значение меньше заданного уровня значимости. Гипотеза отклоняется.'
                else:
                    vivod1 = 'Вывод: распределение можно считать нормальным, так как p-значение больше заданного уровня значимости. Гипотеза не отклоняется.'

                # Добавление легенды
                ax.legend()

                # Добавление заголовка к сабплоту
                ax.set_title('Выборки')

                # Сохранение графика
                plt.savefig('templates/plot.png')

                # Сортировка данных
                data.sort()
                print("\nОтсортированные данные:\n", data)

                # Создание DataFrame для отсортированных данных
                data_sort = pd.DataFrame(data)
                data_sort = np.reshape(data_sort, (5, 12))
                data_sort = pd.DataFrame(data_sort)
                data_sort.index = data_sort.index + 1
                data_sort = data_sort.to_html(
                    classes='table table-striped table-hover table-bordered table-sm table-responsive text-center nowrap',
                    index_names=False,
                    justify='center',
                    header=False)

                # Вычисление среднего значения и стандартного отклонения
                mean = data.mean()
                std_dev = data.std(ddof=1)  # использование несмещенной оценки стандартного отклонения
                print("\nСреднее значение:", mean)
                print("Стандартное отклонение:", std_dev)

                # Создание интервалов
                num_intervals = len(data)
                bins = np.linspace(data.min(), data.max(), num_intervals + 1)
                intervals = [(bins[i], bins[i + 1]) for i in range(num_intervals)]
                print("\nИнтервалы:\n", intervals)

                # Подсчет наблюдаемых частот
                observed_freqs, _ = np.histogram(data, bins=bins)
                print("\nНаблюдаемые частоты:\n", observed_freqs)
                df_observed = pd.DataFrame(
                    {'Интервалы': [f'{int(interval[0])} - {int(interval[1])}' for interval in intervals],
                     'Количество': observed_freqs})
                df_observed.index = df_observed.index + 1
                df_observed = df_observed.to_html(
                    classes='table table-striped table-hover table-bordered table-sm table-responsive text-center nowrap',
                    index_names=False, justify='center')

                # Вычисление ожидаемых частот
                expected_freqs = [len(data) * (stats.norm.cdf(b, mean, std_dev) - stats.norm.cdf(a, mean, std_dev))
                                  for
                                  a, b in
                                  intervals]
                print("\nОжидаемые частоты:\n", expected_freqs)
                df_expected = pd.DataFrame(
                    {'Интервалы': [f'{int(interval[0])} - {int(interval[1])}' for interval in intervals],
                     'Количество': expected_freqs})
                df_expected.index = df_expected.index + 1
                df_expected = df_expected.to_html(
                    classes='table table-striped table-hover table-bordered table-sm table-responsive text-center nowrap',
                    index_names=False, justify='center')

                # Вычисление статистики хи-квадрат
                chi_square = sum((o - e) ** 2 / e for o, e in zip(observed_freqs, expected_freqs))
                print("\nСтатистика хи-квадрат:", chi_square)

                # Вычисление критического значения
                alpha = 0.05  # уровень значимости
                df = num_intervals - 1 - 2  # степени свободы: число интервалов - 1 - 2 (так как мы оцениваем среднее и стандартное отклонение из данных)
                critical_value = stats.chi2.ppf(1 - alpha, df)
                print("\nКритическое значение:", critical_value)

                # Принятие решения
                if chi_square > critical_value:
                    vivod2 = 'Вывод: статистика хи-квадрат больше критического значения, поэтому гипотеза о нормальном распределении отклоняется.'
                else:
                    vivod2 = 'Вывод: статистика хи-квадрат меньше или равна критическому значению, поэтому гипотеза о нормальном распределении не отклоняется.'


                return render(request, 'success.html', {
                    'data_orig': data_orig, 'data_sort': data_sort, 'mean': mean, 'std_dev': std_dev,
                    'intervals': intervals, 'observed_freqs': observed_freqs,
                    'expected_freqs': expected_freqs, 'chi_square': chi_square,
                    'critical_value': critical_value, 'test_statistic': test_statistic, 'p_value': p_value,
                    'alpha': alpha, 'df': df, 'df_observed': df_observed, 'df_expected': df_expected,
                    'vivod1': vivod1,
                    'vivod2': vivod2
                })

            else:
                return render(request, 'upload.html', {'error': 'Неверный формат файла'})
        else:
            return render(request, 'upload.html', {'error': 'Файл не выбран'})
    else:
        return render(request, 'upload.html')
