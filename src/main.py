import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import subprocess
import tkinter as tk
from tkinter import messagebox

# Создание папки для сохранения графиков
# Creating a folder for saving plots
output_dir = '../plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Загрузка данных
# Loading data
try:
    data = pd.read_csv('../data/fake_news.csv')
except Exception as e:
    messagebox.showerror("Error", f"Error loading data: {e}")
    raise

# Разделение данных на признаки (тексты новостей) и метки (REAL/FAKE)
# Splitting data into features (news texts) and labels (REAL/FAKE)
X = data['text']
y = data['label']

# Разделение на тренировочную и тестовую выборки
# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визуализация наиболее часто встречающихся слов
# Visualizing the most frequent words
count_vectorizer = CountVectorizer(stop_words='english', max_features=20)
word_count = count_vectorizer.fit_transform(data['text'])
words = count_vectorizer.get_feature_names_out()
word_freq = np.asarray(word_count.sum(axis=0)).flatten()

# Объединение слов и их частот в DataFrame и сортировка по частоте
# Combining words and their frequencies into a DataFrame and sorting by frequency
word_freq_df = pd.DataFrame({'word': words, 'frequency': word_freq})
word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False)

# Преобразование текстов в TF-IDF векторы
# Converting texts to TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Визуализация значений TF-IDF для первых 100 текстов
# Visualizing TF-IDF values for the first 100 texts
features = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_train[:100].toarray(), columns=features)

# Отбор наиболее значимых слов (например, с наибольшими средними значениями TF-IDF)
# Selecting the most significant words (e.g., with the highest average TF-IDF values)
important_features = tfidf_df.mean().sort_values(ascending=False).head(20).index
tfidf_df_filtered = tfidf_df[important_features]

# Инициализация и обучение модели
# Initializing and training the model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Прогнозирование на тестовых данных
# Predicting on the test data
y_pred = pac.predict(tfidf_test)

# Оценка точности модели
# Evaluating model accuracy
train_accuracy = pac.score(tfidf_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {test_accuracy * 100:.2f}%')
print(f'Точность: {test_accuracy * 100:.2f}%')


# Функция для сохранения графиков
# Function to save plots
def save_plot(filename, plot_func, *args, **kwargs):
    plt.figure(figsize=(8, 6))
    plot_func(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()  # Закрытие фигуры после сохранения


# Сохранение графиков
# Saving plots
save_plot('label_distribution.png', sns.countplot, x='label', hue='label', data=data, palette='viridis')
save_plot('most_frequent_words.png', sns.barplot, x='frequency', y='word', hue='word',
          data=word_freq_df, palette='viridis', dodge=False, legend=False)
save_plot('tfidf_features.png', sns.heatmap, tfidf_df_filtered.T, cmap='YlGnBu', annot=False, cbar=True)
save_plot('model_accuracy.png', sns.barplot, x=['Train Accuracy', 'Test Accuracy'],
          y=[train_accuracy, test_accuracy], hue=['Train Accuracy', 'Test Accuracy'], palette='viridis', legend=False)

# Построение и сохранение матрицы ошибок
# Building and saving the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
save_plot('confusion_matrix.png', sns.heatmap, conf_mat, annot=True, cmap='Blues', fmt='d',
          xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])

# Визуализация и сохранение ROC-кривой
# Visualizing and saving the ROC curve
y_prob = pac.decision_function(tfidf_test)
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='REAL')
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()


# Функция для открытия файлов
# Function to open files
def open_file(filepath):
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', filepath))
        elif platform.system() == 'Linux':
            subprocess.call(('xdg-open', filepath))
    except Exception as e:
        messagebox.showerror("Error", f"Error opening file {filepath}: {e}")


# Функция для запроса у пользователя
# Function to ask the user
def ask_to_open_plots():
    def on_button_click():
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            open_file(filepath)
        root.destroy()

    root = tk.Tk()
    root.title("Open Plots")

    label = tk.Label(root, text="The plots have been saved. Would you like to open them?\n"
                                "Графики были сохранены. Хотите открыть их?")
    label.pack(pady=10)

    button = tk.Button(root, text="Open Plots / Открыть графики", command=on_button_click)
    button.pack(pady=10)

    root.mainloop()


# Запуск GUI
# Running the GUI
ask_to_open_plots()
