import numpy as np
import pandas as pd
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (PolynomialFeatures, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (r2_score, mean_absolute_error)
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

# обозначаем константы
FILE_NAME = "Набор данных для обучения модели.xlsx"                                                                      # имя файла для обучения модели
TARGET_R2 = 0.90                                                                                                         # целевое качество модели
MAX_REMOVAL_PERCENT = 0.15                                                                                               # для улучшения качества модели можно удалить не более 15% данных
REMOVE_PER_ITERATION = 0.01                                                                                              # за одну итерацию откусываем не более 1% данных
RANDOM_STATE = 50                                                                                                        # фиксируем количество данных, отведённых на текстирование модели
RIDGE_ALPHA = 2.0                                                                                                        # зависимости это хорошо, но если случится, что у пары получился абсолютная взаимосвязь, то она будет тянуть обеяло на себя и другие коэффициенты перестанут быть значимыми, поэтому мы ограничим вес коэффициента

df_original = pd.read_excel(FILE_NAME)                                                                                   # собираем датафрейм
df_original["original_index"] = df_original.index                                                                        # фиксируем позиции строк в массиве
df_original["log_price"] = np.log(df_original["цена"])                                                                   # логарифмируем цену - это для выравнивания распределения, чтобы объекты, которые стоят очень дорого или очень дёшево не доминировали в анализе

# распределяем показатели по категориям
numeric_features = [
    "площадь",
    "комнаты",
    "этаж",
    "возраст_дома",
    "расстояние_до_центра",
    "расстояние_до_метро",
    "школ_рядом",
    "индекс_преступности"]

binary_features = [
    "парковка",
    "лифт",
    "вид_на_море"]

categorical_features = [
    "район"]

ALL_FEATURES = (numeric_features + binary_features + categorical_features)

# блок преобразования массивов данных
numeric_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),                                                                       # чтобы данные были сопоставимы, мы закостылим пустоты медианными значениями, но это временная мера
    ("poly", PolynomialFeatures(degree=2, include_bias=False))])                                                         # создаём коэффициенты для нелинейной регрессии

categorical_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transform, numeric_features),
    ("cat", categorical_transform, categorical_features),
    ("bin", "passthrough", binary_features)])

df = df_original.copy()                                                                                                  # копируем датафрейм - с ним будут происходить некоторое насилие
all_removed_anomalies = []                                                                                               # оглашаем список
iteration = 0
max_removals = int(len(df_original) * MAX_REMOVAL_PERCENT)                                                               # отражение предела удаляемых объектов в абсолютном формате - даже если целевое качество модели недостигнуто, иттерации остановятся при удалении 15% объектов

# запускаем цикл проверки
while True:
    iteration += 1
    print("\n")
    print("=" * 70)
    print(f"Итерация №{iteration}")
    print("=" * 70)

    X = df[ALL_FEATURES]
    y = df["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)                  # делим модель на train/test для проверки качества
    model = Pipeline([("preprocessor", preprocessor), ("regression", Ridge(alpha=RIDGE_ALPHA))])                         # основная часть для расчёта и обучения модели: Pipeline - контейнер, preprocessor - заполняет пустоты медианами и генерит показатели (прошлый блок), Ridge - мат. анализ
    model.fit(X_train, y_train)                                                                                          # запускаем обучение на данных

    preds = model.predict(X_test)                                                                                        # запускаем обученную модель на тестовой выборке
    r2 = r2_score(y_test, preds)                                                                                         # применяем коэффициент детерминации, сравнивая фактические цены в тестовых данных с прогнозными
    mae = mean_absolute_error(y_test, preds)                                                                             # страхуем себя коэффициентом MAE - он показывает отклонение в абсолютных величинах (коэффициент детерминации может быть хорошим, но в абсолютном виде ошибка может быть значительной)

    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Current dataset size: {len(df)}")

# оцениваем количество уже удалённых объектов и прерываем если достигли целевого качества или максимума по количеству итераций
    total_removed = sum(len(x) for x in all_removed_anomalies)                                                           # Столько объектов уже удалено
    full_preds = model.predict(X)
    residuals = np.abs(y - full_preds)

    df["residual"] = residuals
    df["прогнозная_цена"] = full_preds
    df["прогнозная_цена"] = np.exp(df["прогнозная_цена"])                                                                # прогнозную цену возвращаем обратно из логарифмического вида в нормальный
    df["коэффициент_отклонения"] = (np.abs(df["цена"] - df["прогнозная_цена"]) / df["цена"]) * 100

    remove_count = max(1, int(len(df) * REMOVE_PER_ITERATION))                                                            # мы разрешили удалять только 1% объектов за 1 иттерацию
    worst_rows = df.nlargest(remove_count, "residual")
    print(f"Удаление {remove_count} наиболее весомых аномалий...")

    all_removed_anomalies.append(worst_rows.copy())
    df = df.drop(worst_rows.index)

# проверяем выполнение условий для завершения цикла
    if r2 >= TARGET_R2:
        print("\nДОСТИГНУТ ПОРОГ КАЧЕСТВА МОДЕЛИ")
        break
    if total_removed >= max_removals:
        print("\nДОСТИГНУТО МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ИСКЛЮЧЕНИЙ")
        break

print("\n")
print("=" * 70)
print("Полностью обученная модель")
print("=" * 70)
print(f"Финальная модель включает в себя объектов: {len(df)}")
print(
    f"Количество выявленных аномалий: "
    f"{len(df_original) - len(df)}")

# преобразование массива: собираем все новые параметры в обученной модели в кучу (х^2, x и т.д.)
poly_names = model.named_steps["preprocessor"].named_transformers_["num"].named_steps["poly"].get_feature_names_out(numeric_features) # получаем названия всех числовых не линейных признаков
cat_names = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features) # получаем названия категориальных признаков
all_feature_names = (list(poly_names) + list(cat_names) + binary_features)                                               # объединяем числовые, категориальные и бинарные признаки в обученной модели
coefficients = model.named_steps["regression"].coef_                                                                     # а это само значение коэффициентов
coef_df = pd.DataFrame({"feature": all_feature_names, "coefficient": coefficients})                                      # делаем из значений коэффициентов таблицу
coef_df["abs_coef"] = np.abs(coef_df["coefficient"])                                                                     # и присваиваем абсолютные значения коэффициентам
coef_df = coef_df.sort_values(by="abs_coef", ascending=False)                                                            # сортируем по весу

print("\n")
print("=" * 70)
print("Топ факторов, наиболее сильно влияющих на прогноз")
print("=" * 70)
print(coef_df.head(10))

# пишем результат в файл
datename = datetime.now().strftime('%d.%m %H.%M.%S')
output_path = "Коэффициенты и аномалии " + datename + ".xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="аномалии", index=False)
    coef_df.to_excel(writer, sheet_name="коэффициенты", index=False)
    if len(all_removed_anomalies) > 0:
        print("\n")
        print("=" * 70)
        print("Полный список аномалий:")
        print("=" * 70)
        anomalies_df = pd.concat(all_removed_anomalies, ignore_index=True)
        anomalies_df = anomalies_df.sort_values(by="residual", ascending=False)
        display_columns = [
            "original_index",
            "цена",
            "прогнозная_цена",
            "коэффициент_отклонения",
            "площадь",
            "комнаты",
            "этаж",
            "возраст_дома",
            "район",
            "расстояние_до_центра",
            "парковка",
            "лифт",
            "вид_на_море",
            "расстояние_до_метро",
            "школ_рядом",
            "индекс_преступности",
            "residual"]
        print(anomalies_df[display_columns].to_string(index=False))
    else: print("\nАномалии не обнаружены.")

root = tk.Tk()
root.title("Система оценки недвижимости")
root.geometry("700x850")
root.resizable(False, False)

area_var = tk.StringVar()
rooms_var = tk.StringVar()
floor_var = tk.StringVar()
age_var = tk.StringVar()
center_var = tk.StringVar()
metro_var = tk.StringVar()
schools_var = tk.StringVar()
crime_var = tk.StringVar()

parking_var = tk.IntVar()
elevator_var = tk.IntVar()
sea_var = tk.IntVar()

region_var = tk.StringVar()
region_var.set("центр")

title_label = tk.Label(root, text="Система оценки недвижимости", font=("Arial", 22, "bold"))
title_label.pack(pady=20)
main_frame = tk.Frame(root)
main_frame.pack(pady=10)

def add_input(text, variable, row):
    label = tk.Label(main_frame, text=text, font=("Arial", 12))
    label.grid(row=row, column=0, sticky="w", padx=10, pady=8)
    entry = tk.Entry(main_frame, textvariable=variable, width=30, font=("Arial", 11))
    entry.grid(row=row, column=1, padx=10, pady=8)

add_input("Площадь, кв.м.", area_var, 0)
add_input("Количество комнат", rooms_var, 1)
add_input("Этаж", floor_var, 2)
add_input("Возраст дома, лет", age_var, 3)
add_input("Расстояние до центра, км", center_var, 4)
add_input("Расстояние до метро, км", metro_var, 5)
add_input("Количество школ в радиусе 1 км", schools_var, 6)
add_input("Индекс преступности района", crime_var, 7)

region_label = tk.Label(main_frame, text="Район", font=("Arial", 12))
region_label.grid(row=8, column=0, sticky="w", padx=10, pady=8)
region_combo = ttk.Combobox(main_frame, textvariable=region_var, values=["центр", "средний", "окраина"], state="readonly", width=27)
region_combo.grid(row=8, column=1, padx=10, pady=8)

checkbox_frame = tk.Frame(root)
checkbox_frame.pack(pady=20)
parking_check = tk.Checkbutton(checkbox_frame, text="Парковка", variable=parking_var, font=("Arial", 11))
parking_check.grid(row=0, column=0, padx=15)
lift_check = tk.Checkbutton(checkbox_frame, text="Лифт", variable=elevator_var, font=("Arial", 11))
lift_check.grid(row=0, column=1, padx=15)
sea_check = tk.Checkbutton(checkbox_frame, text="Вид на море", variable=sea_var, font=("Arial", 11))
sea_check.grid(row=0, column=2, padx=15)

result_label = tk.Label(root, text="", font=("Arial", 20, "bold"), fg="green")
result_label.pack(pady=40)

def predict_price():
    try:
        flat = pd.DataFrame([{
            "площадь": float(area_var.get()),
            "комнаты": int(rooms_var.get()),
            "этаж": int(floor_var.get()),
            "возраст_дома": float(age_var.get()),
            "расстояние_до_центра": float(center_var.get()),
            "расстояние_до_метро": float(metro_var.get()),
            "школ_рядом": float(schools_var.get()),
            "индекс_преступности": float(crime_var.get()),
            "парковка": parking_var.get(),
            "лифт": elevator_var.get(),
            "вид_на_море": sea_var.get(),
            "район": region_var.get()
        }])
        predicted_log_price = model.predict(flat)[0]
        predicted_price = np.exp(predicted_log_price)
        result_label.config(text=(
                f"Прогнозная стоимость:\n"
                f"{predicted_price:,.0f} ₽"))
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

predict_button = tk.Button(
    root,
    text="Рассчитать стоимость",
    command=predict_price,
    font=("Arial", 14, "bold"),
    bg="#4CAF50",
    fg="white",
    width=28,
    height=2)

predict_button.pack(pady=20)
root.mainloop()