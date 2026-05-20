# nonlinear-real-estate-regression
Проект представляет собой обучаемую модель, которая на основании не линейной регрессии строит прогноз цены объекта для конечного потребителя на основании реализованных кейсов
# Модель не линейной регресии для оценки стоимости объектов недвижимости

ML-powered desktop application for nonlinear real estate price estimation.

## Features

* Nonlinear Ridge Regression
* Polynomial feature generation
* Automatic anomaly detection
* Iterative dataset cleaning
* Coefficient analysis
* GUI interface for managers
* Excel export
* Desktop executable support

## Technologies

* Python
* Scikit-learn
* Pandas
* Tkinter

## Model Features

### Numeric Features

* Area
* Rooms
* Floor
* Building age
* Distance to city center
* Distance to metro
* Nearby schools
* Crime index

### Binary Features

* Parking
* Elevator
* Sea view

### Categorical Features

* District

## Run

```bash
Программа с интерфейсом.exe
or
Программа с интерфейсом.py
```

## Output

The application generates:

* anomaly reports
* coefficient analysis
* trained regression model
* valuation predictions

## Notes

The model uses logarithmic target transformation and nonlinear polynomial features to improve regression quality.
