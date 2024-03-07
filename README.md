# :computer:  MLOps UrFU Project

[![Update Ubuntu 22.04](https://github.com/AAEfimov/MLOps_UrFU/actions/workflows/blank.yml/badge.svg)](https://github.com/AAEfimov/MLOps_UrFU/actions/workflows/blank.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/AAEfimov/MLOps_UrFU.svg)](http://isitmaintained.com/project/AAEfimov/MLOps_UrFU "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/AAEfimov/MLOps_UrFU.svg)](http://isitmaintained.com/project/AAEfimov/MLOps_UrFU "Percentage of issues still open")

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/134908631-2f6c75a5-eef8-45b6-ad2d-2f94cac7a83a.png" />
</p>


# Table of Contents
- [Bash pipeline](#bash-pipeline)
- [Jenkins pipeline](#jenkins-pipeline)

# Bash pipeline 
### Лабораторная работа №1 по автоматизации машинного обучения

В данной работе представлен pipeline для решения задачи линей регресси с Kagle.  
Зада заключалась в оценке стоимости домов по предоставленному датасету.  

### Как запустить?

:exclamation: Вам необходимо иметь профиль на  https://www.kaggle.com/  
:exclamation: скачать kaggle.json и разместить в каталоге со скачанными из репозитория файлами  

1) Склонируйте данный репозиторий
```
git clone git@github.com:AAEfimov/MLOps_UrFU.git
cd MLOps_UrFU

```

2) Подготовим окружение
```
python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt
```

3) Запустить скрипт
   
```
chmod a+x pipeline.sh
./pipeline.sh

```

### Результат работы скрипта  

<img src="doc/Final_model.gif" />

# Jenkins pipeline

### Конфигурация 2-х видов pipeline описана в документе Jenkins_pipeline_howto.ipynb и будет доступна на сайте после 17.03.2024
