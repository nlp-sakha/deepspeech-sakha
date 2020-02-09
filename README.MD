Клонируем репозиторий
=======================

```bash
cd ~/
git clone https://github.com/nlp-sakha/deepspeech-sakha
```

Потом скачиваем датасет sah.tar.gz из https://voice.mozilla.org/sah/datasets
и распаковывем

```bash
mkdir sah_data
tar -xvf sah.tar.gz -C ~/sah_data
```

В данной инструкции все файлы копируются в трех папках:
 - ~/deepspeech-sakha
 - ~/sah_data
 - ~/DeepSpeech
  

Инструкция по установке DeepSpeech
=======================

Здесь по большей части инструкция собрана с репозитория https://github.com/mozilla/DeepSpeech
Проверялась на Ubuntu, 
на WSL (Windows Subsystem Linux) тоже наверное получится, но без CUDA

## Сначала нужно установить Git и Git Large File Storage

```bash
apt install git git-lfs
git lfs install
```

## Клонируем репозиторий

Будет клонироваться долго как будто завис, надо просто дождаться. Будет весить около 2Гб.
При клонировании указываем версию, т.к. последняя текущая версия часто меняется и не всегда нормально запускается
```bash
cd ~/
git clone --branch v0.6.1 https://github.com/mozilla/DeepSpeech
cd DeepSpeech
```

## Готовим виртуальное окружение для Python

```bash
apt install virtualenv
virtualenv -p python3 $HOME/tmp/deepspeech-venv/
```

## Активируем и заходим в виртуальное окружение Python

```bash
source $HOME/tmp/deepspeech-venv/bin/activate
```

## Устанавливаем deepspeech
Это уже скомпилированная версия. Вроде, используется только для тестирования результата обучения

```bash
pip3 install deepspeech
```

## Устанавливаем зависимости
```bash
pip3 install -r requirements.txt
pip3 install $(python3 util/taskcluster.py --decoder)
```

Если будут ошибки, их по отдельности пытаться установить
Например, для ошибки с webrtcvad, устанавливаем дополнительно модули и потом повторяем установку зависимостей

```bash
apt update
apt install python3 python-dev python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev python-pip
```

## Подготовка датасета

Теперь готовим наш датасет для обучения с нашим алфавитом (в нашем примере скачанный датасет находится ~/sah_data, а алфавит на ~/deepspeech-sakha)
```bash
bin/import_cv2.py --filter_alphabet ~/deepspeech-sakha/alphabet_sah.txt ~/sah_data/
```

После этого внутри папки ~/sah_data/clips/ будут созданы файлы train.csv, dev.csv, test.csv и mp3 будут конвертированы в wav

Если при этом будут выходить ошибки "This install of SoX cannot process .mp3 files", то надо установить дополнительные модули, например:

```bash
apt install sox libsox-fmt-mp3
pip3 install sox requests
```


Использование NVIDIA CUDA (возможно устарело)
=======================

Для удобства установим настроенный докер контейнер https://github.com/NVIDIA/nvidia-docker

и там у меня на шаге sudo apt-get update возникала ошибка GPG Key и его смог исправить командой
```bash
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | sudo apt-key add -
```
далее на последней команде тоже возникала ошибка `Unknown runtime specified nvidia`
его устанавливаем командой
```bash
sudo apt-get install nvidia-container-runtime
```

Внутри ~/DeepSpeech/native_client
```bash
 deepspeech --model models/output_graph.pb --audio audio/2830-3980-0043.wav --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie
 ```
./DeepSpeech.py --train_files ../sah_data/clips/train.csv --test_files ../sah_data/clips/test.csv --epochs 10


```cuda install
https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070

```
audio check on model
```bash
deepspeech --model ~/sah_data/sakhaspeech/output_graph.pb --alphabet ~/sah_data/sakhaspeech/alphabet_sah.txt --lm data/lm/lm.binary --trie models/trie --audio ~/sah_data/sakhaspeech/test.wav
```

Подготовка среды GPU

```
virtualenv -p python3 $HOME/tmp/deepspeech-gpu-venv/
pip3 install -r requirements.txt
pip3 install $(python3 util/taskcluster.py --decoder)
pip3 install 'tensorflow-gpu==1.14.0'
```

Обучение на нашем датасете
=======================

## Переходим на папку ~/DeepSpeech и активируем deepspeech-venv

```bash
cd ~/DeepSpeech
source $HOME/tmp/deepspeech-venv/bin/activate
```

## Команды для обучения

Далее само обучение должно запускаться командой
```bash
./DeepSpeech.py --train_files ~/sah_data/clips/train.csv --dev_files ~/sah_data/clips/dev.csv --test_files ~/sah_data/clips/test.csv
```

но мы из него убираем --dev_files, т.к. у нас dev.csv пустой и он будет выдавать ошибку

```bash
./DeepSpeech.py --train_files ~/sah_data/clips/train.csv --test_files ~/sah_data/clips/test.csv
```

Можно дополнительные команды указать для обучения

```
./DeepSpeech.py --train_files ../sah_data/clips/train.csv --test_files ../sah_data/clips/test.csv --epochs 10 --checkpoint_dir ../checkpoints --train_batch_size 20 --test_batch_size 48 --alphabet_config_path ../deepspeech-sakha/alphabet_sah.txt
```

## Команда для проверки результата обучения

Тут не очень понятно, похоже, какую-то свою обученную модель использует

```
cd DeepSpeech/native_client
deepspeech --model models/output_graph.pb --audio audio/2830-3980-0043.wav --alphabet ../sah_6h_2019-06-12/alphabet_sah.txt --lm models/lm.binary --trie models/trie
```

Прочее
=======================

Так же в загружено несколько примеров запуска с разными параметрами и их результаты файлы run(1-5).txt
