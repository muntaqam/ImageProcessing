#!/bin/bash
python task1.py --input_path images/t1 --output outputs/task1.png
python task2.py --input_path images/t2 --output outputs/task2.png --json ./task2.json
python task2.py --input_path images/BonusImages --output outputs/bonus.png --json ./bonus.json

python utils.py --ubit $1