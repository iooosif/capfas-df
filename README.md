1)git clone https://github.com/iooosif/capfas-df
cd capfas-df
pip install -r requirements.txt

2)Download the Enron Email Dataset from https://www.kaggle.com/datasets/wcukierski/
enron-email-dataset (approximately 430 MB). Place the extracted emails.csv in the
project root directory before running any scripts.

3)python enron_loader.py --target kaminski -v

4)python generate_suspicious.py

5)python capfas.py
