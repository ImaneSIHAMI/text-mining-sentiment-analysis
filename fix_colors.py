content = open('src/05_evaluation.py', encoding='utf-8').read()
content = content.replace(
    '"Twitter-RoBERTa":"#534AB7"}',
    '"Twitter-RoBERTa":"#534AB7"}'
)
old = 'COLORS = {"Naïve Bayes":"#378ADD","SVM":"#1D9E75",\n          "Random Forest":"#EF9F27","LSTM":"#D4537E","BERT":"#534AB7"}'
new = 'COLORS = {"Naïve Bayes":"#378ADD","SVM":"#1D9E75",\n          "Random Forest":"#EF9F27","LSTM":"#D4537E","BERT":"#534AB7","Twitter-RoBERTa":"#2E8B57"}'
content = content.replace(old, new)
open('src/05_evaluation.py', 'w', encoding='utf-8').write(content)
print('Corrige!')
