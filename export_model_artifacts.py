import json
import os
import re
import joblib # Ainda usado para carregar o pipeline se necessário, mas o foco é exportar para JSON
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline # Para treinar o modelo final
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import string
import numpy as np

# --- Configuração e Funções de Pré-processamento (como na subtarefa anterior) ---
def download_nltk_resources():
    resources_to_download = ['punkt', 'stopwords', 'rslp']
    all_downloaded = True
    for resource in resources_to_download:
        try:
            if resource == 'stopwords':
                stopwords.words('portuguese')
            elif resource == 'punkt':
                word_tokenize("teste", language='portuguese')
            elif resource == 'rslp':
                RSLPStemmer()
        except LookupError:
            print(f"Baixando recurso NLTK: {resource}")
            nltk.download(resource, quiet=True)
            try: # Verificar novamente
                if resource == 'stopwords': stopwords.words('portuguese')
                elif resource == 'punkt': word_tokenize("teste", language='portuguese')
                elif resource == 'rslp': RSLPStemmer()
            except LookupError:
                print(f"Falha ao carregar {resource} mesmo após tentativa de download.")
                all_downloaded = False
    return all_downloaded

if not download_nltk_resources():
    print("Alguns recursos NLTK não puderam ser baixados/carregados. Saindo.")
    exit()

# Carregar dados do JSON original
file_path = 'dataset_planos_saude.json'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Arquivo {file_path} não encontrado.")
    exit()

dataset = data['dataset']
texts = []
intents = []
entities_data = [] # Para o dicionário de entidades

for item in dataset:
    intent = item['intencao']
    for example in item['exemplos_usuario']:
        texts.append(example)
        intents.append(intent)
    if 'entidades' in item and item['entidades']: # Coletar dados para o dicionário de entidades
        entities_data.append({
            'intent': intent,
            'examples': item['exemplos_usuario'],
            'entities': item['entidades']
        })

df = pd.DataFrame({'text': texts, 'intent': intents})

# Pré-processamento
stop_words_pt_list = stopwords.words('portuguese') # Guardar a lista para documentação
stop_words_pt_set = set(stop_words_pt_list)
stemmer_pt = RSLPStemmer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text, language='portuguese')
    tokens = [stemmer_pt.stem(word) for word in tokens if word not in stop_words_pt_set and word.strip()]
    return ' '.join(tokens)

df['text_processed'] = df['text'].apply(preprocess_text)
X_processed = df['text_processed']
y_labels = df['intent']

# --- Treinar TF-IDF + SVM Linear COM TODOS OS DADOS ---
print("Treinando modelo final com todos os dados...")
# Usar os mesmos parâmetros que se mostraram melhores ou padrão na análise anterior
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
svm_classifier = LinearSVC(C=1.0, random_state=42, max_iter=3000, dual=True) # dual=True por n_samples > n_features

# Criar e treinar o pipeline
pipeline_final = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', svm_classifier)
])
pipeline_final.fit(X_processed, y_labels)
print("Modelo treinado.")

# Extrair componentes treinados
fitted_tfidf = pipeline_final.named_steps['tfidf']
fitted_svm = pipeline_final.named_steps['clf']

# --- Exportar Parâmetros do TF-IDF para JSON ---
tfidf_model_data = {
    'vocabulary_': fitted_tfidf.vocabulary_, # dict: term -> index
    'idf_': fitted_tfidf.idf_.tolist(),       # numpy array -> list
    'ngram_range': fitted_tfidf.ngram_range, # tupla (min_n, max_n)
    # Outros parâmetros importantes do TfidfVectorizer para replicar o comportamento:
    'lowercase': fitted_tfidf.lowercase,
    'stop_words': None, # Já aplicamos stopwords antes, então o TF-IDF não precisa.
    'use_idf': fitted_tfidf.use_idf,
    'smooth_idf': fitted_tfidf.smooth_idf,
    'sublinear_tf': fitted_tfidf.sublinear_tf,
    'norm': fitted_tfidf.norm # geralmente 'l2'
}
tfidf_output_path = 'tfidf_model.json'
with open(tfidf_output_path, 'w', encoding='utf-8') as f:
    json.dump(tfidf_model_data, f, ensure_ascii=False, indent=2)
print(f"Parâmetros do TF-IDF exportados para: {tfidf_output_path} ({os.path.getsize(tfidf_output_path)/1024:.2f} KB)")

# --- Exportar Parâmetros do SVM para JSON ---
svm_model_data = {
    'classes_': fitted_svm.classes_.tolist(), # numpy array -> list
    'coef_': fitted_svm.coef_.tolist(),       # numpy array -> list of lists
    'intercept_': fitted_svm.intercept_.tolist() # numpy array -> list
    # 'C', 'max_iter' etc. são hiperparâmetros de treino, não necessários para predição com coef_ e intercept_
}
svm_output_path = 'svm_model.json'
with open(svm_output_path, 'w', encoding='utf-8') as f:
    json.dump(svm_model_data, f, ensure_ascii=False, indent=2)
print(f"Parâmetros do SVM exportados para: {svm_output_path} ({os.path.getsize(svm_output_path)/1024:.2f} KB)")


# --- Consolidar/Gerar Dicionário de Entidades ---
# (Reutilizando a lógica da subtarefa anterior para garantir que está atualizado)
entity_dictionaries = {}
for item in entities_data: # entities_data foi populado ao carregar o dataset
    for entity_info in item['entities']:
        entity_type = entity_info['tipo_entidade']
        entity_value = entity_info['valor_entidade'].lower() # Normalizar
        if entity_type not in entity_dictionaries:
            entity_dictionaries[entity_type] = set()
        entity_dictionaries[entity_type].add(entity_value)

for entity_type in entity_dictionaries: # Converter sets para listas para JSON
    entity_dictionaries[entity_type] = sorted(list(entity_dictionaries[entity_type])) # Ordenar para consistência

entity_dict_path = 'entity_dictionaries.json'
with open(entity_dict_path, 'w', encoding='utf-8') as f:
    json.dump(entity_dictionaries, f, ensure_ascii=False, indent=2)
print(f"Dicionário de entidades consolidado em: {entity_dict_path} ({os.path.getsize(entity_dict_path)/1024:.2f} KB)")


# --- Gerar arquivo de stopwords portuguesas para referência ---
stopwords_path = 'portuguese_stopwords.json'
with open(stopwords_path, 'w', encoding='utf-8') as f:
    json.dump(sorted(stop_words_pt_list), f, ensure_ascii=False, indent=2)
print(f"Lista de stopwords exportada para: {stopwords_path}")


print("\n--- Criação do arquivo preprocessing_logic.md ---")
# O conteúdo do .md será definido no próximo passo, usando uma chamada de ferramenta separada.
# Por agora, apenas confirmamos que os artefatos do modelo foram gerados.

print("\nScript de exportação finalizado.")
print("Próximo passo: criar o arquivo preprocessing_logic.md.")
