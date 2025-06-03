import json
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import string
import numpy as np

# Baixar recursos do NLTK
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
            # Verificar novamente após o download
            try:
                if resource == 'stopwords':
                    stopwords.words('portuguese')
                elif resource == 'punkt':
                    word_tokenize("teste", language='portuguese')
                elif resource == 'rslp':
                    RSLPStemmer()
            except LookupError:
                print(f"Falha ao carregar {resource} mesmo após tentativa de download.")
                all_downloaded = False
    return all_downloaded

if not download_nltk_resources():
    print("Alguns recursos NLTK não puderam ser baixados/carregados. O script pode falhar.")
    # Considerar sair se recursos críticos como 'punkt' ou 'stopwords' falharem.

# Carregar dados
file_path = 'dataset_planos_saude.json'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Arquivo {file_path} não encontrado. Certifique-se de que ele existe.")
    exit()

dataset = data['dataset']
texts = []
intents = []
entities_data = []

for item in dataset:
    intent = item['intencao']
    for example in item['exemplos_usuario']:
        texts.append(example)
        intents.append(intent)
    if 'entidades' in item and item['entidades']:
        entities_data.append({
            'intent': intent,
            'examples': item['exemplos_usuario'],
            'entities': item['entidades']
        })

df = pd.DataFrame({'text': texts, 'intent': intents})

# Pré-processamento
stop_words_pt = set(stopwords.words('portuguese'))
stemmer_pt = RSLPStemmer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text, language='portuguese')
    tokens = [stemmer_pt.stem(word) for word in tokens if word not in stop_words_pt and word.strip()]
    return ' '.join(tokens)

df['text_processed'] = df['text'].apply(preprocess_text)

X_processed = df['text_processed']
y_labels = df['intent']

# Tratamento para estratificação: garantir que cada classe no conjunto de teste/treino tenha pelo menos 1 membro.
# E que o número de splits (k em k-fold, ou 1/test_size) seja menor ou igual ao número de membros na menor classe.
intent_counts = y_labels.value_counts()
y_labels_unique_count = len(intent_counts)

# Filtrar classes com menos de 2 amostras, pois não podem ser estratificadas em treino/teste
valid_classes = intent_counts[intent_counts >= 2].index
df_filtered = df[df['intent'].isin(valid_classes)]

if len(df_filtered) < len(df):
    print(f"Filtradas {len(df) - len(df_filtered)} amostras de classes com < 2 exemplos.")

X_processed = df_filtered['text_processed']
y_labels = df_filtered['intent']
y_labels_unique_count = len(y_labels.unique()) # Recalcular

if len(X_processed) < 2 or y_labels_unique_count < 1 : # Se não houver dados ou classes suficientes
    print("Não há dados/classes suficientes para treinar após a filtragem. Saindo.")
    # Gerar arquivos vazios ou com mensagens de erro para não quebrar a cadeia de ferramentas
    open('intent_classifier_nb.joblib', 'w').close()
    open('intent_classifier_svm.joblib', 'w').close()
    open('entity_dictionaries.json', 'w').write(json.dumps({}))
    # Imprimir placeholders para as métricas
    print(f"Naive Bayes Acurácia: 0.0000, Tamanho: 0.00 KB")
    print(f"SVM Linear Acurácia: 0.0000, Tamanho: 0.00 KB")
    print(f"\nTamanho do arquivo de dicionários de entidades: 0.00 KB")
    # ... (placeholders para as seções de estimativa de NN e conclusões)
    exit()


test_size_ratio = 0.2
# O número de amostras de teste deve ser >= número de classes para estratificação
# n_samples_test = n_samples * test_size_ratio
# n_samples_test >= y_labels_unique_count  => test_size_ratio >= y_labels_unique_count / n_samples

required_test_size_for_stratify = y_labels_unique_count / len(X_processed) if len(X_processed) > 0 else 1.0

if test_size_ratio < required_test_size_for_stratify and required_test_size_for_stratify < 1.0:
    # Se o test_size padrão (0.2) é muito pequeno para o número de classes,
    # ajustá-lo para o mínimo necessário para estratificação, se esse mínimo for razoável.
    # No entanto, se o required_test_size for muito grande (e.g. >0.5), a estratificação pode não ser ideal.
    # E o conjunto de treino ficaria muito pequeno.
    print(f"Ajustando test_size de {test_size_ratio:.2f} para {required_test_size_for_stratify:.2f} para tentar acomodar {y_labels_unique_count} classes com estratificação.")
    # Não vamos aumentar demais o test_size, pois precisamos de dados para treino.
    # Se required_test_size_for_stratify for muito alto (ex: > 0.4), é melhor não estratificar
    # ou aceitar que algumas classes não estarão no teste.
    if required_test_size_for_stratify > 0.4: # Limite arbitrário
        print("O test_size necessário para estratificação é muito alto. Tentando sem estratificação.")
        stratify_option = None
    else:
        test_size_ratio = required_test_size_for_stratify + 0.01 # Adicionar uma pequena margem
        stratify_option = y_labels

elif len(X_processed) * test_size_ratio < y_labels_unique_count :
     print("Mesmo com ajuste, test_size * n_samples < n_classes. Tentando sem estratificação.")
     stratify_option = None
else:
    stratify_option = y_labels


# Garantir que test_size não seja 1.0 (o que significaria nenhum dado de treino)
if test_size_ratio >= 0.99: # Quase todo o dataset como teste
    test_size_ratio = 0.5 # Reset para um valor mais razoável, e sem estratificação
    stratify_option = None
    print("Test size ratio muito alto, resetando para 0.5 e sem estratificação.")


try:
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_labels, test_size=test_size_ratio, random_state=42, stratify=stratify_option)
except ValueError as e:
    print(f"Erro na divisão treino/teste mesmo após ajustes: {e}. Tentando sem estratificação.")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_labels, test_size=test_size_ratio, random_state=42, stratify=None)
    except ValueError as e2: # Se ainda falhar (ex: dataset muito pequeno)
        print(f"Erro final na divisão treino/teste: {e2}. Treinando com todos os dados e avaliando no treino.")
        X_train, X_test, y_train, y_test = X_processed, X_processed, y_labels, y_labels # Avaliar no próprio treino

# --- Naive Bayes ---
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', MultinomialNB(alpha=0.1))
])
pipeline_nb.fit(X_train, y_train)
y_pred_nb = pipeline_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

model_nb_path = 'intent_classifier_nb.joblib'
joblib.dump(pipeline_nb, model_nb_path)
model_nb_size = os.path.getsize(model_nb_path)

# --- SVM Linear ---
pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LinearSVC(C=1.0, random_state=42, max_iter=3000, dual=True))
])
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

model_svm_path = 'intent_classifier_svm.joblib'
joblib.dump(pipeline_svm, model_svm_path)
model_svm_size = os.path.getsize(model_svm_path)

print(f"Naive Bayes Acurácia: {accuracy_nb:.4f}, Tamanho: {model_nb_size / 1024:.2f} KB")
print(f"SVM Linear Acurácia: {accuracy_svm:.4f}, Tamanho: {model_svm_size / 1024:.2f} KB")

# --- Análise de Entidades (Esboço) ---
print("\n--- Esboço da Estratégia de Extração de Entidades ---")
entity_dictionaries = {}
# (Restante do código de entidades, formatos de exportação e NN permanece o mesmo)
for item in entities_data:
    for entity_info in item['entities']:
        entity_type = entity_info['tipo_entidade']
        entity_value = entity_info['valor_entidade'].lower()
        if entity_type not in entity_dictionaries:
            entity_dictionaries[entity_type] = set()
        entity_dictionaries[entity_type].add(entity_value)

for entity_type in entity_dictionaries:
    entity_dictionaries[entity_type] = list(entity_dictionaries[entity_type])

print("\nDicionários de Entidades (amostra):")
for entity_type, values in list(entity_dictionaries.items())[:2]: # Amostra
    print(f"  {entity_type}: {values[:5]}") # Amostra de valores

dict_path = 'entity_dictionaries.json'
with open(dict_path, 'w', encoding='utf-8') as f:
    json.dump(entity_dictionaries, f, ensure_ascii=False, indent=2)
dict_size = os.path.getsize(dict_path)
print(f"\nTamanho do arquivo de dicionários de entidades: {dict_size / 1024:.2f} KB")

print("\nExemplos de Regex para Entidades:")
print("  - Idade: \\b\\d{1,2}\\s*(anos)?\\b")
print("  - Nome de Plano (padrão): plano\\s+[A-Z_0-9]+")

print("\n--- Formatos de Exportação Leves para NodeJS ---")
print("1. Classificador de Intenções:")
print("   - Naive Bayes: Os parâmetros (probabilidades condicionais e priores) podem ser exportados para JSON.")
print("     - Exemplo: {'class_log_prior_': [...], 'feature_log_prob_': [[...],[...]], 'classes_': [...], 'vocabulary_': {term: index}}")
print("   - SVM Linear: Coeficientes (coef_) e intercepto (intercept_) podem ser exportados para JSON.")
print("     - Exemplo: {'coef_': [[...],[...]], 'intercept_': [...], 'classes_': [...], 'vocabulary_': {term: index}}")
print("   - Compressão (gzip) sobre o JSON pode reduzir significativamente o tamanho.")

print("\n2. Dicionários de Entidades:")
print("   - JSON (como o 'entity_dictionaries.json' gerado) é o formato mais direto e já é leve.")

print("\n3. Regex:")
print("   - As próprias strings de regex podem ser armazenadas em um JSON ou diretamente no código NodeJS.")

print("\n--- Esboço de Rede Neural com TensorFlow.js (Estimativa Teórica) ---")
# Estimativa de vocabulário a partir do TfidfVectorizer já treinado
# Isso é apenas uma aproximação, pois o TFJS pode ter seu próprio pré-processamento/tokenização
# No entanto, o vocabulário do TFIDF nos dá uma ideia do número de features de entrada para a camada de embedding.
vocab_size_approx = len(pipeline_nb.named_steps['tfidf'].vocabulary_)
embedding_dim = 30  # Dimensão de embedding pequena para manter o modelo leve
# Usar X_train para calcular max_seq_length, pois X_processed pode ter sido alterado
# Garantir que X_train não está vazio e que os textos não são vazios
valid_texts_for_length_calc = [s for s in X_train if isinstance(s, str) and len(s.split()) > 0]
if not valid_texts_for_length_calc: # Se não houver textos válidos (e.g. todos filtrados)
    max_seq_length = 0
else:
    max_seq_length = max(len(s.split()) for s in valid_texts_for_length_calc)

num_classes = len(y_labels.unique()) # Usar y_labels que foi filtrado

print(f"Tamanho aproximado do vocabulário (pós-stemming): {vocab_size_approx}")
print(f"Dimensão do Embedding: {embedding_dim}")
print(f"Número de Classes (Intenções): {num_classes}")
print(f"Comprimento máximo da sequência (aprox): {max_seq_length}")

# Arquitetura Simples: Embedding -> GlobalAveragePooling1D -> Dense (output)
# 1. Camada de Embedding:
#    Parâmetros: vocab_size_approx * embedding_dim
embedding_params = vocab_size_approx * embedding_dim
print(f"Parâmetros da Camada de Embedding: {embedding_params}")

# 2. Camada GlobalAveragePooling1D:
#    Sem parâmetros treináveis.

# 3. Camada Densa (Saída):
#    Parâmetros: (embedding_dim * num_classes) + num_classes (bias)
dense_params = (embedding_dim * num_classes) + num_classes
print(f"Parâmetros da Camada Densa: {dense_params}")

total_nn_params = embedding_params + dense_params
print(f"Total de Parâmetros Estimados da Rede Neural: {total_nn_params}")

# Estimativa de Tamanho:
# Assumindo float32 (4 bytes por parâmetro)
size_nn_float32_kb = (total_nn_params * 4) / 1024
print(f"Tamanho Estimado da Rede Neural (float32): {size_nn_float32_kb:.2f} KB")

# Com quantização (e.g., uint8 - 1 byte por peso, bias ainda float32)
size_nn_quantized_kb = ( (embedding_params + (embedding_dim * num_classes)) * 1 + (num_classes * 4) ) / 1024 # Bias float32
print(f"Tamanho Estimado da Rede Neural (quantizada, ~1 byte/parâmetro + bias float32): {size_nn_quantized_kb:.2f} KB")
print("Nota: A quantização no TFJS pode variar, e o overhead do formato do modelo também conta.")

print("\n--- Conclusão Preliminar da Análise ---")
# Usar try-except para o caso de os arquivos não serem criados se o script sair mais cedo
try:
    nb_size_kb = os.path.getsize(model_nb_path) / 1024
    svm_size_kb = os.path.getsize(model_svm_path) / 1024
    dict_s_kb = os.path.getsize(dict_path) / 1024
    print(f"Modelos Clássicos (NB: {nb_size_kb:.2f} KB, SVM: {svm_size_kb:.2f} KB) são muito leves.")
    print(f"Dicionários de Entidades: {dict_s_kb:.2f} KB, também muito leve.")
    print(f"Soma total (NB + SVM + Dicionários): {nb_size_kb + svm_size_kb + dict_s_kb:.2f} KB")
except FileNotFoundError:
    print("Arquivos de modelo/dicionário não foram criados devido a erro anterior.")

print("A rede neural estimada, mesmo quantizada e com embedding_dim=30, ainda pode ser maior que os modelos clássicos,")
print("especialmente se o vocabulário pós-stemming ainda for grande.")
print("A abordagem com modelos clássicos parece mais promissora para o limite de 2MB e simplicidade inicial.")

print("\nScript finalizado.")
