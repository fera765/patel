# Lógica de Pré-processamento e Predição para Implementação em NodeJS

Este documento detalha os passos de pré-processamento de texto e a lógica de predição de intenção e extração de entidades, baseados nos modelos Python treinados e exportados (`tfidf_model.json`, `svm_model.json`, `entity_dictionaries.json`).

## A. Pré-processamento de Texto (Replicar em JavaScript)

A seguinte sequência de pré-processamento foi aplicada aos textos de treinamento em Python e DEVE ser replicada em JavaScript para qualquer nova entrada do usuário antes da predição:

1.  **Conversão para Minúsculas:**
    *   Transformar todo o texto da entrada do usuário para letras minúsculas.
    *   Exemplo Python: `text.lower()`

2.  **Remoção de Pontuação:**
    *   Remover todos os caracteres de pontuação padrão (ex: `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`).
    *   Exemplo Python: `text = ''.join([char for char in text if char not in string.punctuation])` (onde `string.punctuation` contém os caracteres a serem removidos).
    *   Em JavaScript, pode-se usar regex: `text.replace(/[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/g, '')`

3.  **Tokenização:**
    *   Dividir o texto (já sem pontuação e em minúsculas) em uma lista de palavras (tokens).
    *   Em Python, foi usado `nltk.word_tokenize(text, language='portuguese')`. Este tokenizador é relativamente simples para o português após a remoção de pontuação, geralmente dividindo por espaços.
    *   Em JavaScript, uma abordagem simples é dividir por espaços: `text.split(/\s+/)`. É importante filtrar tokens vazios que podem surgir de múltiplos espaços.

4.  **Remoção de Stopwords:**
    *   Remover palavras comuns que geralmente não carregam significado semântico para a classificação de intenção (ex: "de", "a", "o", "que", "e").
    *   A lista de stopwords portuguesas usadas no treinamento está no arquivo `portuguese_stopwords.json`. Esta lista deve ser carregada em JavaScript e usada para filtrar os tokens.
    *   Exemplo Python: `tokens = [word for word in tokens if word not in stop_words_pt_set and word.strip()]`

5.  **Stemming:**
    *   Reduzir as palavras à sua raiz ou radical. Em Python, foi usado `nltk.stem.RSLPStemmer()`.
    *   Exemplo Python: `tokens = [stemmer_pt.stem(word) for word in tokens]`
    *   **Desafio em JavaScript:** Encontrar uma biblioteca de stemming RSLP (ou similar para português) pode ser difícil.
        *   **Alternativa 1 (Implementada):** Procurar por uma biblioteca JS que implemente RSLP ou um stemmer de qualidade para português. *Nota: Na implementação de referência (`nlp_utils.js`), foi utilizado o `PorterStemmerPt` da biblioteca `natural`.*
        *   **Alternativa 2 (Mais simples, menor fidelidade):** Não aplicar stemming em JavaScript. Isso significa que o vocabulário no `tfidf_model.json` (que contém palavras stemizadas) pode não corresponder perfeitamente. O impacto na acurácia precisaria ser avaliado. Se o stemming não for aplicado, os termos do vocabulário do TF-IDF devem ser comparados com os tokens não stemizados do usuário.
        *   **Alternativa 3 (Muito simples):** Aplicar remoção de sufixos comuns baseada em regex (ex: remover "ando", "endo", "indo", "ar", "er", "ir"). Isso é rudimentar e menos preciso que RSLP.
    *   Se o stemmer JS escolhido divergir significativamente do RSLP, pode ser necessário reavaliar ou treinar o modelo Python com o mesmo tipo de stemmer (ou sem stemming) para consistência.

**Resultado do Pré-processamento:** Um array de tokens processados (palavras stemizadas e sem stopwords). Ex: `['compr', 'plan', 'saud']`. Este array é a entrada para a etapa de geração de N-gramas.

## B. Cálculo do Vetor TF-IDF para Nova Entrada

Após pré-processar a entrada do usuário conforme a Seção A (resultando em um array de tokens), calcule o vetor TF-IDF usando `tfidf_model.json`:

1.  **Carregar `tfidf_model.json`:**
    *   `vocabulary_`: Dicionário de `termo -> índice`. (Os termos aqui são os n-gramas stemizados e unidos por espaço que foram aprendidos no treino).
    *   `idf_`: Lista de pesos IDF. O índice na lista `idf_` corresponde ao valor do índice no `vocabulary_`.
    *   `ngram_range`: (min_n, max_n) - usado para gerar n-gramas. O modelo foi treinado com `(1,2)`, então bigramas e unigramas são considerados.
    *   `norm`: Tipo de normalização aplicada (geralmente 'l2').
    *   `sublinear_tf`: Booleano, se True, usa `1 + log(tf)` para frequência do termo.

2.  **Gerar N-gramas da Entrada Processada:**
    *   A entrada é o array de tokens da Seção A.
    *   Gere unigramas (palavras individuais) e bigramas (pares de palavras consecutivas) a partir dos tokens. Cada n-grama gerado deve ser uma string com os tokens unidos por espaço.
    *   Exemplo: Se tokens = `["compr", "plan", "saud"]` e `ngram_range = (1,2)`:
        *   Unigramas (strings): `["compr", "plan", "saud"]`
        *   Bigramas (strings): `["compr plan", "plan saud"]`
        *   Lista de N-gramas a serem processados: `["compr", "plan", "saud", "compr plan", "plan saud"]`

3.  **Calcular Frequência dos Termos (TF - Term Frequency):**
    *   Crie um vetor (`tf_vector`) do tamanho do `vocabulary_` (ou seja, `Object.keys(vocabulary_).length`), inicializado com zeros.
    *   Para cada n-grama (string) gerado no passo anterior:
        *   Se o n-grama existir como uma chave no `vocabulary_`:
            *   Obtenha seu índice: `idx = vocabulary_[ngram]`.
            *   Incremente a contagem em `tf_vector[idx]`.

4.  **Aplicar Sublinear TF (se `sublinear_tf` for True no `tfidf_model.json`):**
    *   Se `tfidf_model_data['sublinear_tf']` for verdadeiro:
        *   Para cada valor `count` em `tf_vector` onde `count > 0`, substitua por `1 + log(count)`.

5.  **Calcular TF-IDF:**
    *   Crie um vetor `tfidf_vector` do tamanho do `vocabulary_`.
    *   Para cada índice `i` de 0 ao tamanho do `vocabulary_ - 1`:
        *   `tfidf_vector[i] = tf_vector[i] * idf_[i]` (onde `idf_[i]` é o peso IDF para o termo no índice `i`).

6.  **Normalização L2 (se `norm` for 'l2' no `tfidf_model.json`):**
    *   Calcule a magnitude (norma L2) do `tfidf_vector`: `magnitude = sqrt(sum(val^2 for val in tfidf_vector))`.
    *   Se `magnitude > 0`, normalize o vetor: `tfidf_vector[i] = tfidf_vector[i] / magnitude` para cada `i`.
    *   Este vetor `tfidf_vector` normalizado é a entrada para o modelo SVM.

## C. Predição de Intenção com SVM

Use o `tfidf_vector` da Seção B e os dados de `svm_model.json` para prever a intenção:

1.  **Carregar `svm_model.json`:**
    *   `classes_`: Lista de nomes de classes (intenções). A ordem é importante.
    *   `coef_`: Lista de listas. Cada sub-lista contém os coeficientes para a respectiva classe em `classes_`. O comprimento de cada sub-lista é o tamanho do vocabulário.
    *   `intercept_`: Lista de valores de intercepto, um para cada classe.

2.  **Calcular Scores de Decisão:**
    *   Para cada classe `j` (de 0 ao número de classes - 1):
        *   Obtenha os coeficientes para esta classe: `coeffs_j = coef_[j]`.
        *   Obtenha o intercepto para esta classe: `intercept_j = intercept_[j]`.
        *   Calcule o produto escalar entre o `tfidf_vector` da entrada do usuário e `coeffs_j`.
            `score_j = sum(tfidf_vector[i] * coeffs_j[i] for i in range(len(tfidf_vector))) + intercept_j`.

3.  **Determinar a Intenção Predita:**
    *   A intenção predita é a classe correspondente ao maior `score_j`.
    *   `predicted_intent_index = indexOfMax(scores)`
    *   `predicted_intent_label = classes_[predicted_intent_index]`

## D. Extração de Entidades

A extração de entidades é um processo separado da classificação de intenção e pode ser feita em paralelo ou após.

1.  **Carregar `entity_dictionaries.json`:**
    *   Este arquivo contém um objeto onde cada chave é um `tipo_entidade` (ex: "nome_plano", "tipo_plano") e o valor é uma lista de `valores_entidade` conhecidos (já em minúsculas e ordenados).

2.  **Lógica de Extração:**
    *   **Busca por Dicionário:**
        *   Para cada `tipo_entidade` e sua lista de `valores_entidade` no dicionário:
            *   Itere sobre os `valores_entidade`.
            *   Verifique se o `valor_entidade` (como uma substring completa ou usando limites de palavra `\bvalue\b` com regex) está presente na **entrada original do usuário (antes do pré-processamento extenso como stemming)**. É comum usar a entrada original ou levemente normalizada (minúsculas, talvez remoção de pontuação básica) para a extração de entidades para preservar a forma original das entidades.
            *   Se encontrado, registre a entidade (`tipo_entidade`, `valor_entidade`, e opcionalmente a posição no texto).
    *   **Extração por Regex:**
        *   Defina padrões de regex para entidades como idade, e-mail, etc.
        *   Exemplo (idade): `/\b\d{1,2}\s*(anos)?\b/gi` (aplicar na entrada original ou levemente normalizada).
        *   Se um match for encontrado, registre a entidade.

**Considerações Adicionais para JavaScript:**
*   **Performance:** Para a busca em dicionários de entidades, se as listas de valores forem muito grandes, considere estruturas de dados mais eficientes para busca (ex: `Set` para verificação de existência rápida, ou Aho-Corasick para múltiplas buscas de string). Para o tamanho atual do dicionário, a busca linear simples deve ser aceitável.
*   **Bibliotecas JS:**
    *   Para stemming (se decidido implementar): `natural` (usada na PoC com `PorterStemmerPt`), `stemmer`.
    *   Não há necessidade de bibliotecas pesadas de álgebra linear para o cálculo TF-IDF e predição SVM, pois são operações vetoriais diretas (loops e somas).
*   **Ordem das Operações:** Geralmente, a classificação de intenção é feita primeiro. A extração de entidades pode ser feita independentemente ou ser condicionada pela intenção predita (por exemplo, só procurar por "nome_medicamento" se a intenção for "informacao_medicamento"). Para este projeto, uma extração independente parece ser o ponto de partida.

Este documento deve fornecer um guia claro para a implementação da lógica em um ambiente NodeJS.
