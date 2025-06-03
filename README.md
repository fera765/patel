# Chatbot de Planos de Saúde - Prova de Conceito

## 1. Visão Geral do Projeto

Este projeto é uma prova de conceito (PoC) de um chatbot para planos de saúde. O objetivo é demonstrar um pipeline funcional de Processamento de Linguagem Natural (NLP) e gerenciamento de conversas capaz de:

*   Entender as intenções do usuário relacionadas a planos de saúde.
*   Extrair entidades relevantes das mensagens do usuário.
*   Gerenciar um diálogo simples, guiando o usuário através de um funil de vendas (ToFu, MoFu, BoFu).
*   Coletar informações de forma interativa para uma cotação de plano de saúde.
*   Responder a Perguntas Frequentes (FAQs).

A aplicação é construída em Node.js com um backend Express.js e um cliente de chat HTML simples para interação.

## 2. Estrutura do Projeto

```
.
├── examples/
│   └── chat_client.html       # Cliente HTML simples para testar o chat
├── models/
│   ├── dataset_planos_saude.json # Dataset principal com intenções, exemplos e respostas
│   ├── entity_dictionaries.json # Dicionários de entidades para extração
│   ├── portuguese_stopwords.json # Lista de stopwords em português
│   ├── svm_model.json           # Parâmetros exportados do classificador SVM
│   └── tfidf_model.json         # Parâmetros exportados do vetorizador TF-IDF
├── .gitignore                 # Arquivos e pastas a serem ignorados pelo Git
├── app.js                     # Ponto de entrada da aplicação Node.js (servidor Express)
├── conversation_manager.js    # Lógica de gerenciamento de diálogo e estado da conversa
├── nlp_utils.js               # Utilitários de NLP (pré-processamento, predição)
├── package.json               # Metadados do projeto Node.js e dependências
├── preprocessing_logic.md     # Documentação da lógica de NLP para replicação
└── README.md                  # Este arquivo
```

## 3. Tecnologias Utilizadas

*   **Backend:**
    *   Node.js
    *   Express.js (para o servidor API)
    *   `natural` (biblioteca JS para NLP, usada aqui para stemming em português)
*   **Modelagem de IA (Python - para treinamento e exportação):**
    *   Scikit-learn (para TF-IDF e SVM Linear)
    *   NLTK (para pré-processamento de texto em Python)
    *   Pandas
*   **Frontend (Exemplo):**
    *   HTML5
    *   JavaScript (vanilla, com `fetch` API)
    *   CSS (inline, básico)
*   **Formato de Dados/Modelos:**
    *   JSON

## 4. Descrição dos Modelos de IA (arquivos em `models/`)

Os modelos de IA foram treinados em Python usando Scikit-learn e depois exportados para JSON para serem utilizados pela aplicação Node.js.

*   **`dataset_planos_saude.json`:**
    *   Contém o corpus principal para o chatbot.
    *   Estruturado com intenções (`intencao`), exemplos de frases de usuário (`exemplos_usuario`), respostas da IA (`resposta_ia`), etapa do funil (`etapa_funil`), e entidades (`entidades`).
    *   Para o fluxo de cotação, inclui `respostas_sequenciais_cotacao` para guiar a coleta de dados.
*   **`tfidf_model.json`:**
    *   Armazena os parâmetros do `TfidfVectorizer` treinado.
    *   Inclui:
        *   `vocabulary_`: Dicionário de termos (palavras e n-gramas) e seus índices.
        *   `idf_`: Pesos IDF (Inverse Document Frequency) para cada termo do vocabulário.
        *   Outros parâmetros de configuração como `ngram_range`, `norm`, `sublinear_tf`.
*   **`svm_model.json`:**
    *   Armazena os parâmetros do classificador `LinearSVC` (Support Vector Machine) treinado.
    *   Inclui:
        *   `classes_`: Lista das intenções que o modelo pode prever.
        *   `coef_`: Coeficientes do hiperplano para cada classe/intenção.
        *   `intercept_`: Termos de intercepto para cada classe/intenção.
*   **`entity_dictionaries.json`:**
    *   Dicionários usados para extração de entidades baseada em lookup.
    *   Ex: mapeia tipos de entidade (como `nome_plano`) para uma lista de valores conhecidos.
*   **`portuguese_stopwords.json`:**
    *   Lista de stopwords em português usadas durante o pré-processamento.

A lógica de como usar `tfidf_model.json` e `svm_model.json` para predição está documentada em `preprocessing_logic.md`.

## 5. Instruções de Configuração do Ambiente

1.  **Instalar Node.js:** Certifique-se de ter o Node.js (versão 14.x ou superior recomendada) e o npm instalados.
2.  **Clonar o Repositório (se aplicável):**
    ```bash
    # git clone <url_do_repositorio>
    # cd <pasta_do_projeto>
    ```
3.  **Instalar Dependências:** Navegue até a pasta raiz do projeto e execute:
    ```bash
    npm install
    ```
    Isso instalará `express` e `natural` (e suas dependências) listadas no `package.json`.

## 6. Instruções de Como Executar a Aplicação

1.  Após a configuração do ambiente e instalação das dependências, execute o seguinte comando na pasta raiz do projeto:
    ```bash
    node app.js
    ```
2.  Se tudo ocorrer bem, você verá mensagens no console indicando que os modelos foram carregados e que o servidor está rodando, geralmente na porta 3000.
    ```
    tfidf_model.json carregado com sucesso.
    svm_model.json carregado com sucesso.
    entity_dictionaries.json carregado com sucesso.
    portuguese_stopwords.json carregado com sucesso.
    Stemmer (PorterStemmerPt) inicializado.
    nlp_utils.js carregado e pronto para uso.
    dataset_planos_saude.json carregado para o ConversationManager.
    conversation_manager.js carregado e pronto (v3 - fluxo de cotação sequencial).
    Servidor rodando na porta 3000
    Endpoint POST /chat esperando por requisições (Ex: {"message": "Olá", "userId": "user123"})
    Endpoint GET /session/:userId disponível para checagem de sessão.
    Endpoint GET /health disponível para checagem de status.
    ```

## 7. Instruções de Como Usar o Exemplo `examples/chat_client.html`

1.  Certifique-se de que a aplicação Node.js (`app.js`) esteja em execução (conforme o passo anterior).
2.  Abra o arquivo `examples/chat_client.html` diretamente em um navegador web moderno (Chrome, Firefox, Edge, etc.).
    *   Você pode fazer isso clicando duas vezes no arquivo ou usando "Abrir com..." no seu gerenciador de arquivos.
3.  Uma interface de chat simples será exibida.
4.  Digite suas mensagens na caixa de texto e clique em "Enviar" ou pressione Enter.
5.  As respostas da IA, a intenção detectada, entidades e a etapa atual do funil serão exibidas na interface.
6.  Cada vez que você atualizar a página do `chat_client.html`, uma nova `userId` de teste será gerada, simulando uma nova sessão de usuário. Para continuar a mesma sessão, interaja sem atualizar a página.

## 8. Descrição dos Endpoints da API

A aplicação Node.js expõe os seguintes endpoints:

*   **`POST /chat`**
    *   **Descrição:** Endpoint principal para interagir com o chatbot.
    *   **Corpo da Requisição (JSON):**
        ```json
        {
            "message": "Texto da mensagem do usuário",
            "userId": "identificador_opcional_do_usuario"
        }
        ```
        *   `message` (string, obrigatório): A mensagem enviada pelo usuário.
        *   `userId` (string, opcional): Um identificador para o usuário. Se não fornecido, um `userId` padrão ('defaultUser') é usado internamente para gerenciar o estado da conversa.
    *   **Resposta de Sucesso (JSON, status 200):**
        ```json
        {
            "userInput": "Texto original do usuário",
            "detectedIntent": "intenção_classificada_pelo_nlp",
            "extractedEntities": [
                { "type": "tipo_entidade", "value": "valor_entidade", "rawMatchInText": "texto_original_match" }
            ],
            "iaReply": "Resposta da IA para o usuário.",
            "conversationState": {
                "userId": "identificador_do_usuario",
                "currentEtapaFunil": "etapa_atual_no_funil_de_vendas"
            }
        }
        ```
    *   **Resposta de Erro (JSON):**
        *   Status `400` para requisições inválidas (ex: mensagem faltando).
        *   Status `500` para erros internos do servidor.

*   **`GET /session/:userId`**
    *   **Descrição:** Endpoint de depuração para inspecionar o estado atual da sessão de um usuário específico.
    *   **Parâmetro de URL:**
        *   `:userId` (string): O identificador do usuário cuja sessão você deseja verificar.
    *   **Resposta de Sucesso (JSON, status 200):**
        ```json
        {
            "userId": "identificador_do_usuario",
            "etapa_funil": "etapa_atual_no_funil",
            "currentCollectionStep": "passo_atual_na_coleta_de_dados_cotacao", // ou null
            "collectedQuoteData": {
                "nome_usuario": null, // ou valor coletado
                "idade": null,        // ou valor coletado
                "cidade_cotacao": null, // ou valor coletado
                "tem_dependentes": null, // ou 'sim'/'nao'
                "info_dependentes": null // ou valor coletado
            },
            "history": [ /* últimas interações */ ]
        }
        ```

*   **`GET /health`**
    *   **Descrição:** Endpoint para verificação de saúde da aplicação.
    *   **Resposta de Sucesso (JSON, status 200):**
        ```json
        {
            "status": "UP",
            "message": "Serviço de NLP e Conversa está operacional."
        }
        ```

## 9. Próximos Passos e Melhorias Futuras

Para detalhes sobre os próximos passos e melhorias planejadas, consulte o arquivo [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md).

---
**Nota:** Esta PoC foca na demonstração da lógica de NLP e conversação. Para um ambiente de produção, várias melhorias seriam necessárias (ver `FUTURE_IMPROVEMENTS.md`).
