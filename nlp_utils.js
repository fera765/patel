const fs = require('fs');
const path = require('path');
const natural = require('natural');

// --- Carregar Modelos e Dados Auxiliares ---
const modelsDir = path.join(__dirname, 'models');

let tfidfModel = null;
try {
    const tfidfData = fs.readFileSync(path.join(modelsDir, 'tfidf_model.json'), 'utf8');
    tfidfModel = JSON.parse(tfidfData);
    console.log("tfidf_model.json carregado com sucesso.");
} catch (err) {
    console.error("Erro ao carregar tfidf_model.json:", err);
    // Tratar o erro ou lançar para interromper a inicialização se crítico
}

let svmModel = null;
try {
    const svmData = fs.readFileSync(path.join(modelsDir, 'svm_model.json'), 'utf8');
    svmModel = JSON.parse(svmData);
    console.log("svm_model.json carregado com sucesso.");
} catch (err) {
    console.error("Erro ao carregar svm_model.json:", err);
}

let entityDictionaries = null;
try {
    const entityData = fs.readFileSync(path.join(modelsDir, 'entity_dictionaries.json'), 'utf8');
    entityDictionaries = JSON.parse(entityData);
    console.log("entity_dictionaries.json carregado com sucesso.");
} catch (err) {
    console.error("Erro ao carregar entity_dictionaries.json:", err);
}

let portugueseStopwords = [];
try {
    const stopwordsData = fs.readFileSync(path.join(modelsDir, 'portuguese_stopwords.json'), 'utf8');
    portugueseStopwords = JSON.parse(stopwordsData);
    console.log("portuguese_stopwords.json carregado com sucesso.");
} catch (err) {
    console.error("Erro ao carregar portuguese_stopwords.json:", err);
}

const stemmer = new natural.PorterStemmerPt();
console.log("Stemmer (PorterStemmerPt) inicializado.");

// --- Funções de Pré-processamento ---

function preprocessText(text) {
    if (typeof text !== 'string') {
        console.warn("preprocessText: Entrada não é string, retornando array vazio.");
        return [];
    }
    // 1. Lowercase
    text = text.toLowerCase();

    // 2. Remoção de Pontuação (simples)
    text = text.replace(/[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/g, '');

    // 3. Tokenização (simples)
    let tokens = text.split(/\s+/).filter(token => token.length > 0);

    // 4. Remoção de Stopwords
    tokens = tokens.filter(token => !portugueseStopwords.includes(token));

    // 5. Stemming
    tokens = tokens.map(token => stemmer.stem(token));

    return tokens;
}

// --- Cálculo TF-IDF ---
function calculateTfIdf(processedTokens) {
    if (!tfidfModel || !tfidfModel.vocabulary_ || !tfidfModel.idf_) {
        console.error("calculateTfIdf: Modelo TF-IDF não carregado ou inválido.");
        return null;
    }
    if (!Array.isArray(processedTokens)) {
        console.error("calculateTfIdf: Entrada não é um array de tokens.");
        return null;
    }


    const vocab = tfidfModel.vocabulary_;
    const idfValues = tfidfModel.idf_;
    const ngramRange = tfidfModel.ngram_range || [1, 1];
    const sublinearTf = tfidfModel.sublinear_tf || false;
    const normType = tfidfModel.norm || 'l2';

    let ngrams = [];
    for (let n = ngramRange[0]; n <= ngramRange[1]; n++) {
        if (processedTokens.length >= n) { // Garante que há tokens suficientes para o n-grama
            for (let i = 0; i <= processedTokens.length - n; i++) {
                ngrams.push(processedTokens.slice(i, i + n).join(' '));
            }
        }
    }

    const tfVector = new Array(idfValues.length).fill(0);
    for (const ngram of ngrams) {
        if (vocab.hasOwnProperty(ngram)) {
            const termIndex = vocab[ngram];
            tfVector[termIndex]++;
        }
    }

    if (sublinearTf) {
        for (let i = 0; i < tfVector.length; i++) {
            if (tfVector[i] > 0) {
                tfVector[i] = 1 + Math.log(tfVector[i]);
            }
        }
    }

    const tfidfVector = new Array(idfValues.length).fill(0);
    for (let i = 0; i < tfidfVector.length; i++) {
        tfidfVector[i] = tfVector[i] * idfValues[i];
    }

    if (normType === 'l2') {
        let magnitude = 0;
        for (const val of tfidfVector) {
            magnitude += val * val;
        }
        magnitude = Math.sqrt(magnitude);
        if (magnitude > 0) {
            for (let i = 0; i < tfidfVector.length; i++) {
                tfidfVector[i] = tfidfVector[i] / magnitude;
            }
        }
    }
    return tfidfVector;
}

// --- Predição de Intenção SVM ---
function predictIntent(tfidfVector) {
    if (!svmModel || !svmModel.coef_ || !svmModel.intercept_ || !svmModel.classes_) {
        console.error("predictIntent: Modelo SVM não carregado ou inválido.");
        return null;
    }
    if (!tfidfVector || !Array.isArray(tfidfVector) || tfidfVector.length === 0) {
        console.error("predictIntent: Vetor TF-IDF de entrada é inválido ou vazio.");
        // Retornar null ou uma intenção de fallback, dependendo da política de erro.
        // Se o tfidfVector for menor que o esperado, pode dar erro no produto escalar.
        // O vocabulário do TF-IDF tem um tamanho fixo. O tfidfVector deve ter esse tamanho.
        if (!tfidfVector || tfidfVector.length !== (tfidfModel.idf_ ? tfidfModel.idf_.length : -1) ) {
             console.error(`predictIntent: Comprimento do tfidfVector (${tfidfVector ? tfidfVector.length : 'undefined'}) não corresponde ao esperado (${tfidfModel.idf_ ? tfidfModel.idf_.length : 'undefined'}).`);
             return 'fallback_intent_vector_length_mismatch'; // ou null
        }
    }

    const coefficients = svmModel.coef_;
    const intercepts = svmModel.intercept_;
    const classes = svmModel.classes_;
    let maxScore = -Infinity;
    let predictedIntentIndex = -1;

    for (let i = 0; i < classes.length; i++) {
        let score = 0;
        const classCoefficients = coefficients[i];
        // O comprimento do classCoefficients deve ser o mesmo do tfidfVector (tamanho do vocabulário)
        const len = Math.min(tfidfVector.length, classCoefficients.length); // Segurança, mas deveriam ser iguais
        if (tfidfVector.length !== classCoefficients.length) {
            // Isso não deveria acontecer se o tfidfVector foi gerado corretamente com base no vocabulário do tfidfModel
            console.warn(`predictIntent: Discrepância de comprimento para classe ${classes[i]}. TF-IDF: ${tfidfVector.length}, Coefs: ${classCoefficients.length}`);
        }

        for (let j = 0; j < len; j++) {
            score += classCoefficients[j] * tfidfVector[j];
        }
        score += intercepts[i];

        if (score > maxScore) {
            maxScore = score;
            predictedIntentIndex = i;
        }
    }

    if (predictedIntentIndex !== -1) {
        return classes[predictedIntentIndex];
    }
    console.warn("predictIntent: Nenhuma intenção pôde ser predita (predictedIntentIndex === -1).");
    return 'fallback_intent_no_prediction'; // Ou uma intenção padrão de fallback
}

// --- Extração de Entidades ---
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& significa a string toda matched
}

function extractEntities(text) {
    if (!entityDictionaries) {
        console.error("extractEntities: Dicionários de entidades não carregados.");
        return [];
    }
    if (typeof text !== 'string' || text.trim() === '') {
        // console.warn("extractEntities: Texto de entrada é inválido ou vazio.");
        return [];
    }

    const foundEntities = [];
    const textLower = text.toLowerCase();

    for (const entityType in entityDictionaries) {
        if (entityDictionaries.hasOwnProperty(entityType)) {
            const values = entityDictionaries[entityType];
            for (const value of values) {
                const escapedValue = escapeRegExp(value.toLowerCase());
                const regex = new RegExp(`\\b${escapedValue}\\b`, 'g');
                if (textLower.match(regex)) {
                    // Para obter o valor original do texto, precisamos buscar a correspondência no texto original
                    // Isso é mais complexo se houver sobreposição ou múltiplas ocorrências.
                    // Por simplicidade, retornamos o valor do dicionário (que já está em minúsculas).
                    foundEntities.push({ type: entityType, value: value, rawMatchInText: value }); // Simplificado
                }
            }
        }
    }

    const ageRegex = /\b(\d{1,2})\s*(anos)?\b/gi;
    let match;
    while ((match = ageRegex.exec(text)) !== null) {
        foundEntities.push({ type: 'idade', value: parseInt(match[1]), rawMatchInText: match[0] });
    }

    const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
    while ((match = emailRegex.exec(textLower)) !== null) { // Usar textLower para regex de email
        foundEntities.push({ type: 'email', value: match[0], rawMatchInText: match[0] });
    }

    const uniqueEntities = [];
    const seen = new Set();
    for (const entity of foundEntities) {
        const key = `${entity.type}:${entity.value}`;
        if (!seen.has(key)) {
            uniqueEntities.push(entity);
            seen.add(key);
        }
    }
    return uniqueEntities;
}

module.exports = {
    preprocessText,
    calculateTfIdf,
    predictIntent,
    extractEntities,
    // Apenas para fins de teste ou depuração, se necessário:
    // getTfidfModel: () => tfidfModel,
    // getSvmModel: () => svmModel
};

console.log("nlp_utils.js carregado e pronto para uso.");
