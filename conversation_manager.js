const fs = require('fs');
const path = require('path');

// --- Carregar Dataset ---
const modelsDir = path.join(__dirname, 'models');
let dataset = [];
let sequentialQuoteResponses = {}; // Para armazenar as respostas sequenciais
try {
    const datasetData = fs.readFileSync(path.join(modelsDir, 'dataset_planos_saude.json'), 'utf8');
    const fullDataset = JSON.parse(datasetData);
    dataset = fullDataset.dataset;
    // Carregar as respostas sequenciais de cotação da intenção 'informar_dado_para_cotacao'
    const coletaIntent = dataset.find(entry => entry.intencao === 'informar_dado_para_cotacao');
    if (coletaIntent && coletaIntent.respostas_sequenciais_cotacao) {
        sequentialQuoteResponses = coletaIntent.respostas_sequenciais_cotacao;
    } else {
        console.error("Estrutura 'respostas_sequenciais_cotacao' não encontrada no dataset para a intenção 'informar_dado_para_cotacao'.");
    }
    console.log("dataset_planos_saude.json carregado para o ConversationManager.");
} catch (err) {
    console.error("Erro ao carregar dataset_planos_saude.json:", err);
}

// --- Gerenciamento de Estado da Conversa ---
const userSessionStore = {};
const QUOTE_STEPS = ['ask_nome', 'ask_idade', 'ask_cidade', 'ask_dependentes_sim_nao', 'ask_dependentes_detalhes', 'finalizar_cotacao'];

function getUserSession(userId = 'defaultUser') {
    if (!userSessionStore[userId]) {
        userSessionStore[userId] = {
            userId: userId,
            etapa_funil: 'ToFu',
            currentCollectionStep: null, // Ex: 'ask_nome', 'ask_idade', etc.
            collectedQuoteData: {
                nome_usuario: null,
                idade: null,
                cidade_cotacao: null,
                tem_dependentes: null, // 'sim' ou 'nao'
                info_dependentes: null
            },
            history: []
        };
        // console.log(`Nova sessão criada para ${userId}, etapa inicial: ToFu`);
    }
    return userSessionStore[userId];
}

function updateUserSession(userId = 'defaultUser', updates) {
    const session = getUserSession(userId);
    const oldEtapa = session.etapa_funil;
    const oldStep = session.currentCollectionStep;

    if (updates.hasOwnProperty('etapa_funil')) {
        session.etapa_funil = updates.etapa_funil;
    }
    if (updates.hasOwnProperty('currentCollectionStep')) {
        session.currentCollectionStep = updates.currentCollectionStep;
    }
    if (updates.hasOwnProperty('collectedQuoteData')) {
        session.collectedQuoteData = { ...session.collectedQuoteData, ...updates.collectedQuoteData };
    }
     if (updates.hasOwnProperty('history_intent')) { // Para logar a intenção que levou à mudança
        session.history.push({
            intent: updates.history_intent,
            oldEtapa: oldEtapa,
            newEtapa: session.etapa_funil,
            oldStep: oldStep,
            newStep: session.currentCollectionStep,
            quoteDataSnapshot: JSON.parse(JSON.stringify(session.collectedQuoteData)),
            timestamp: new Date().toISOString()
        });
        if(session.history.length > 10) session.history.shift();
    }
    // console.log(`Sessão ${userId} atualizada: Etapa ${oldEtapa}->${session.etapa_funil}, Step ${oldStep}->${session.currentCollectionStep}`);
    return session;
}

function resetQuoteFlow(userId = 'defaultUser', intentForHistory = 'reset_flow') {
    // console.log(`Resetando fluxo de cotação para usuário ${userId}.`);
    updateUserSession(userId, {
        etapa_funil: 'BoFu', // Volta para o início do BoFu para cotação
        currentCollectionStep: 'ask_nome', // Começa pedindo o nome
        collectedQuoteData: {
            nome_usuario: null,
            idade: null,
            cidade_cotacao: null,
            tem_dependentes: null,
            info_dependentes: null
        },
        history_intent: intentForHistory
    });
    return sequentialQuoteResponses.ask_nome || "Vamos começar sua cotação. Qual seu nome completo?";
}


// --- Lógica de Resposta e Transição de Funil ---

function findRawResponse(intent, userSession) {
    const currentEtapaFunil = userSession.etapa_funil;
    const currentStep = userSession.currentCollectionStep;

    // 1. Fluxo de Coleta de Cotação Sequencial
    if (currentEtapaFunil === 'BoFu' && currentStep && currentStep !== 'finalizar_cotacao') {
        // A intenção aqui pode ser 'informar_dado_para_cotacao' ou o usuário pode ter dito algo que
        // o NLP classificou diferente. Independentemente da intenção, se estamos num passo de coleta,
        // pegamos a próxima pergunta da sequência.
        let nextStepLogic = QUOTE_STEPS[QUOTE_STEPS.indexOf(currentStep)]; // Próxima pergunta baseada no passo atual
        let responseText = sequentialQuoteResponses[nextStepLogic];

        if (currentStep === 'ask_dependentes_detalhes' && userSession.collectedQuoteData.tem_dependentes === 'nao') {
            // Pulou este passo, vai para finalizar
            responseText = sequentialQuoteResponses[userSession.collectedQuoteData.tem_dependentes === 'sim' ? 'final_com_dependentes' : 'final_sem_dependentes'];
            updateUserSession(userSession.userId, { currentCollectionStep: 'finalizar_cotacao', history_intent: intent });
            return { resposta_ia: responseText, next_etapa_funil: 'BoFu_CotacaoConcluida', is_final_step: true };
        }

        if (!responseText) {
            console.error(`Texto de resposta não encontrado para o passo de coleta: ${nextStepLogic}`);
            responseText = "Desculpe, houve um problema em nosso fluxo. Poderia tentar novamente?";
        }
        // Não há mudança de etapa do funil aqui ainda, só do passo de coleta (feito em getNextResponse)
        return { resposta_ia: responseText, next_etapa_funil: currentEtapaFunil, is_final_step: false };
    }

    // 2. Intenções Gerais e FAQs
    const entry = dataset.find(e => e.intencao === intent);
    if (entry) {
        // Se for FAQ, a etapa do funil da entrada do dataset é 'FAQ'.
        // Se for outra intenção, a etapa do funil da entrada do dataset pode sugerir uma transição.
        const nextEtapa = (entry.etapa_funil === 'FAQ') ? currentEtapaFunil : entry.etapa_funil;
        return { resposta_ia: entry.resposta_ia, next_etapa_funil: nextEtapa, is_final_step: false };
    }

    // 3. Fallbacks por Intenção (se não houver entrada no dataset)
    const fallbackResponsesByIntent = {
        'saudacao': "Olá! Como posso te ajudar com planos de saúde hoje?",
        'despedida': "Até logo! Se precisar de mais alguma coisa, é só chamar.",
        'agradecimento': "De nada! Fico feliz em ajudar.",
    };
    if (fallbackResponsesByIntent[intent]) {
        return { resposta_ia: fallbackResponsesByIntent[intent], next_etapa_funil: currentEtapaFunil, is_final_step: false };
    }

    console.warn(`Nenhuma entrada no dataset ou fallback para intenção "${intent}".`);
    return { resposta_ia: "Não entendi muito bem. Pode tentar reformular?", next_etapa_funil: currentEtapaFunil, is_final_step: false };
}


function formatResponse(responseText, userSession) {
    if (typeof responseText !== 'string') responseText = "Desculpe, não tenho uma resposta para isso no momento.";
    let formattedText = responseText;
    const dataToFormat = userSession.collectedQuoteData;

    for (const key in dataToFormat) {
        if (dataToFormat[key]) { // Apenas se o dado existir
            const placeholder = `[${key.toLowerCase()}]`; // ex: [nome_usuario]
            try {
                // Usar regex global para substituir todas as ocorrências do placeholder
                formattedText = formattedText.replace(new RegExp(escapeRegExp(placeholder), 'gi'), dataToFormat[key]);
            } catch (e) {
                console.error(`Erro ao tentar substituir placeholder: ${placeholder} com valor ${dataToFormat[key]}`, e);
            }
        }
    }
    // Limpar quaisquer placeholders restantes que não foram preenchidos
    formattedText = formattedText.replace(/\[\w+\]/g, '(informação pendente)');
    return formattedText;
}

function escapeRegExp(string) {
  if (typeof string !== 'string') return '';
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function determineNextEtapaFunil(currentIntent, currentEtapaFunil, nextEtapaSugeridaPelaResposta, isFinalStepInFlow) {
    if (isFinalStepInFlow) {
        return 'BoFu_CotacaoConcluida';
    }
    // Se estivermos no meio da coleta de cotação, permanecemos em BoFu
    if (currentEtapaFunil === 'BoFu' && getUserSession().currentCollectionStep && getUserSession().currentCollectionStep !== 'finalizar_cotacao') {
        return 'BoFu';
    }

    const etapasOrdenadas = ['ToFu', 'MoFu', 'BoFu', 'BoFu_CotacaoConcluida'];
    const indiceAtual = etapasOrdenadas.indexOf(currentEtapaFunil);
    const indiceSugerido = etapasOrdenadas.indexOf(nextEtapaSugeridaPelaResposta);

    if (indiceSugerido > indiceAtual) {
        return nextEtapaSugeridaPelaResposta;
    }

    return currentEtapaFunil; // Padrão: manter etapa atual
}

function getNextResponse(userId = 'defaultUser', currentIntent, extractedEntities = [], userInputText = "") {
    const userSession = getUserSession(userId);
    let currentEtapaFunil = userSession.etapa_funil;
    let currentStep = userSession.currentCollectionStep;
    let collectedDataUpdates = {};

    // Se o usuário pedir para iniciar cotação, reseta e começa o fluxo.
    if (currentIntent === 'solicitar_cotacao_plano') {
        const initialQuestion = resetQuoteFlow(userId, currentIntent);
        // A sessão já foi atualizada por resetQuoteFlow para o primeiro passo.
        return { iaReply: formatResponse(initialQuestion, userSession), userSession: userSession };
    }

    // Se estiver no fluxo de cotação (BoFu e um currentStep está definido)
    if (currentEtapaFunil === 'BoFu' && currentStep && currentStep !== 'finalizar_cotacao') {
        // Processar a resposta do usuário para o passo atual
        switch (currentStep) {
            case 'ask_nome':
                // Tenta pegar de entidade, senão usa input direto. Nome é mais difícil para entidade genérica.
                const nomeEnt = extractedEntities.find(e => e.type === 'nome_usuario');
                collectedDataUpdates.nome_usuario = nomeEnt ? nomeEnt.value : userInputText;
                currentStep = 'ask_idade';
                break;
            case 'ask_idade':
                const idadeEnt = extractedEntities.find(e => e.type === 'idade');
                if (idadeEnt) collectedDataUpdates.idade = idadeEnt.value;
                else { /* Tentar parsear idade do userInputText se necessário */ }
                currentStep = 'ask_cidade';
                break;
            case 'ask_cidade':
                const cidadeEnt = extractedEntities.find(e => e.type === 'cidade_cotacao');
                if (cidadeEnt) collectedDataUpdates.cidade_cotacao = cidadeEnt.value;
                else { /* Tentar parsear cidade do userInputText se necessário */ }
                currentStep = 'ask_dependentes_sim_nao';
                break;
            case 'ask_dependentes_sim_nao':
                const respDep = userInputText.toLowerCase();
                if (respDep.includes('sim') || respDep.includes('s')) {
                    collectedDataUpdates.tem_dependentes = 'sim';
                    currentStep = 'ask_dependentes_detalhes';
                } else {
                    collectedDataUpdates.tem_dependentes = 'nao';
                    currentStep = 'finalizar_cotacao'; // Pula para o final
                }
                break;
            case 'ask_dependentes_detalhes':
                // Armazena a informação de dependentes como fornecida.
                collectedDataUpdates.info_dependentes = userInputText;
                currentStep = 'finalizar_cotacao';
                break;
        }
        updateUserSession(userId, { collectedQuoteData: collectedDataUpdates, currentCollectionStep: currentStep, history_intent: currentIntent });
    }

    // Lógica para estado 'BoFu_CotacaoConcluida'
    if (currentEtapaFunil === 'BoFu_CotacaoConcluida') {
        if (currentIntent === 'saudacao' || currentIntent === 'despedida' || currentIntent === 'agradecimento') {
            // Permite interações genéricas
        } else {
             const prevNome = userSession.collectedQuoteData.nome_usuario || "Olá";
             updateUserSession(userId, {history_intent: currentIntent});
             return {
                iaReply: formatResponse(`${prevNome}! Sua cotação anterior já foi processada. Nossos consultores devem entrar em contato em breve. Posso ajudar com mais alguma dúvida geral ou FAQ?`, userSession),
                userSession: userSession
            };
        }
    }

    // Obter a resposta da IA baseada na intenção ou no passo atual da coleta
    const { resposta_ia: rawResponseText, next_etapa_funil: etapaSugeridaPelaResposta, is_final_step } = findRawResponse(currentIntent, userSession);

    // Formatar a resposta com os dados coletados (se houver placeholders)
    const formattedReply = formatResponse(rawResponseText, userSession); // Passa a sessão inteira para ter acesso a collectedQuoteData

    // Determinar a próxima etapa do funil
    const newEtapaFunil = determineNextEtapaFunil(currentIntent, currentEtapaFunil, etapaSugeridaPelaResposta, is_final_step);

    updateUserSession(userId, { etapa_funil: newEtapaFunil, history_intent: currentIntent });

    return {
        iaReply: formattedReply,
        userSession: userSession
    };
}

module.exports = {
    getNextResponse,
    getUserSession
};

console.log("conversation_manager.js carregado e pronto (v3 - fluxo de cotação sequencial).");
