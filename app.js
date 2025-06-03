const express = require('express');
const nlpUtils = require('./nlp_utils');
const conversationManager = require('./conversation_manager'); // Importar o ConversationManager

const app = express();
const port = process.env.PORT || 3000;

// Middleware para parsear o corpo da requisição como JSON
app.use(express.json());

// Endpoint POST /chat
app.post('/chat', (req, res) => {
    const userMessage = req.body.message;
    // Permitir que o cliente envie um userId, ou usar 'defaultUser'
    const userId = req.body.userId || 'defaultUser';

    if (!userMessage || typeof userMessage !== 'string' || userMessage.trim() === '') {
        return res.status(400).json({ error: 'Mensagem inválida. Forneça uma string não vazia no campo "message".' });
    }

    try {
        // 1. Processamento NLP (Intenção e Entidades)
        const processedTokens = nlpUtils.preprocessText(userMessage);
        const tfidfVector = nlpUtils.calculateTfIdf(processedTokens);

        if (!tfidfVector) {
            console.error(`Falha ao calcular TF-IDF para: "${userMessage}" (Tokens: ${processedTokens ? processedTokens.join(',') : 'N/A'})`);
            return res.status(500).json({ error: 'Falha ao calcular o vetor TF-IDF. Verifique os logs do servidor.' });
        }

        const predictedIntent = nlpUtils.predictIntent(tfidfVector);
        const extractedEntities = nlpUtils.extractEntities(userMessage); // Usar a mensagem original

        // 2. Gerenciamento da Conversa e Geração da Resposta da IA
        if (!predictedIntent) { // Se nlp_utils.predictIntent retornar null ou um fallback problemático
             console.warn(`Intenção não pode ser determinada para a mensagem: "${userMessage}". Usando fallback.`);
             // A lógica em conversationManager.getNextResponse já lida com currentIntent nula ou desconhecida.
        }

        const conversationResult = conversationManager.getNextResponse(userId, predictedIntent || 'intent_unknown', extractedEntities);

        // Construir a resposta final para o cliente
        const responseToClient = {
            userInput: userMessage,
            detectedIntent: predictedIntent || 'intent_unknown', // A intenção que o NLP detectou
            extractedEntities: extractedEntities || [],
            iaReply: conversationResult.iaReply,
            conversationState: { // Enviar o estado atualizado da conversa
                userId: conversationResult.userSession.userId,
                currentEtapaFunil: conversationResult.userSession.etapa_funil,
                // O histórico pode ser muito verboso para enviar sempre, mas útil para debug
                // history: conversationResult.userSession.history
            }
            // processedTokens: processedTokens.join(' ') // Para depuração
        };

        console.log(`[${userId}] Mensagem: "${userMessage}" -> Intenção: ${responseToClient.detectedIntent}, Etapa Funil: ${responseToClient.conversationState.currentEtapaFunil}, Resposta IA: "${responseToClient.iaReply}"`);
        res.json(responseToClient);

    } catch (error) {
        console.error(`[${userId}] Erro inesperado no endpoint /chat para mensagem "${userMessage}":`, error);
        res.status(500).json({ error: 'Erro interno no servidor ao processar a mensagem.' });
    }
});

// Endpoint GET /session/:userId para inspecionar o estado da sessão (para depuração)
app.get('/session/:userId', (req, res) => {
    const userId = req.params.userId;
    const session = conversationManager.getUserSession(userId);
    if (session) {
        res.json(session);
    } else {
        // getUserSession agora sempre retorna/cria uma sessão, então isso não deve ser atingido
        res.status(404).json({ error: 'Sessão não encontrada para este userId (isso não deveria acontecer).' });
    }
});


// Endpoint GET /health para verificações de saúde básicas
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'UP', message: 'Serviço de NLP e Conversa está operacional.' });
});

// Iniciar o servidor
app.listen(port, () => {
    console.log(`Servidor rodando na porta ${port}`);
    console.log('Endpoint POST /chat esperando por requisições (Ex: {"message": "Olá", "userId": "user123"})');
    console.log('Endpoint GET /session/:userId disponível para checagem de sessão.');
    console.log('Endpoint GET /health disponível para checagem de status.');
});
