// TypeScript Client Example for Function Calling with VertexAI

export interface FunctionDefinition {
    name: string;
    description: string;
    parameters: {
        type: "object";
        properties: Record<string, {
            type: string;
            description?: string;
        }>;
        required?: string[];
    };
}

export interface LLMResponse {
    response: string;
    provider: string;
    usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
    };
    model?: string;
    function_call?: {
        name: string;
        arguments: Record<string, any>;
        result: string;
    };
}

export class LLMService {
    private baseUrl: string;
    
    constructor(baseUrl: string = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    /**
     * Generate response with optional function calling
     */
    async generateResponse(
        prompt: string,
        model: string,
        history: Array<{ role: string; content: string }> = [],
        provider: string = 'vertexai',
        functions?: FunctionDefinition[],
        personaId?: string,
        userId?: string,
        webhookUrl?: string
    ): Promise<LLMResponse> {
        const response = await fetch(`${this.baseUrl}/api/llm/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt,
                model,
                history,
                llm_provider: provider,
                functions,
                persona_id: personaId,
                user_id: userId,
                webhook_url: webhookUrl
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
    }
}

// Usage Examples

async function example1_SendPersonaImage() {
    const llmService = new LLMService('http://localhost:8000');
    
    // Define the sendPersonaImage function
    const functions: FunctionDefinition[] = [
        {
            name: "sendPersonaImage",
            description: "Sends persona images from database to the user",
            parameters: {
                type: "object",
                properties: {
                    personaId: {
                        type: "string",
                        description: "The ID of the persona whose images to retrieve"
                    },
                    limit: {
                        type: "number",
                        description: "Maximum number of images to return (default: 5)"
                    }
                },
                required: ["personaId"]
            }
        }
    ];
    
    // Generate with VertexAI
    const response = await llmService.generateResponse(
        "Please send me images of persona xyz-123",
        "models/gemini-2.0-flash",
        [],
        "vertexai",
        functions,
        "xyz-123",  // personaId
        "user-456",  // userId
        "http://localhost:3000/webhooks/ai-gateway/database-query"  // webhookUrl
    );
    
    console.log('Response:', response.response);
    
    if (response.function_call) {
        console.log('Function called:', response.function_call.name);
        console.log('Arguments:', response.function_call.arguments);
        console.log('Result:', response.function_call.result);
    }
}

async function example2_Calculator() {
    const llmService = new LLMService('http://localhost:8000');
    
    const functions: FunctionDefinition[] = [
        {
            name: "calculator",
            description: "Performs mathematical calculations",
            parameters: {
                type: "object",
                properties: {
                    expression: {
                        type: "string",
                        description: "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
                    }
                },
                required: ["expression"]
            }
        }
    ];
    
    const response = await llmService.generateResponse(
        "What is sqrt(144) + 10?",
        "models/gemini-2.0-flash",
        [],
        "vertexai",
        functions
    );
    
    console.log('Response:', response.response);
}

async function example3_MultipleProviders() {
    const llmService = new LLMService('http://localhost:8000');
    
    const functions: FunctionDefinition[] = [
        {
            name: "calculator",
            description: "Performs math calculations",
            parameters: {
                type: "object",
                properties: {
                    expression: { type: "string" }
                },
                required: ["expression"]
            }
        }
    ];
    
    // Try with different providers
    const providers = [
        { name: 'vertexai', model: 'models/gemini-2.0-flash' },
        { name: 'openai', model: 'gpt-3.5-turbo' },
        { name: 'gemini', model: 'models/gemini-2.5-flash' }
    ];
    
    for (const { name, model } of providers) {
        try {
            console.log(`\nTesting with ${name}...`);
            const response = await llmService.generateResponse(
                "Calculate 25 * 4",
                model,
                [],
                name,
                functions
            );
            console.log(`${name} response:`, response.response);
        } catch (error) {
            console.error(`${name} error:`, error);
        }
    }
}

async function example4_ChatHistory() {
    const llmService = new LLMService('http://localhost:8000');
    
    const functions: FunctionDefinition[] = [
        {
            name: "sendPersonaImage",
            description: "Retrieves persona images",
            parameters: {
                type: "object",
                properties: {
                    personaId: { type: "string" }
                },
                required: ["personaId"]
            }
        }
    ];
    
    const history = [
        { role: "user", content: "Hi, I'm looking for some images" },
        { role: "assistant", content: "I can help you with that! Which persona's images would you like to see?" },
        { role: "user", content: "Show me images for persona abc-123" }
    ];
    
    const response = await llmService.generateResponse(
        "Show me images for persona abc-123",
        "models/gemini-2.0-flash",
        history,
        "vertexai",
        functions,
        "abc-123",
        "user-789",
        "http://localhost:3000/webhooks/ai-gateway/database-query"
    );
    
    console.log('Response:', response.response);
}

// Run examples
if (require.main === module) {
    (async () => {
        console.log('🚀 Function Calling Examples\n');
        
        try {
            console.log('Example 1: Send Persona Image');
            console.log('=' .repeat(60));
            await example1_SendPersonaImage();
            
            console.log('\n\nExample 2: Calculator');
            console.log('='.repeat(60));
            await example2_Calculator();
            
            console.log('\n\nExample 3: Multiple Providers');
            console.log('='.repeat(60));
            await example3_MultipleProviders();
            
            console.log('\n\nExample 4: With Chat History');
            console.log('='.repeat(60));
            await example4_ChatHistory();
            
        } catch (error) {
            console.error('Error:', error);
        }
    })();
}

export default LLMService;
