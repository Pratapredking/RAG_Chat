// Import required modules from LangChain ecosystem
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai" // For creating vector embeddings from text using Gemini
import { ChatGoogleGenerativeAI } from "@langchain/google-genai" // Google's Gemini AI model
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages" // Message types for conversations
import {
  ChatPromptTemplate,      // For creating structured prompts with placeholders
  MessagesPlaceholder,     // Placeholder for dynamic message history
} from "@langchain/core/prompts"
import { StateGraph } from "@langchain/langgraph"              // State-based workflow orchestration
import { Annotation } from "@langchain/langgraph"              // Type annotations for state management
import { tool } from "@langchain/core/tools" // For creating custom tools/functions
import { ToolNode } from "@langchain/langgraph/prebuilt"       // Pre-built node for executing tools
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb" // For saving conversation state
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"   // Vector search integration with MongoDB
import { MongoClient } from "mongodb"                          // MongoDB database client
import { z } from "zod"                                        // Schema validation library
import "dotenv/config"                                         // Load environment variables from .env file

// Detect temporary Gemini/API failures that are safe to retry
function isRetryableGeminiError(error: any): boolean {
  const status = error?.status ?? error?.response?.status
  const message = String(error?.message ?? error ?? "").toLowerCase()

  return (
    status === 429 ||
    status === 500 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    message.includes("[429") ||
    message.includes("[500") ||
    message.includes("[502") ||
    message.includes("[503") ||
    message.includes("[504") ||
    message.includes("rate limit") ||
    message.includes("resource exhausted") ||
    message.includes("service unavailable") ||
    message.includes("high demand") ||
    message.includes("fetch failed") ||
    message.includes("econnreset") ||
    message.includes("etimedout")
  )
}

// Retry temporary Gemini failures with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxAttempts = 3,
  label = "Gemini"
): Promise<T> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn()
    } catch (error: any) {
      if (!isRetryableGeminiError(error) || attempt === maxAttempts) {
        throw error
      }

      // 1s, 2s, 4s... with a 30-second cap
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 30000)
      console.warn(
        `${label} temporary failure (attempt ${attempt}/${maxAttempts}). ` +
        `Retrying in ${delay / 1000}s...`
      )
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }

  throw new Error(`${label} retry attempts exhausted`)
}

// Main function that creates and runs the AI agent
export async function callAgent(client: MongoClient, query: string, thread_id: string) {
  try {
    // Database configuration
    const dbName = "inventory_database"        // Name of the MongoDB database
    const db = client.db(dbName)              // Get database instance
    const collection = db.collection("items") // Get the 'items' collection

    // Define the state structure for the agent workflow
    const GraphState = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        // Reducer function: how to combine old and new messages
        reducer: (x, y) => x.concat(y), // Simply concatenate old messages (x) with new messages (y)
      }),
    })

    // Define the tool input schema separately to avoid excessively deep TypeScript inference
    const itemLookupSchema = z.object({
      query: z.string().describe("The search query"),
      n: z.number().optional().default(10)
        .describe("Number of results to return"),
    })

    type ItemLookupInput = z.infer<typeof itemLookupSchema>

    // Break LangChain/Zod generic inference at the tool() boundary to avoid TS2589
    const createTool: any = tool

    // Create a custom tool for searching furniture inventory
    const itemLookupTool = createTool(
      // Explicit input/output types keep LangChain/Zod generic inference bounded
      async ({ query, n = 10 }: ItemLookupInput): Promise<string> => {
        try {
          console.log("Item lookup tool called with query:", query)

          // Check if database has any data at all
          const totalCount = await collection.countDocuments()
          console.log(`Total documents in collection: ${totalCount}`)

          // Early return if database is empty
          if (totalCount === 0) {
            console.log("Collection is empty")
            return JSON.stringify({ 
              error: "No items found in inventory", 
              message: "The inventory database appears to be empty",
              count: 0 
            })
          }

          // Get sample documents for debugging purposes
          const sampleDocs = await collection.find({}).limit(3).toArray()
          console.log("Sample documents:", sampleDocs)

          // Configuration for MongoDB Atlas Vector Search
          const dbConfig = {
            collection: collection,           // MongoDB collection to search
            indexName: "vector_index",       // Name of the vector search index
            textKey: "embedding_text",       // Field containing the text used for embeddings
            embeddingKey: "embedding",       // Field containing the vector embeddings
          }

          // Create vector store instance for semantic search using Google Gemini embeddings
          const vectorStore = new MongoDBAtlasVectorSearch(
            new GoogleGenerativeAIEmbeddings({
              apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment
              model: process.env.GEMINI_EMBEDDING_MODEL ?? "gemini-embedding-001", // text-embedding-004 was retired
            }),
            dbConfig
          )

          // Text-search fallback. This keeps the chatbot usable while embeddings
          // are being re-generated or the Atlas vector index is being rebuilt.
          async function runTextSearch() {
            const stopWords = new Set([
              "the", "and", "for", "with", "have", "any", "you", "your",
              "are", "can", "show", "please", "looking", "need", "want", "do"
            ])

            const terms = query
              .toLowerCase()
              .split(/[^a-z0-9]+/)
              .filter(term => term.length >= 3 && !stopWords.has(term))

            const escapeRegex = (value: string) =>
              value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

            const pattern = terms.length > 0
              ? terms.map(escapeRegex).join("|")
              : escapeRegex(query)

            const textResults = await collection.find({
              $or: [
                { item_name: { $regex: pattern, $options: "i" } },
                { item_description: { $regex: pattern, $options: "i" } },
                { categories: { $regex: pattern, $options: "i" } },
                { embedding_text: { $regex: pattern, $options: "i" } }
              ]
            }).limit(n).toArray()

            console.log(`Text search returned ${textResults.length} results`)

            return JSON.stringify({
              results: textResults,
              searchType: "text",
              query,
              count: textResults.length
            })
          }

          try {
            console.log("Performing vector search...")
            const result = await vectorStore.similaritySearchWithScore(query, n)
            console.log(`Vector search returned ${result.length} results`)

            if (result.length === 0) {
              console.log("Vector search returned no results; falling back to text search...")
              return await runTextSearch()
            }

            return JSON.stringify({
              results: result,
              searchType: "vector",
              query,
              count: result.length
            })
          } catch (vectorError: any) {
            console.warn(
              "Vector search failed; falling back to text search:",
              vectorError?.message ?? vectorError
            )
            return await runTextSearch()
          }
          
        } catch (error: any) {
          // Log detailed error information for debugging
          console.error("Error in item lookup:", error)
          console.error("Error details:", {
            message: error.message,
            stack: error.stack,
            name: error.name
          })
          
          // Return error information as JSON string
          return JSON.stringify({ 
            error: "Failed to search inventory", 
            details: error.message,
            query: query
          })
        }
      },
      // Tool metadata and schema definition
      {
        name: "item_lookup",                                    // Tool name that the AI will reference
        description: "Gathers furniture item details from the Inventory database", // Description for the AI
        schema: itemLookupSchema,                                // Reuse explicitly typed input schema
      }
    )

    // Keep the tool collection shallowly typed to avoid recursive LangChain generic expansion
    const tools: any[] = [itemLookupTool]

    // Let ToolNode infer its own state type; passing GraphState here can trigger TS2589
    const toolNode = new ToolNode(tools)

    // Initialize Gemini models.
    // Gemini 2.5 is used here to stay compatible with the current LangChain Google integration,
    // which does not preserve Gemini 3 thought_signature metadata across tool calls.
    // Keep these shallowly typed to avoid deep LangChain generic inference (TS2589).
    const primaryModel: any = new ChatGoogleGenerativeAI({
      model: process.env.GEMINI_CHAT_MODEL ?? "gemini-2.5-flash",
      temperature: 0,
      maxRetries: 0,                     // We handle retries explicitly below
      apiKey: process.env.GOOGLE_API_KEY,
    }).bindTools(tools)

    const fallbackModel: any = new ChatGoogleGenerativeAI({
      model: process.env.GEMINI_FALLBACK_MODEL ?? "gemini-2.5-flash-lite",
      temperature: 0,
      maxRetries: 0,
      apiKey: process.env.GOOGLE_API_KEY,
    }).bindTools(tools)

    // Try the primary model first. If Google remains temporarily unavailable
    // after retries (for example HTTP 503/high demand), use the fallback model.
    async function invokeGeminiWithFallback(messages: BaseMessage[]) {
      try {
        return await retryWithBackoff(
          () => primaryModel.invoke(messages),
          3,
          "gemini-2.5-flash"
        )
      } catch (primaryError: any) {
        if (!isRetryableGeminiError(primaryError)) {
          throw primaryError
        }

        console.warn(
          "Primary Gemini model is temporarily unavailable. " +
          "Falling back to gemini-2.5-flash-lite."
        )

        return retryWithBackoff(
          () => fallbackModel.invoke(messages),
          2,
          "gemini-2.5-flash-lite"
        )
      }
    }

    // Decision function: determines next step in the workflow
    function shouldContinue(state: typeof GraphState.State): "tools" | "__end__" {
      const messages = state.messages                               // Get all messages
      const lastMessage = messages[messages.length - 1] as AIMessage // Get the most recent message

      // If the AI wants to use tools, go to tools node; otherwise end
      if (lastMessage.tool_calls?.length) {
        return "tools"  // Route to tool execution
      }
      return "__end__"  // End the workflow
    }

    // Function that calls the AI model with retry + fallback logic
    async function callModel(state: typeof GraphState.State) {
      // Create a structured prompt template
      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system", // System message defines the AI's role and behavior
          `You are a helpful E-commerce Chatbot Agent for a furniture store. 

IMPORTANT: You have access to an item_lookup tool that searches the furniture inventory database. ALWAYS use this tool when customers ask about furniture items, even if the tool returns errors or empty results.

When using the item_lookup tool:
- If it returns results, provide helpful details about the furniture items
- If it returns an error or no results, acknowledge this and offer to help in other ways
- If the database appears to be empty, let the customer know that inventory might be being updated

Current time: {time}`,
        ],
        new MessagesPlaceholder("messages"), // Placeholder for conversation history
      ])

      // Fill in the prompt template with actual values
      const formattedPrompt = await prompt.formatMessages({
        time: new Date().toISOString(), // Current timestamp
        messages: state.messages,       // All previous messages
      })

      // Primary model -> retry -> fallback model -> retry
      const result = await invokeGeminiWithFallback(formattedPrompt)

      // Return new state with the AI's response added
      return { messages: [result] }
    }

    // Build the workflow graph
    const workflow = new StateGraph(GraphState)
      .addNode("agent", callModel)                    // Add AI model node
      .addNode("tools", toolNode)                     // Add tool execution node
      .addEdge("__start__", "agent")                  // Start workflow at agent
      .addConditionalEdges("agent", shouldContinue, ["tools", "__end__"]) // Agent decides: tools or end
      .addEdge("tools", "agent")                      // After tools, go back to agent

    // Initialize conversation state persistence
    const checkpointer = new MongoDBSaver({ client, dbName })
    // Compile the workflow with state saving
    const app = workflow.compile({ checkpointer })

    // Execute the workflow
    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(query)], // Start with user's question
      },
      { 
        recursionLimit: 15,                   // Prevent infinite loops
        configurable: { thread_id: thread_id } // Conversation thread identifier
      }
    )

    // Extract the final response from the conversation
    const response = finalState.messages[finalState.messages.length - 1].content
    console.log("Agent response:", response)

    return response // Return the AI's final response

  } catch (error: any) {
    // Handle different types of errors with user-friendly messages
    console.error("Error in callAgent:", error.message)
    
    const errorMessage = String(error?.message ?? error)

    if (
      error?.status === 429 ||
      errorMessage.includes("[429") ||
      errorMessage.toLowerCase().includes("rate limit") ||
      errorMessage.toLowerCase().includes("resource exhausted")
    ) {
      throw new Error("Gemini rate limit reached after retries. Please try again shortly.")
    } else if (
      error?.status === 503 ||
      errorMessage.includes("[503") ||
      errorMessage.toLowerCase().includes("high demand") ||
      errorMessage.toLowerCase().includes("service unavailable")
    ) {
      throw new Error("Gemini is temporarily unavailable after primary and fallback retries. Please try again shortly.")
    } else if (error?.status === 401 || errorMessage.includes("[401")) {
      throw new Error("Authentication failed. Please check your GOOGLE_API_KEY configuration.")
    } else {
      throw new Error(`Agent failed: ${errorMessage}`)
    }
  }
}
