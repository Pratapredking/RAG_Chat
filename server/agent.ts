// Import required modules from LangChain ecosystem
import { ChatGoogleGenerativeAI } from "@langchain/google-genai" // Google's Gemini AI model
import { HumanMessage, SystemMessage } from "@langchain/core/messages" // Message types for conversations
import { DynamicStructuredTool } from "@langchain/core/tools" // For creating custom tools/functions
import type { JSONSchema } from "@langchain/core/utils/json_schema"
import { GeminiEmbeddings } from "./gemini-embeddings"
import { createReactAgent } from "@langchain/langgraph/prebuilt" // Pre-built tool-calling agent workflow
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb" // For saving conversation state
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"   // Vector search integration with MongoDB
import { MongoClient } from "mongodb"                          // MongoDB database client
import "dotenv/config"                                         // Load environment variables from .env file

type ItemLookupInput = {
  query: string
  n?: number
}

// Utility function to handle API rate limits with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,    // The function to retry (generic type T for return value)
  maxRetries = 3           // Maximum number of retry attempts (default 3)
): Promise<T> {
  // Loop through retry attempts
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn()    // Try to execute the function
    } catch (error: any) {
      const status = error.status ?? error.response?.status
      const networkCode = error.code ?? error.cause?.code
      const isTransient = status === 429
        || status >= 500
        || error.name === "TypeError"
        || error.name === "AbortError"
        || error.message?.includes("fetch failed")
        || ["ABORT_ERR", "ECONNRESET", "ENOTFOUND", "EAI_AGAIN", "ETIMEDOUT"].includes(networkCode)

      // Retry rate limits and temporary provider/network failures.
      if (isTransient && attempt < maxRetries) {
        // Calculate exponential backoff delay: 2^attempt seconds, max 30 seconds
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000)
        console.log(`Temporary AI service error. Retrying in ${delay/1000} seconds...`)
        // Wait for the calculated delay before retrying
        await new Promise(resolve => setTimeout(resolve, delay))
        continue // Go to next iteration (retry)
      }
      throw error // If not rate limit or out of retries, throw the error
    }
  }
  throw new Error("Max retries exceeded") // This should never be reached
}

// Main function that creates and runs the AI agent
export async function callAgent(client: MongoClient, query: string, thread_id: string) {
  try {
    // Database configuration
    const dbName = "inventory_database"        // Name of the MongoDB database
    const db = client.db(dbName)              // Get database instance
    const collection = db.collection("items") // Get the 'items' collection

    // Create a custom tool for searching furniture inventory
    const itemLookupSchema: JSONSchema = {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query",
        },
        n: {
          type: "number",
          default: 10,
          description: "Number of results to return",
        },
      },
      required: ["query"],
      additionalProperties: false,
    }

    const itemLookupTool = new DynamicStructuredTool<
      JSONSchema,
      ItemLookupInput,
      ItemLookupInput,
      string
    >({
      // The actual function that will be executed when tool is called
      func: async ({ query, n = 10 }) => {
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
          const sampleDocs = await collection.find({}, {
            projection: { item_id: 1, item_name: 1 },
          }).limit(3).toArray()
          console.log("Sample documents:", sampleDocs)

          // Configuration for MongoDB Atlas Vector Search
          const dbConfig = {
            collection: collection,           // MongoDB collection to search
            indexName: "vector_index",       // Name of the vector search index
            textKey: "embedding_text",       // Field containing the text used for embeddings
            embeddingKey: "embedding",       // Field containing the vector embeddings
          }

          try {
            // Create vector store instance for semantic search using Gemini embeddings
            const vectorStore = new MongoDBAtlasVectorSearch(
              new GeminiEmbeddings({
                apiKey: process.env.GOOGLE_API_KEY,
              }),
              dbConfig
            )

            console.log("Performing vector search...")
            const result = await vectorStore.similaritySearchWithScore(query, n)
            console.log(`Vector search returned ${result.length} results`)

            if (result.length > 0) {
              return JSON.stringify({
                results: result,
                searchType: "vector",
                query,
                count: result.length,
              })
            }
          } catch (vectorError: any) {
            // Existing databases may still contain vectors from the retired
            // text-embedding-004 model. Keep inventory usable until reseeded.
            console.warn("Vector search failed; falling back to text search:", vectorError.message)
          }

          console.log("Trying text search...")
          const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
          let textResults = await collection.find({
            $or: [
              { item_name: { $regex: escapedQuery, $options: "i" } },
              { item_description: { $regex: escapedQuery, $options: "i" } },
              { categories: { $regex: escapedQuery, $options: "i" } },
              { embedding_text: { $regex: escapedQuery, $options: "i" } },
            ],
          }).limit(n).toArray()

          // Natural-language tool queries often contain plural forms (for
          // example "dining tables") while inventory names are singular.
          // If the exact phrase misses, require every meaningful token and
          // allow an optional trailing "s" for a small, predictable fallback.
          if (textResults.length === 0) {
            const tokens = query.toLowerCase().match(/[a-z0-9]+/g)
              ?.filter((token) => token.length > 2)
              .map((token) => token.endsWith("s") ? token.slice(0, -1) : token)
              .filter((token, index, all) => all.indexOf(token) === index)
              ?? []

            if (tokens.length > 0) {
              textResults = await collection.find({
                $and: tokens.map((token) => {
                  const escapedToken = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
                  const tokenPattern = `${escapedToken}s?`
                  return {
                    $or: [
                      { item_name: { $regex: tokenPattern, $options: "i" } },
                      { item_description: { $regex: tokenPattern, $options: "i" } },
                      { categories: { $regex: tokenPattern, $options: "i" } },
                      { embedding_text: { $regex: tokenPattern, $options: "i" } },
                    ],
                  }
                }),
              }).limit(n).toArray()
            }
          }

          console.log(`Text search returned ${textResults.length} results`)
          return JSON.stringify({
            results: textResults,
            searchType: "text",
            query,
            count: textResults.length,
          })
          
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
      name: "item_lookup",                                    // Tool name that the AI will reference
      description: "Gathers furniture item details from the Inventory database", // Description for the AI
      schema: itemLookupSchema,
    })

    // Array of all available tools (just one in this case)
    const tools = [itemLookupTool]
    // Initialize the AI model (Google's Gemini)
    const model = new ChatGoogleGenerativeAI({
      model: process.env.GEMINI_CHAT_MODEL || "gemini-2.5-flash",
      temperature: 0,                    // Deterministic responses (no randomness)
      maxRetries: 0,                     // Disable built-in retries (we handle our own)
      apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment
    })

    // Initialize conversation state persistence
    const checkpointer = new MongoDBSaver({ client, dbName })
    // Build the standard tool-calling workflow with state saving. Using the
    // prebuilt agent avoids TypeScript's excessively-deep generic inference in
    // a manually assembled StateGraph while keeping the same runtime behavior.
    const app = createReactAgent({
      llm: model,
      tools,
      checkpointSaver: checkpointer,
      prompt: (state) => [
        new SystemMessage(`You are a helpful E-commerce Chatbot Agent for a furniture store.

IMPORTANT: You have access to an item_lookup tool that searches the furniture inventory database. ALWAYS use this tool when customers ask about furniture items, even if the tool returns errors or empty results.

When using the item_lookup tool:
- If it returns results, provide helpful details about the furniture items
- If it returns an error or no results, acknowledge this and offer to help in other ways
- If the database appears to be empty, let the customer know that inventory might be being updated

Current time: ${new Date().toISOString()}`),
        ...state.messages,
      ],
    })

    // Execute the workflow
    const finalState = await retryWithBackoff(() => app.invoke(
      {
        messages: [new HumanMessage(query)], // Start with user's question
      },
      {
        recursionLimit: 15,                    // Prevent infinite loops
        configurable: { thread_id: thread_id }, // Conversation thread identifier
        signal: AbortSignal.timeout(20_000),    // Bound stalled provider requests
      }
    ))

    // Extract the final response from the conversation
    const response = finalState.messages[finalState.messages.length - 1].content
    console.log("Agent response:", response)

    return response // Return the AI's final response

  } catch (error: any) {
    // Handle different types of errors with user-friendly messages
    console.error("Error in callAgent:", error.message)
    
    if (error.status === 429) { // Rate limit error
      throw new Error("Service temporarily unavailable due to rate limits. Please try again in a minute.")
    } else if (error.status === 401) { // Authentication error
      throw new Error("Authentication failed. Please check your API configuration.")
    } else { // Generic error
      throw new Error(`Agent failed: ${error.message}`)
    }
  }
}
