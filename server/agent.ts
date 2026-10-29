// Import required modules from LangChain ecosystem
import { HumanMessage, SystemMessage } from "@langchain/core/messages" // Message types for conversations
import { DynamicStructuredTool } from "@langchain/core/tools" // For creating custom tools/functions
import type { JSONSchema } from "@langchain/core/utils/json_schema"
import { createReactAgent } from "@langchain/langgraph/prebuilt" // Pre-built tool-calling agent workflow
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb" // For saving conversation state
import { MongoClient } from "mongodb"                          // MongoDB database client
import { PDF_COLLECTION_NAME, searchPdfChunks } from "./pdf-rag"
import { searchInventoryItems } from "./inventory-search"
import {
  FallbackChatGoogleGenerativeAI,
  getGeminiChatModelNames,
} from "./fallback-chat-model"
import "dotenv/config"                                         // Load environment variables from .env file

type ItemLookupInput = {
  query: string
  n?: number
}

type PdfSearchInput = {
  query: string
  document_id?: string
  n?: number
}

type PdfCitation = {
  fileName: string
  pageNumber: number
}

function appendPdfCitations(response: string, citations: PdfCitation[]): string {
  if (citations.length === 0 || /\*\*Source:[\s\S]*?Page:\s*\d+\*\*/i.test(response)) {
    return response
  }

  const uniqueCitations = Array.from(new Map(
    citations.map((citation) => [
      `${citation.fileName}\u0000${citation.pageNumber}`,
      citation,
    ])
  ).values())

  const sourceLines = uniqueCitations
    .map(({ fileName, pageNumber }) => `**Source: ${fileName} - Page: ${pageNumber}**`)
    .join("\n")

  return `${response.trim()}\n\n${sourceLines}`
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
    const pdfCitations: PdfCitation[] = []

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

          const results = await searchInventoryItems(collection, query, n)
          console.log(`Hybrid search returned ${results.length} results`)
          return JSON.stringify({
            results,
            searchType: "hybrid",
            query,
            count: results.length,
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

    const pdfSearchSchema: JSONSchema = {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Question or concept to search for in the indexed PDFs",
        },
        document_id: {
          type: "string",
          description: "Optional document ID used to restrict search to one PDF",
        },
        n: {
          type: "number",
          default: 5,
          description: "Number of PDF chunks to return",
        },
      },
      required: ["query"],
      additionalProperties: false,
    }

    const pdfSearchTool = new DynamicStructuredTool<
      JSONSchema,
      PdfSearchInput,
      PdfSearchInput,
      string
    >({
      name: "pdf_search",
      description: "Searches indexed PDF documents and returns relevant text with filename and page citations",
      schema: pdfSearchSchema,
      func: async ({ query, document_id, n = 5 }) => {
        try {
          const pdfCount = await db.collection(PDF_COLLECTION_NAME).countDocuments()
          if (pdfCount === 0) {
            return JSON.stringify({
              query,
              count: 0,
              message: "No PDF documents have been indexed yet",
            })
          }

          const results = await searchPdfChunks(client, query, {
            documentId: document_id,
            limit: n,
          })

          for (const result of results) {
            if (Number.isInteger(result.page_number) && (result.page_number as number) > 0) {
              pdfCitations.push({
                fileName: result.file_name,
                pageNumber: result.page_number as number,
              })
            }
          }

          return JSON.stringify({
            query,
            document_id,
            searchType: "hybrid",
            count: results.length,
            results,
          })
        } catch (error: any) {
          console.error("PDF search failed:", error)
          return JSON.stringify({
            query,
            error: "Failed to search indexed PDFs",
            details: error.message,
          })
        }
      },
    })

    const tools = [itemLookupTool, pdfSearchTool]
    // Initialize Gemini with ordered fallbacks. The wrapper remains a normal
    // LangChain chat model, so createReactAgent can bind both search tools once
    // and use the same fallback order at every reasoning step.
    const [primaryModel, ...fallbackModels] = getGeminiChatModelNames()
    const model = new FallbackChatGoogleGenerativeAI({
      model: primaryModel,
      fallbackModels,
      temperature: 0,                    // Deterministic responses (no randomness)
      maxRetries: 0,                     // Disable built-in retries (we handle our own)
      apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment
    })
    console.log(`Gemini chat model order: ${model.modelNames.join(" -> ")}`)

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

IMPORTANT: You have access to two search tools:
- item_lookup searches the furniture inventory. Use it for prices, availability, reviews, comparisons, recommendations, and general product discovery.
- pdf_search searches indexed PDF documents and provides filename/page citations. Use it for product features, specifications, included parts or hardware, assembly, safety, care, warranty, manuals, policies, catalogs, uploaded documents, or when the customer asks what a document says.
- When a question needs both inventory data and documented product details, use both tools.

When using the item_lookup tool:
- If it returns results, provide helpful details about the furniture items
- If it returns an error or no results, acknowledge this and offer to help in other ways
- If the database appears to be empty, let the customer know that inventory might be being updated

When using pdf_search:
- Base document answers only on retrieved PDF text
- Cite the filename and page number for each factual claim
- Put each citation on its own line and bold the entire source and page reference using exactly this format: **Source: <filename> - Page: <page number>**
- If no relevant text is returned, say the indexed documents do not contain enough information

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
        signal: AbortSignal.timeout(60_000),    // Allow per-model fallback within an overall bound
      }
    ))

    // Extract the final response from the conversation
    const responseContent = finalState.messages[finalState.messages.length - 1].content
    const response = typeof responseContent === "string"
      ? appendPdfCitations(responseContent, pdfCitations)
      : responseContent
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
