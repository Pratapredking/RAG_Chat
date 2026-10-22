import { Embeddings, EmbeddingsParams } from "@langchain/core/embeddings"

type GeminiEmbeddingTask = "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY"

interface GeminiEmbeddingsParams extends EmbeddingsParams {
  apiKey?: string
  model?: string
  outputDimensionality?: number
}

/**
 * Minimal Gemini embeddings client for the current Generative Language API.
 *
 * The version of @langchain/google-genai used by this project predates
 * outputDimensionality. Atlas requires queries and indexed documents to use
 * the same vector size, so this adapter explicitly requests 768 dimensions.
 */
export class GeminiEmbeddings extends Embeddings {
  private readonly apiKey: string
  private readonly model: string
  private readonly outputDimensionality: number

  constructor({
    apiKey = process.env.GOOGLE_API_KEY,
    model = "gemini-embedding-001",
    outputDimensionality = 768,
    ...params
  }: GeminiEmbeddingsParams = {}) {
    super(params)

    if (!apiKey) {
      throw new Error("GOOGLE_API_KEY is required for Gemini embeddings")
    }

    this.apiKey = apiKey
    this.model = model
    this.outputDimensionality = outputDimensionality
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    return Promise.all(
      documents.map((document) => this.embed(document, "RETRIEVAL_DOCUMENT"))
    )
  }

  async embedQuery(document: string): Promise<number[]> {
    return this.embed(document, "RETRIEVAL_QUERY")
  }

  private async embed(text: string, taskType: GeminiEmbeddingTask): Promise<number[]> {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:embedContent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": this.apiKey,
        },
        body: JSON.stringify({
          content: { parts: [{ text }] },
          taskType,
          outputDimensionality: this.outputDimensionality,
        }),
      }
    )

    if (!response.ok) {
      const details = await response.text()
      const error = new Error(
        `Gemini embedding request failed (${response.status}): ${details}`
      ) as Error & { status?: number }
      error.status = response.status
      throw error
    }

    const payload = await response.json() as {
      embedding?: { values?: number[] }
    }
    const values = payload.embedding?.values

    if (!values || values.length !== this.outputDimensionality) {
      throw new Error(
        `Gemini returned ${values?.length ?? 0} embedding dimensions; expected ${this.outputDimensionality}`
      )
    }

    return values
  }
}
