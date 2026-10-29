import type { Collection, Document, MongoClient } from "mongodb"
import { GeminiEmbeddings } from "./gemini-embeddings"
import { reciprocalRankFusion, RankedSearchResult } from "./hybrid-search"

export const INVENTORY_VECTOR_INDEX_NAME = "vector_index"
export const INVENTORY_TEXT_INDEX_NAME = "inventory_text_index"

const DATABASE_NAME = "inventory_database"
const COLLECTION_NAME = "items"
const EMBEDDING_DIMENSIONS = 768

type InventoryCandidate = Document & {
  vector_score?: number
  text_score?: number
}

export type InventorySearchResult = Document & {
  score: number
  matched_by: string[]
  source_scores: Record<string, number>
}

function candidateId(candidate: InventoryCandidate): string {
  return candidate._id?.toString() ?? String(candidate.item_id)
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

async function vectorSearch(
  collection: Collection,
  query: string,
  limit: number
): Promise<InventoryCandidate[]> {
  const queryVector = await new GeminiEmbeddings().embedQuery(query)

  return collection.aggregate<InventoryCandidate>([
    {
      $vectorSearch: {
        index: INVENTORY_VECTOR_INDEX_NAME,
        path: "embedding",
        queryVector,
        numCandidates: Math.max(100, limit * 10),
        limit,
      },
    },
    {
      $project: {
        embedding: 0,
        vector_score: { $meta: "vectorSearchScore" },
      },
    },
  ]).toArray()
}

async function regexTextSearch(
  collection: Collection,
  query: string,
  limit: number
): Promise<InventoryCandidate[]> {
  const fields = ["item_name", "item_description", "categories", "brand", "embedding_text"]
  const exactPattern = escapeRegex(query)
  let results = await collection.find({
    $or: fields.map((field) => ({ [field]: { $regex: exactPattern, $options: "i" } })),
  }, { projection: { embedding: 0 } }).limit(limit).toArray()

  if (results.length === 0) {
    const tokens = query.toLowerCase().match(/[a-z0-9]+/g)
      ?.filter((token) => token.length > 2)
      .map((token) => token.endsWith("s") ? token.slice(0, -1) : token)
      .filter((token, index, all) => all.indexOf(token) === index)
      ?? []

    if (tokens.length > 0) {
      results = await collection.find({
        $and: tokens.map((token) => ({
          $or: fields.map((field) => ({
            [field]: { $regex: `${escapeRegex(token)}s?`, $options: "i" },
          })),
        })),
      }, { projection: { embedding: 0 } }).limit(limit).toArray()
    }
  }

  return results
}

async function lexicalSearch(
  collection: Collection,
  query: string,
  limit: number
): Promise<InventoryCandidate[]> {
  try {
    return await collection.aggregate<InventoryCandidate>([
      {
        $search: {
          index: INVENTORY_TEXT_INDEX_NAME,
          compound: {
            should: [
              {
                text: {
                  query,
                  path: "item_name",
                  score: { boost: { value: 3 } },
                },
              },
              {
                text: {
                  query,
                  path: "categories",
                  score: { boost: { value: 2 } },
                },
              },
              {
                text: {
                  query,
                  path: ["item_description", "brand", "embedding_text"],
                },
              },
            ],
            minimumShouldMatch: 1,
          },
        },
      },
      { $limit: limit },
      {
        $project: {
          embedding: 0,
          text_score: { $meta: "searchScore" },
        },
      },
    ]).toArray()
  } catch (error: any) {
    console.warn(`Atlas inventory text search unavailable; using regex fallback: ${error.message}`)
    return regexTextSearch(collection, query, limit)
  }
}

export async function searchInventoryItems(
  collection: Collection,
  query: string,
  requestedLimit = 10
): Promise<InventorySearchResult[]> {
  const trimmedQuery = query.trim()
  if (!trimmedQuery) throw new Error("Inventory search query cannot be empty")

  const limit = Math.min(Math.max(requestedLimit, 1), 20)
  const candidateLimit = Math.min(Math.max(limit * 4, 20), 100)

  const [vectorResults, textResults] = await Promise.all([
    vectorSearch(collection, trimmedQuery, candidateLimit).catch((error: any) => {
      console.warn(`Inventory vector search unavailable: ${error.message}`)
      return [] as InventoryCandidate[]
    }),
    lexicalSearch(collection, trimmedQuery, candidateLimit),
  ])

  const toRanked = (
    results: InventoryCandidate[],
    scoreField: "vector_score" | "text_score"
  ): RankedSearchResult<InventoryCandidate>[] => results.map((document) => ({
    id: candidateId(document),
    document,
    score: document[scoreField],
  }))

  return reciprocalRankFusion([
    { name: "vector", results: toRanked(vectorResults, "vector_score") },
    { name: "text", results: toRanked(textResults, "text_score") },
  ], limit).map(({ document, ...ranking }) => ({
    ...document,
    ...ranking,
  }))
}

export async function ensureInventorySearchIndexes(client: MongoClient): Promise<boolean> {
  const db = client.db(DATABASE_NAME)
  const collections = await db.listCollections({ name: COLLECTION_NAME }).toArray()
  if (collections.length === 0) {
    await db.createCollection(COLLECTION_NAME)
  }
  const collection = db.collection(COLLECTION_NAME)
  const indexes = await collection.listSearchIndexes().toArray() as Array<{ name?: string }>
  const existingNames = new Set(indexes.map((index) => index.name))
  let created = false

  if (!existingNames.has(INVENTORY_VECTOR_INDEX_NAME)) {
    await collection.createSearchIndex({
      name: INVENTORY_VECTOR_INDEX_NAME,
      type: "vectorSearch",
      definition: {
        fields: [{
          type: "vector",
          path: "embedding",
          numDimensions: EMBEDDING_DIMENSIONS,
          similarity: "cosine",
        }],
      },
    })
    created = true
  }

  if (!existingNames.has(INVENTORY_TEXT_INDEX_NAME)) {
    await collection.createSearchIndex({
      name: INVENTORY_TEXT_INDEX_NAME,
      definition: {
        mappings: {
          dynamic: false,
          fields: {
            item_name: { type: "string" },
            item_description: { type: "string" },
            categories: { type: "string" },
            brand: { type: "string" },
            embedding_text: { type: "string" },
          },
        },
      },
    })
    created = true
  }

  return created
}
