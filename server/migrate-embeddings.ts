import "dotenv/config"
import { MongoClient } from "mongodb"
import { GeminiEmbeddings } from "./gemini-embeddings"

const mongoUri = process.env.MONGODB_ATLAS_URI

if (!mongoUri) {
  throw new Error("MONGODB_ATLAS_URI is required")
}

const client = new MongoClient(mongoUri)

async function withRetry<T>(operation: () => Promise<T>, maxAttempts = 3): Promise<T> {
  let lastError: unknown

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await operation()
    } catch (error) {
      lastError = error
      if (attempt < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt))
      }
    }
  }

  throw lastError
}

async function migrateEmbeddings(): Promise<void> {
  try {
    await client.connect()
    const collection = client.db("inventory_database").collection("items")
    const items = await collection.find({
      embedding_text: { $type: "string" },
      $or: [
        { embedding: { $exists: false } },
        { embedding: { $size: 0 } },
      ],
    }).project({ embedding_text: 1 }).toArray()

    if (items.length === 0) {
      console.log("No missing embeddings found")
      return
    }

    const embeddings = new GeminiEmbeddings()
    let migrated = 0

    for (const item of items) {
      const embedding = await withRetry(
        () => embeddings.embedQuery(item.embedding_text as string)
      )
      const result = await collection.updateOne(
        {
          _id: item._id,
          $or: [
            { embedding: { $exists: false } },
            { embedding: { $size: 0 } },
          ],
        },
        { $set: { embedding } }
      )
      migrated += result.modifiedCount
    }

    console.log(`Migrated ${migrated} of ${items.length} missing embeddings`)
  } finally {
    await client.close()
  }
}

migrateEmbeddings().catch((error) => {
  console.error("Embedding migration failed:", error)
  process.exitCode = 1
})
