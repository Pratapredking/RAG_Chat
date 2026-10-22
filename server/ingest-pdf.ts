import "dotenv/config"
import { MongoClient } from "mongodb"
import { ingestPdf } from "./pdf-rag"

function getOption(args: string[], name: string): string | undefined {
  const index = args.indexOf(name)
  return index >= 0 ? args[index + 1] : undefined
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const pdfPath = args.find((argument, index) =>
    !argument.startsWith("--") && args[index - 1] !== "--query"
  )
  const verificationQuery = getOption(args, "--query")

  if (!pdfPath) {
    console.error('Usage: npm run ingest:pdf -- "path/to/file.pdf" [--query "verification question"]')
    process.exitCode = 1
    return
  }

  const mongoUri = process.env.MONGODB_ATLAS_URI
  if (!mongoUri) {
    throw new Error("MONGODB_ATLAS_URI is required")
  }
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY is required")
  }

  const client = new MongoClient(mongoUri)

  try {
    await client.connect()
    const result = await ingestPdf(client, pdfPath, verificationQuery)
    console.log(JSON.stringify(result, null, 2))

    if (!result.indexReady) {
      console.error(
        "PDF chunks were stored, but the Atlas vector index was not ready within two minutes. Run the command again to verify retrieval."
      )
      process.exitCode = 2
    }

    if (result.ocrRequiredPages.length > 0) {
      console.warn(
        `OCR is required for skipped pages: ${result.ocrRequiredPages.join(", ")}`
      )
    }
  } finally {
    await client.close()
  }
}

main().catch((error) => {
  console.error("PDF ingestion failed:", error instanceof Error ? error.message : error)
  process.exitCode = 1
})
