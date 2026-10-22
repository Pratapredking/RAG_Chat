import "dotenv/config"
import { createHash } from "crypto"
import { readFile } from "fs/promises"
import path from "path"
import { MongoClient } from "mongodb"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { GeminiEmbeddings } from "./gemini-embeddings"

const pdfParse: (
  data: Buffer,
  options?: { pagerender?: (pageData: any) => Promise<string> }
) => Promise<{ numpages: number }> = require("pdf-parse")

export const PDF_COLLECTION_NAME = "pdf_chunks"
export const PDF_VECTOR_INDEX_NAME = "pdf_vector_index"

const DATABASE_NAME = "inventory_database"
const EMBEDDING_DIMENSIONS = 768
const DEFAULT_CHUNK_SIZE = 1000
const DEFAULT_CHUNK_OVERLAP = 150
const DEFAULT_MAX_FILE_SIZE_MB = 25
const MIN_EXTRACTED_PAGE_CHARACTERS = 10

type ExtractedPage = {
  pageNumber: number
  text: string
}

type PdfChunk = {
  documentId: string
  fileName: string
  pageNumber: number
  chunkIndex: number
  text: string
}

export type PdfSearchResult = {
  document_id: string
  file_name: string
  page_number: number | null
  chunk_index: number
  text: string
  score: number
}

export type PdfIngestionResult = {
  documentId: string
  fileName: string
  totalPages: number
  searchablePages: number
  ocrRequiredPages: number[]
  chunkCount: number
  indexCreated: boolean
  indexReady: boolean
  verificationQuery: string
  verificationResults: Array<{
    fileName: string
    pageNumber: number | null
    score: number
    preview: string
  }>
}

function normalizePageText(text: string): string {
  return text
    .replace(/\u0000/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim()
}

async function renderPdfPage(pageData: any): Promise<string> {
  const textContent = await pageData.getTextContent()
  let text = ""
  let previousY: number | undefined

  for (const item of textContent.items ?? []) {
    const value = typeof item.str === "string" ? item.str.trim() : ""
    if (!value) continue

    const currentY = Array.isArray(item.transform) ? item.transform[5] : undefined
    const startsNewLine = previousY !== undefined
      && currentY !== undefined
      && Math.abs(currentY - previousY) > 1

    if (text) text += startsNewLine ? "\n" : " "
    text += value
    previousY = currentY
  }

  return normalizePageText(text)
}

export async function extractPdfPages(pdfPath: string): Promise<{
  documentId: string
  fileName: string
  pages: ExtractedPage[]
  totalPages: number
  ocrRequiredPages: number[]
}> {
  const resolvedPath = path.resolve(pdfPath)

  if (path.extname(resolvedPath).toLowerCase() !== ".pdf") {
    throw new Error("The input file must have a .pdf extension")
  }

  const source = await readFile(resolvedPath)
  const maxSizeMb = Number(process.env.PDF_MAX_FILE_SIZE_MB || DEFAULT_MAX_FILE_SIZE_MB)
  const maxSizeBytes = maxSizeMb * 1024 * 1024

  if (source.length > maxSizeBytes) {
    throw new Error(`PDF exceeds the ${maxSizeMb} MB size limit`)
  }

  if (source.subarray(0, 5).toString("ascii") !== "%PDF-") {
    throw new Error("The input does not have a valid PDF file signature")
  }

  const extractedTexts: string[] = []
  let parsed: { numpages: number }

  try {
    parsed = await pdfParse(source, {
      pagerender: async (pageData) => {
        const pageText = await renderPdfPage(pageData)
        extractedTexts.push(pageText)
        return pageText
      },
    })
  } catch (error: any) {
    if (/password|encrypted/i.test(error?.message ?? "")) {
      throw new Error("The PDF is encrypted or password-protected")
    }
    throw new Error(`Unable to extract PDF text: ${error?.message ?? error}`)
  }

  const totalPages = parsed.numpages || extractedTexts.length
  const pages: ExtractedPage[] = []
  const ocrRequiredPages: number[] = []

  for (let index = 0; index < totalPages; index++) {
    const pageNumber = index + 1
    const text = normalizePageText(extractedTexts[index] ?? "")

    if (text.length < MIN_EXTRACTED_PAGE_CHARACTERS) {
      ocrRequiredPages.push(pageNumber)
    } else {
      pages.push({ pageNumber, text })
    }
  }

  if (pages.length === 0) {
    throw new Error(
      `No searchable text was extracted. OCR is required for pages: ${ocrRequiredPages.join(", ")}`
    )
  }

  return {
    documentId: createHash("sha256").update(source).digest("hex"),
    fileName: path.basename(resolvedPath),
    pages,
    totalPages,
    ocrRequiredPages,
  }
}

async function createPdfChunks(
  documentId: string,
  fileName: string,
  pages: ExtractedPage[]
): Promise<PdfChunk[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: DEFAULT_CHUNK_SIZE,
    chunkOverlap: DEFAULT_CHUNK_OVERLAP,
  })
  const chunks: PdfChunk[] = []

  for (const page of pages) {
    const pageChunks = await splitter.splitText(page.text)
    for (const text of pageChunks) {
      chunks.push({
        documentId,
        fileName,
        pageNumber: page.pageNumber,
        chunkIndex: chunks.length,
        text,
      })
    }
  }

  return chunks
}

async function retry<T>(operation: () => Promise<T>, maxAttempts = 3): Promise<T> {
  let lastError: unknown

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await operation()
    } catch (error: any) {
      lastError = error
      const retryable = error?.status === 429
        || error?.status >= 500
        || error?.name === "TypeError"
        || error?.message?.includes("fetch failed")

      if (!retryable || attempt === maxAttempts) throw error
      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt))
    }
  }

  throw lastError
}

export async function ensurePdfIndexes(client: MongoClient): Promise<boolean> {
  const collection = client.db(DATABASE_NAME).collection(PDF_COLLECTION_NAME)

  await collection.createIndex(
    { document_id: 1, chunk_index: 1 },
    { unique: true, name: "pdf_document_chunk_unique" }
  )

  const indexes = await collection
    .listSearchIndexes(PDF_VECTOR_INDEX_NAME)
    .toArray()

  if (indexes.length > 0) return false

  await collection.createSearchIndex({
    name: PDF_VECTOR_INDEX_NAME,
    type: "vectorSearch",
    definition: {
      fields: [
        {
          type: "vector",
          path: "embedding",
          numDimensions: EMBEDDING_DIMENSIONS,
          similarity: "cosine",
        },
        {
          type: "filter",
          path: "document_id",
        },
      ],
    },
  })

  return true
}

export async function waitForPdfIndex(
  client: MongoClient,
  timeoutMs = 120_000
): Promise<boolean> {
  const collection = client.db(DATABASE_NAME).collection(PDF_COLLECTION_NAME)
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    const indexes = await collection
      .listSearchIndexes(PDF_VECTOR_INDEX_NAME)
      .toArray() as Array<{ queryable?: boolean; status?: string }>
    const index = indexes[0]

    if (index?.queryable === true || index?.status === "READY") return true
    await new Promise((resolve) => setTimeout(resolve, 5_000))
  }

  return false
}

export async function searchPdfChunks(
  client: MongoClient,
  query: string,
  options: { documentId?: string; limit?: number } = {}
): Promise<PdfSearchResult[]> {
  const trimmedQuery = query.trim()
  if (!trimmedQuery) throw new Error("PDF search query cannot be empty")

  const limit = Math.min(Math.max(options.limit ?? 5, 1), 20)
  const embeddings = new GeminiEmbeddings()
  const queryVector = await retry(() => embeddings.embedQuery(trimmedQuery))
  const vectorSearch: Record<string, unknown> = {
    index: PDF_VECTOR_INDEX_NAME,
    path: "embedding",
    queryVector,
    numCandidates: Math.max(100, limit * 20),
    limit,
  }

  if (options.documentId) {
    vectorSearch.filter = { document_id: options.documentId }
  }

  return client.db(DATABASE_NAME).collection(PDF_COLLECTION_NAME).aggregate<PdfSearchResult>([
    { $vectorSearch: vectorSearch },
    {
      $project: {
        _id: 0,
        document_id: 1,
        file_name: 1,
        page_number: 1,
        chunk_index: 1,
        text: 1,
        score: { $meta: "vectorSearchScore" },
      },
    },
  ]).toArray()
}

export async function ingestPdf(
  client: MongoClient,
  pdfPath: string,
  verificationQuery?: string
): Promise<PdfIngestionResult> {
  const extracted = await extractPdfPages(pdfPath)
  const chunks = await createPdfChunks(
    extracted.documentId,
    extracted.fileName,
    extracted.pages
  )

  if (chunks.length === 0) {
    throw new Error("PDF extraction produced no chunks")
  }

  const indexCreated = await ensurePdfIndexes(client)
  const collection = client.db(DATABASE_NAME).collection(PDF_COLLECTION_NAME)
  const embeddings = new GeminiEmbeddings()
  const createdAt = new Date()
  const batchSize = 10

  for (let offset = 0; offset < chunks.length; offset += batchSize) {
    const batch = chunks.slice(offset, offset + batchSize)
    const vectors = await retry(() => embeddings.embedDocuments(
      batch.map((chunk) => chunk.text)
    ))

    if (vectors.some((vector) => vector.length !== EMBEDDING_DIMENSIONS)) {
      throw new Error("Gemini returned an unexpected PDF embedding size")
    }

    await collection.bulkWrite(batch.map((chunk, index) => ({
      updateOne: {
        filter: {
          document_id: chunk.documentId,
          chunk_index: chunk.chunkIndex,
        },
        update: {
          $set: {
            document_id: chunk.documentId,
            file_name: chunk.fileName,
            page_number: chunk.pageNumber,
            chunk_index: chunk.chunkIndex,
            text: chunk.text,
            embedding: vectors[index],
            created_at: createdAt,
          },
        },
        upsert: true,
      },
    })), { ordered: true })
  }

  await collection.deleteMany({
    document_id: extracted.documentId,
    chunk_index: { $gte: chunks.length },
  })

  const invalidEmbeddings = await collection.countDocuments({
    document_id: extracted.documentId,
    $expr: { $ne: [{ $size: "$embedding" }, EMBEDDING_DIMENSIONS] },
  })

  if (invalidEmbeddings > 0) {
    throw new Error(`${invalidEmbeddings} stored PDF chunks have invalid embeddings`)
  }

  const indexReady = await waitForPdfIndex(client)
  const query = verificationQuery?.trim()
    || chunks[0].text.slice(0, 180)
  const searchResults = indexReady
    ? await searchPdfChunks(client, query, {
        documentId: extracted.documentId,
        limit: 3,
      })
    : []

  if (indexReady && searchResults.length === 0) {
    throw new Error("PDF chunks were stored, but semantic retrieval returned no results")
  }

  return {
    documentId: extracted.documentId,
    fileName: extracted.fileName,
    totalPages: extracted.totalPages,
    searchablePages: extracted.pages.length,
    ocrRequiredPages: extracted.ocrRequiredPages,
    chunkCount: chunks.length,
    indexCreated,
    indexReady,
    verificationQuery: query,
    verificationResults: searchResults.map((result) => ({
      fileName: result.file_name,
      pageNumber: result.page_number,
      score: result.score,
      preview: result.text.slice(0, 180),
    })),
  }
}
