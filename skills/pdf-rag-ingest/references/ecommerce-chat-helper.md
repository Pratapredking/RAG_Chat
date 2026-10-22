# ecommerce-chat-helper conventions

Use these conventions only when operating on this repository.

## Existing components

- Backend directory: `server/`
- Environment variables: `GOOGLE_API_KEY`, `MONGODB_ATLAS_URI`
- Database: `inventory_database`
- Embedding adapter: `server/gemini-embeddings.ts`
- Embedding model: `gemini-embedding-001`
- Vector size: `768`
- Similarity: `cosine`

Reuse `GeminiEmbeddings`; do not create a second embedding implementation.

## PDF storage

Use a separate `pdf_chunks` collection and a `pdf_vector_index` vector-search index.

Each chunk should have this shape:

```ts
{
  document_id: string,       // SHA-256 of source PDF bytes
  file_name: string,
  page_number: number | null,
  chunk_index: number,
  text: string,
  embedding: number[],       // exactly 768 values
  created_at: Date
}
```

Create the index with:

```ts
{
  name: "pdf_vector_index",
  type: "vectorSearch",
  definition: {
    fields: [
      {
        type: "vector",
        path: "embedding",
        numDimensions: 768,
        similarity: "cosine"
      },
      {
        type: "filter",
        path: "document_id"
      }
    ]
  }
}
```

## Chunking

Start with 1,000 characters and 150 characters of overlap. Preserve page boundaries where practical. Adjust only when the document structure or retrieval evaluation gives a concrete reason.

## Agent integration

Expose retrieval as a `pdf_search` tool alongside `item_lookup`. Return a compact JSON payload containing text, filename, page number, document ID, and score. Instruct the chat model to cite filename and page and to say when retrieved context is insufficient.

## Verification

After ingestion:

1. Confirm no stored embedding is empty and every embedding has 768 values.
2. Wait for the Atlas search index to become queryable if it was just created.
3. Search for a distinctive phrase or concept from the PDF.
4. Confirm the expected page appears among the top results.
5. Do not claim completion if only text extraction or MongoDB insertion succeeded; semantic retrieval must also work.

Add `server/uploads/` to `.gitignore` if uploaded PDFs will be stored there temporarily.
