export type RankedSearchResult<T> = {
  id: string
  document: T
  score?: number
}

export type RankedSearchSource<T> = {
  name: string
  weight?: number
  results: RankedSearchResult<T>[]
}

export type FusedSearchResult<T> = {
  document: T
  score: number
  matched_by: string[]
  source_scores: Record<string, number>
}

/**
 * Combines independently ranked result lists using weighted Reciprocal Rank
 * Fusion. RRF uses ranks instead of raw scores because vector similarity and
 * BM25 scores are not directly comparable.
 */
export function reciprocalRankFusion<T>(
  sources: RankedSearchSource<T>[],
  limit: number,
  rankConstant = 60
): FusedSearchResult<T>[] {
  const fused = new Map<string, FusedSearchResult<T>>()

  for (const source of sources) {
    const weight = source.weight ?? 1

    source.results.forEach((result, index) => {
      const existing = fused.get(result.id) ?? {
        document: result.document,
        score: 0,
        matched_by: [],
        source_scores: {},
      }

      existing.score += weight / (rankConstant + index + 1)
      if (!existing.matched_by.includes(source.name)) {
        existing.matched_by.push(source.name)
      }
      if (typeof result.score === "number") {
        existing.source_scores[source.name] = result.score
      }
      fused.set(result.id, existing)
    })
  }

  return Array.from(fused.values())
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
}
