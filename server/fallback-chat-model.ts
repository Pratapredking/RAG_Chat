import {
  ChatGoogleGenerativeAI,
  type GoogleGenerativeAIChatCallOptions,
  type GoogleGenerativeAIChatInput,
} from "@langchain/google-genai"
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager"
import type { BaseMessage } from "@langchain/core/messages"
import type { ChatResult } from "@langchain/core/outputs"

const DEFAULT_PRIMARY_MODEL = "gemini-2.5-flash"
const DEFAULT_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

type GeminiFallbackModelInput = GoogleGenerativeAIChatInput & {
  fallbackModels?: string[]
  requestTimeoutMs?: number
}

type ProviderError = Error & {
  status?: number
  response?: { status?: number }
  code?: string
  cause?: { code?: string }
}

function positiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? "", 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

function errorSummary(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

async function withRequestTimeout<T>(
  operation: () => Promise<T>,
  timeoutMs: number,
  externalSignal?: AbortSignal
): Promise<T> {
  let timeoutId: NodeJS.Timeout | undefined
  let abortHandler: (() => void) | undefined

  const interruption = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      const timeoutError = new Error(`Gemini request timed out after ${timeoutMs}ms`)
      timeoutError.name = "TimeoutError"
      reject(timeoutError)
    }, timeoutMs)

    if (externalSignal) {
      abortHandler = () => {
        const abortError = new Error("Gemini request was aborted")
        abortError.name = "AbortError"
        reject(abortError)
      }

      if (externalSignal.aborted) {
        abortHandler()
      } else {
        externalSignal.addEventListener("abort", abortHandler, { once: true })
      }
    }
  })

  try {
    return await Promise.race([operation(), interruption])
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    if (externalSignal && abortHandler) {
      externalSignal.removeEventListener("abort", abortHandler)
    }
  }
}

export function isGeminiFallbackError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false
  }

  const providerError = error as ProviderError
  const status = providerError.status ?? providerError.response?.status
  const networkCode = providerError.code ?? providerError.cause?.code
  const message = error.message.toLowerCase()

  return status === 404
    || status === 408
    || status === 409
    || status === 429
    || (typeof status === "number" && status >= 500)
    || error.name === "TypeError"
    || error.name === "AbortError"
    || error.name === "TimeoutError"
    || message.includes("fetch failed")
    || message.includes("network")
    || ["ABORT_ERR", "ECONNRESET", "ENOTFOUND", "EAI_AGAIN", "ETIMEDOUT"].includes(networkCode ?? "")
}

export function getGeminiChatModelNames(
  environment: NodeJS.ProcessEnv = process.env
): string[] {
  const primary = environment.GEMINI_CHAT_MODEL?.trim() || DEFAULT_PRIMARY_MODEL
  const configuredFallbacks = environment.GEMINI_FALLBACK_MODELS
    ?.split(",")
    .map((name) => name.trim())
    .filter(Boolean)
  const fallbacks = configuredFallbacks?.length
    ? configuredFallbacks
    : DEFAULT_FALLBACK_MODELS

  return [...new Set([primary, ...fallbacks])]
}

/**
 * A Gemini chat model that keeps LangChain's normal tool-binding behavior but
 * transparently tries the next configured model for transient/provider errors.
 */
export class FallbackChatGoogleGenerativeAI extends ChatGoogleGenerativeAI {
  private readonly fallbackModels: Array<{
    name: string
    model: ChatGoogleGenerativeAI
  }>

  private readonly requestTimeoutMs: number

  readonly modelNames: string[]

  constructor(fields: GeminiFallbackModelInput) {
    const {
      fallbackModels = [],
      requestTimeoutMs = positiveInteger(process.env.GEMINI_MODEL_TIMEOUT_MS, 15_000),
      ...modelFields
    } = fields

    super(modelFields)

    const uniqueFallbacks = [...new Set(fallbackModels)]
      .filter((name) => name && name !== modelFields.model)

    this.requestTimeoutMs = requestTimeoutMs
    this.modelNames = [modelFields.model, ...uniqueFallbacks]
    this.fallbackModels = uniqueFallbacks.map((name) => ({
      name,
      model: new ChatGoogleGenerativeAI({ ...modelFields, model: name }),
    }))
  }

  async _generate(
    messages: BaseMessage[],
    options: GoogleGenerativeAIChatCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const candidates = [
      {
        name: this.modelNames[0],
        generate: (callOptions: GoogleGenerativeAIChatCallOptions) =>
          super._generate(messages, callOptions, runManager),
      },
      ...this.fallbackModels.map(({ name, model }) => ({
        name,
        generate: (callOptions: GoogleGenerativeAIChatCallOptions) =>
          model._generate(messages, callOptions, runManager),
      })),
    ]
    const failures: Array<{ model: string; error: unknown }> = []

    for (let index = 0; index < candidates.length; index++) {
      const candidate = candidates[index]
      const timeoutSignal = AbortSignal.timeout(this.requestTimeoutMs)
      const signal = options.signal
        ? AbortSignal.any([options.signal, timeoutSignal])
        : timeoutSignal

      try {
        // @langchain/google-genai 0.2.x does not forward the signal from its
        // non-streaming _generate path, so enforce the deadline independently.
        return await withRequestTimeout(
          () => candidate.generate({ ...options, signal }),
          this.requestTimeoutMs,
          options.signal
        )
      } catch (error) {
        failures.push({ model: candidate.name, error })
        const nextModel = candidates[index + 1]?.name

        if (!nextModel || !isGeminiFallbackError(error)) {
          if (failures.length === 1) {
            throw error
          }

          const failureSummary = failures
            .map(({ model, error: modelError }) => `${model}: ${errorSummary(modelError)}`)
            .join("; ")
          const allModelsError = new Error(
            `All configured Gemini chat models failed. ${failureSummary}`
          ) as ProviderError
          const finalProviderError = error as ProviderError
          allModelsError.cause = finalProviderError
          allModelsError.status = finalProviderError.status
            ?? finalProviderError.response?.status
          allModelsError.code = finalProviderError.code
            ?? finalProviderError.cause?.code
          throw allModelsError
        }

        console.warn(
          `Gemini model "${candidate.name}" failed: ${errorSummary(error)}. `
          + `Trying fallback model "${nextModel}".`
        )
      }
    }

    throw new Error("No Gemini chat models were configured")
  }
}
