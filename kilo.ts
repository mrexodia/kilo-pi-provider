/**
 * Kilo Provider Extension
 *
 * Provides access to 300+ AI models via the Kilo Gateway (OpenRouter-compatible).
 * Uses device code flow for browser-based authentication.
 *
 * Usage:
 *   pi install git:github.com/mrexodia/kilo-pi-provider
 *   # Then /login kilo, or set KILO_API_KEY=...
 */

import {
  createProvider,
  type Credential,
  type Model,
  openAICompletionsApi,
  type OAuthCredential,
  type ProviderAuthInteraction,
} from "@earendil-works/pi-ai/compat";
import type {
  ExtensionAPI,
  ExtensionContext,
} from "@earendil-works/pi-coding-agent";

// =============================================================================
// Constants
// =============================================================================

const KILO_API_BASE = process.env.KILO_API_URL || "https://api.kilo.ai";
const KILO_GATEWAY_BASE = `${KILO_API_BASE}/api/gateway`;
const KILO_DEVICE_AUTH_ENDPOINT = `${KILO_API_BASE}/api/device-auth/codes`;
const POLL_INTERVAL_MS = 3000;
const MODELS_FETCH_TIMEOUT_MS = 10_000;
const TOKEN_EXPIRATION_MS = 365 * 24 * 60 * 60 * 1000; // 1 year
const KILO_FREE_API_KEY = "kilo-free";
const KILO_TOS_URL = "https://kilo.ai/terms";
const KILO_PROFILE_ENDPOINT = `${KILO_API_BASE}/api/profile`;

// =============================================================================
// Balance Fetching
// =============================================================================

interface KiloBalance {
  balance?: number;
}

async function fetchKiloBalance(token: string): Promise<number | null> {
  try {
    const response = await fetch(`${KILO_PROFILE_ENDPOINT}/balance`, {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as KiloBalance;
    return data.balance ?? null;
  } catch {
    return null;
  }
}

function formatCredits(balance: number): string {
  if (balance >= 1000) {
    return `$${(balance / 1000).toFixed(1)}k`;
  } else {
    return `$${balance.toFixed(2)}`;
  }
}

// =============================================================================
// Device Authorization Flow
// =============================================================================

interface DeviceAuthResponse {
  code: string;
  verificationUrl: string;
  expiresIn: number;
}

interface DeviceAuthPollResponse {
  status: "pending" | "approved" | "denied" | "expired";
  token?: string;
  userEmail?: string;
}

function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Login cancelled"));
      return;
    }
    const timeout = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timeout);
        reject(new Error("Login cancelled"));
      },
      { once: true },
    );
  });
}

async function initiateDeviceAuth(signal: AbortSignal): Promise<DeviceAuthResponse> {
  const response = await fetch(KILO_DEVICE_AUTH_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
  });

  if (!response.ok) {
    if (response.status === 429) {
      throw new Error(
        "Too many pending authorization requests. Please try again later.",
      );
    }
    throw new Error(
      `Failed to initiate device authorization: ${response.status}`,
    );
  }

  return (await response.json()) as DeviceAuthResponse;
}

async function pollDeviceAuth(
  code: string,
  signal: AbortSignal,
): Promise<DeviceAuthPollResponse> {
  const response = await fetch(`${KILO_DEVICE_AUTH_ENDPOINT}/${code}`, {
    signal,
  });

  if (response.status === 202) return { status: "pending" };
  if (response.status === 403) return { status: "denied" };
  if (response.status === 410) return { status: "expired" };

  if (!response.ok) {
    throw new Error(`Failed to poll device authorization: ${response.status}`);
  }

  return (await response.json()) as DeviceAuthPollResponse;
}

async function loginKilo(
  interaction: ProviderAuthInteraction,
): Promise<OAuthCredential> {
  interaction.notify({
    type: "progress",
    message: "Initiating device authorization...",
  });
  const authData = await initiateDeviceAuth(interaction.signal);
  const { code, verificationUrl, expiresIn } = authData;

  interaction.notify({
    type: "device_code",
    userCode: code,
    verificationUri: verificationUrl,
    intervalSeconds: POLL_INTERVAL_MS / 1000,
    expiresInSeconds: expiresIn,
  });
  interaction.notify({
    type: "progress",
    message: "Waiting for browser authorization...",
  });

  const deadline = Date.now() + expiresIn * 1000;
  while (Date.now() < deadline) {
    interaction.signal.throwIfAborted();
    await abortableSleep(POLL_INTERVAL_MS, interaction.signal);

    const result = await pollDeviceAuth(code, interaction.signal);

    if (result.status === "approved") {
      if (!result.token) {
        throw new Error("Authorization approved but no token received");
      }
      interaction.notify({ type: "progress", message: "Login successful!" });
      return {
        type: "oauth",
        refresh: result.token,
        access: result.token,
        expires: Date.now() + TOKEN_EXPIRATION_MS,
      };
    }

    if (result.status === "denied") {
      throw new Error("Authorization denied by user.");
    }

    if (result.status === "expired") {
      throw new Error("Authorization code expired. Please try again.");
    }

    const remaining = Math.ceil((deadline - Date.now()) / 1000);
    interaction.notify({
      type: "progress",
      message: `Waiting for browser authorization... (${remaining}s remaining)`,
    });
  }

  throw new Error("Authentication timed out. Please try again.");
}

async function refreshKiloToken(
  credentials: OAuthCredential,
  signal: AbortSignal,
): Promise<OAuthCredential> {
  signal.throwIfAborted();
  if (credentials.expires > Date.now()) {
    return credentials;
  }
  throw new Error(
    "Kilo token expired. Please run /login kilo to re-authenticate.",
  );
}

// =============================================================================
// Dynamic Model Loading
// =============================================================================

interface OpenRouterModel {
  id: string;
  name: string;
  context_length: number;
  max_completion_tokens?: number | null;
  pricing?: {
    prompt?: string | null;
    completion?: string | null;
    input_cache_write?: string | null;
    input_cache_read?: string | null;
  };
  architecture?: {
    input_modalities?: string[] | null;
    output_modalities?: string[] | null;
  };
  top_provider?: { max_completion_tokens?: number | null };
  supported_parameters?: string[];
}

function parsePrice(price: string | null | undefined): number {
  if (!price) return 0;
  const parsed = parseFloat(price);
  if (isNaN(parsed)) return 0;
  // OpenRouter prices are per-token; Pi expects per-million-token
  return parsed * 1_000_000;
}

function isKnownFreeModelId(id: string): boolean {
  // Zero pricing alone isn't reliable (some models report "0" but require auth).
  // Use the :free suffix (OpenRouter convention), Kilo-native models (no vendor
  // prefix), or known Kilo/OpenRouter free routers.
  return (
    id === "kilo-auto/free" ||
    id.includes(":free") ||
    !id.includes("/") ||
    id.startsWith("kilo/") ||
    id.startsWith("openrouter/")
  );
}

function isFreeModel(m: OpenRouterModel): boolean {
  const prompt = parseFloat(m.pricing?.prompt ?? "1");
  const completion = parseFloat(m.pricing?.completion ?? "1");
  return prompt === 0 && completion === 0 && isKnownFreeModelId(m.id);
}

type KiloModel = Model<"openai-completions">;
type KiloModelCompat = NonNullable<KiloModel["compat"]>;

function getKiloModelCompat(m: OpenRouterModel): KiloModel["compat"] {
  const compat: KiloModelCompat = {};

  // Kilo's gateway is OpenRouter-compatible, but it uses api.kilo.ai so
  // pi-ai's URL/provider auto-detection cannot infer OpenRouter model quirks.
  if (m.id.startsWith("anthropic/")) {
    compat.cacheControlFormat = "anthropic";
  }

  if (m.id === "deepseek/deepseek-v4-flash" || m.id === "deepseek/deepseek-v4-pro") {
    compat.requiresReasoningContentOnAssistantMessages = true;
  }

  return Object.keys(compat).length > 0 ? compat : undefined;
}

function mapOpenRouterModel(m: OpenRouterModel): KiloModel {
  const inputModalities = m.architecture?.input_modalities ?? ["text"];
  const supportsImages = inputModalities.includes("image");
  const supportsReasoning =
    m.supported_parameters?.includes("reasoning") ?? false;
  const maxTokens =
    m.top_provider?.max_completion_tokens ??
    m.max_completion_tokens ??
    Math.ceil(m.context_length * 0.2);

  return {
    id: m.id,
    name: m.name,
    api: "openai-completions",
    provider: "kilo",
    baseUrl: KILO_GATEWAY_BASE,
    reasoning: supportsReasoning,
    thinkingLevelMap:
      m.id === "deepseek/deepseek-v4-pro" ? { xhigh: "max" } : undefined,
    input: supportsImages ? ["text", "image"] : ["text"],
    cost: {
      input: parsePrice(m.pricing?.prompt),
      output: parsePrice(m.pricing?.completion),
      cacheRead: parsePrice(m.pricing?.input_cache_read),
      cacheWrite: parsePrice(m.pricing?.input_cache_write),
    },
    contextWindow: m.context_length,
    maxTokens: maxTokens,
    compat: getKiloModelCompat(m),
  };
}

async function fetchKiloModels(options?: {
  token?: string;
  freeOnly?: boolean;
  signal?: AbortSignal;
}): Promise<KiloModel[]> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": "pi-kilo-provider",
  };
  if (options?.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }

  const timeoutSignal = AbortSignal.timeout(MODELS_FETCH_TIMEOUT_MS);
  const signal = options?.signal
    ? AbortSignal.any([options.signal, timeoutSignal])
    : timeoutSignal;
  const response = await fetch(`${KILO_GATEWAY_BASE}/models`, {
    headers,
    signal,
  });

  if (!response.ok) {
    throw new Error(
      `Failed to fetch models: ${response.status} ${response.statusText}`,
    );
  }

  const json = (await response.json()) as { data?: OpenRouterModel[] };
  if (!json.data || !Array.isArray(json.data)) {
    throw new Error("Invalid models response: missing data array");
  }

  return json.data
    .filter((m) => {
      // Skip image generation models
      const outputMods = m.architecture?.output_modalities ?? [];
      if (outputMods.includes("image")) return false;
      // When unauthenticated, only show free models
      if (options?.freeOnly && !isFreeModel(m)) return false;
      return true;
    })
    .map(mapOpenRouterModel);
}

// =============================================================================
// Provider
// =============================================================================

function getCredentialToken(
  credential: Credential | undefined,
): string | undefined {
  if (credential?.type === "oauth") return credential.access;
  return credential?.key === KILO_FREE_API_KEY ? undefined : credential?.key;
}

function createKiloProvider(initialModels: KiloModel[]) {
  const openAI = openAICompletionsApi();
  const withoutFreePlaceholder = <
    T extends {
      apiKey?: string;
      headers?: Record<string, string | null>;
    },
  >(
    options: T | undefined,
  ): T | undefined => {
    if (options?.apiKey !== KILO_FREE_API_KEY) return options;
    return {
      ...options,
      apiKey: "unused",
      // A null request header suppresses the OpenAI SDK's generated bearer
      // header while retaining a non-empty internal key for client creation.
      headers: { ...options.headers, Authorization: null },
    };
  };

  return createProvider({
    id: "kilo",
    name: "Kilo",
    baseUrl: KILO_GATEWAY_BASE,
    headers: {
      "X-KILOCODE-EDITORNAME": "Pi",
      "User-Agent": "pi-kilo-provider",
    },
    auth: {
      // Kilo's free catalog is usable without credentials, so this auth method
      // deliberately resolves even when KILO_API_KEY is unset.
      apiKey: {
        name: "Kilo API key",
        async check({ ctx, credential }) {
          const key = credential?.key ?? (await ctx.env("KILO_API_KEY"));
          return {
            type: "api_key",
            source: key ? "KILO_API_KEY" : "Kilo free access",
          };
        },
        async resolve({ ctx, credential }) {
          const key = credential?.key ?? (await ctx.env("KILO_API_KEY"));
          return {
            // Pi requires a non-empty key to mark the provider available. The
            // stream wrapper strips this sentinel before free-model requests.
            auth: { apiKey: key ?? KILO_FREE_API_KEY },
            source: key ? "KILO_API_KEY" : "Kilo free access",
          };
        },
      },
      oauth: {
        name: "Kilo",
        login: loginKilo,
        refresh: refreshKiloToken,
        async toAuth(credential) {
          return { apiKey: credential.access };
        },
      },
    },
    models: initialModels,
    async fetchModels(context) {
      const token = getCredentialToken(context.credential);
      return fetchKiloModels({
        token,
        freeOnly: !token,
        signal: context.signal,
      });
    },
    // Never expose a persisted authenticated catalog after logout/offline
    // startup unless a real API key or OAuth token is currently configured.
    filterModels(models, credential) {
      if (getCredentialToken(credential)) return models;
      return models.filter(
        (model) =>
          model.cost.input === 0 &&
          model.cost.output === 0 &&
          isKnownFreeModelId(model.id),
      );
    },
    api: {
      stream(model, context, options) {
        return openAI.stream(model, context, withoutFreePlaceholder(options));
      },
      streamSimple(model, context, options) {
        return openAI.streamSimple(
          model,
          context,
          withoutFreePlaceholder(options),
        );
      },
    },
  });
}

// =============================================================================
// Extension Entry Point
// =============================================================================

async function updateKiloCredits(ctx: ExtensionContext): Promise<void> {
  if (ctx.model?.provider !== "kilo") {
    ctx.ui.setStatus("kilo-credits", undefined);
    return;
  }

  try {
    const auth = await ctx.modelRegistry.getProviderAuth("kilo");
    const token = auth?.auth.apiKey;
    if (!token || token === KILO_FREE_API_KEY) {
      ctx.ui.setStatus("kilo-credits", undefined);
      return;
    }

    const balance = await fetchKiloBalance(token);
    ctx.ui.setStatus(
      "kilo-credits",
      balance === null
        ? undefined
        : ctx.ui.theme.fg("accent", `💰 ${formatCredits(balance)}`),
    );
  } catch (error) {
    ctx.ui.setStatus("kilo-credits", undefined);
    console.warn(
      "[kilo] Failed to fetch balance:",
      error instanceof Error ? error.message : error,
    );
  }
}

export default async function (pi: ExtensionAPI) {
  let initialModels: KiloModel[] = [];
  if (process.env.PI_OFFLINE !== "1") {
    try {
      const token = process.env.KILO_API_KEY;
      initialModels = await fetchKiloModels({ token, freeOnly: !token });
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch models at startup:",
        error instanceof Error ? error.message : error,
      );
    }
  }

  pi.registerProvider(createKiloProvider(initialModels));

  pi.on("session_start", async (_event, ctx) => {
    const result = await ctx.modelRegistry.refresh({
      providers: ["kilo"],
      allowNetwork: process.env.PI_OFFLINE !== "1",
    });
    const refreshError = result.errors.get("kilo");
    if (refreshError) {
      console.warn("[kilo] Failed to refresh models:", refreshError.message);
    }
    await updateKiloCredits(ctx);
  });

  pi.on("model_select", async (_event, ctx) => {
    await updateKiloCredits(ctx);
  });

  pi.on("turn_end", async (_event, ctx) => {
    await updateKiloCredits(ctx);
  });

  // On first use of a Kilo model without login, print ToS notice.
  let tosShown = false;

  pi.on("before_agent_start", async (_event, ctx) => {
    if (tosShown) return;
    if (ctx.model?.provider !== "kilo") return;

    const auth = await ctx.modelRegistry.getProviderAuth("kilo");
    if (auth?.auth.apiKey && auth.auth.apiKey !== KILO_FREE_API_KEY) {
      tosShown = true;
      return;
    }

    tosShown = true;

    return {
      message: {
        customType: "kilo",
        content: `By using Kilo, you agree to the Terms of Service: ${KILO_TOS_URL}`,
        display: true,
      },
    };
  });
}
