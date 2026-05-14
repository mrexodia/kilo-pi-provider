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

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type {
  Api,
  Model,
  OAuthCredentials,
  OAuthLoginCallbacks,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI, ProviderModelConfig } from "@mariozechner/pi-coding-agent";
import { visibleWidth } from "@mariozechner/pi-tui";

// =============================================================================
// Constants
// =============================================================================

const KILO_API_BASE = process.env.KILO_API_URL || "https://api.kilo.ai";
const KILO_GATEWAY_BASE = `${KILO_API_BASE}/api/gateway`;
const KILO_DEVICE_AUTH_ENDPOINT = `${KILO_API_BASE}/api/device-auth/codes`;
const POLL_INTERVAL_MS = 3000;
const MODELS_FETCH_TIMEOUT_MS = 10_000;
const TOKEN_EXPIRATION_MS = 365 * 24 * 60 * 60 * 1000; // 1 year
const KILO_TOS_URL = "https://kilo.ai/terms";
const KILO_PROFILE_ENDPOINT = `${KILO_API_BASE}/api/profile`;
const KILO_ORG_HEADER = "X-KiloCode-OrganizationId";

function getEnvOrganizationId(): string | undefined {
  return process.env.KILO_ORG_ID || process.env.KILOCODE_ORGANIZATION_ID;
}

function getAgentDir(): string {
  return process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
}

function readStoredKiloCredentials(): OAuthCredentials | undefined {
  try {
    const authPath = join(getAgentDir(), "auth.json");
    if (!existsSync(authPath)) return undefined;
    const auth = JSON.parse(readFileSync(authPath, "utf8")) as {
      kilo?: { type?: string } & OAuthCredentials;
    };
    const cred = auth.kilo;
    if (cred?.type !== "oauth" || !cred.access) return undefined;
    return cred;
  } catch {
    return undefined;
  }
}

function getCredentialOrganizationId(credentials?: OAuthCredentials): string | undefined {
  const accountId = credentials?.accountId;
  return typeof accountId === "string" && accountId.trim() ? accountId : undefined;
}

function getEffectiveOrganizationId(credentials?: OAuthCredentials): string | undefined {
  return getCredentialOrganizationId(credentials) ?? getEnvOrganizationId();
}

function withOrganizationHeader(
  headers: Record<string, string>,
  organizationId?: string,
): Record<string, string> {
  if (!organizationId) return headers;
  return { ...headers, [KILO_ORG_HEADER]: organizationId };
}

// =============================================================================
// Profile and Balance Fetching
// =============================================================================

interface KiloOrganization {
  id: string;
  name: string;
  role?: string;
}

interface KiloProfile {
  user?: { email?: string; name?: string };
  email?: string;
  name?: string;
  organizations?: KiloOrganization[];
}

interface KiloBalance {
  balance?: number;
}

async function fetchKiloProfile(token: string): Promise<KiloProfile> {
  const response = await fetch(KILO_PROFILE_ENDPOINT, {
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch Kilo profile: ${response.status}`);
  }

  return (await response.json()) as KiloProfile;
}

async function selectKiloOrganization(
  token: string,
  callbacks: OAuthLoginCallbacks,
): Promise<string | undefined> {
  let profile: KiloProfile;
  try {
    callbacks.onProgress?.("Fetching Kilo profile...");
    profile = await fetchKiloProfile(token);
  } catch (error) {
    console.warn(
      "[kilo] Failed to fetch profile for organization selection:",
      error instanceof Error ? error.message : error,
    );
    return getEnvOrganizationId();
  }

  const organizations = profile.organizations ?? [];
  const envOrganizationId = getEnvOrganizationId();
  if (envOrganizationId && organizations.some((org) => org.id === envOrganizationId)) {
    return envOrganizationId;
  }
  if (!callbacks.onSelect || organizations.length === 0) {
    return envOrganizationId;
  }

  const selected = await callbacks.onSelect({
    message: "Select Kilo account",
    options: [
      { id: "personal", label: "Personal Account" },
      ...organizations.map((org) => ({
        id: org.id,
        label: `${org.name}${org.role ? ` (${org.role})` : ""}`,
      })),
    ],
  });

  if (!selected || selected === "personal") return undefined;
  return selected;
}

async function fetchKiloBalance(
  token: string,
  organizationId?: string,
): Promise<number | null> {
  try {
    const response = await fetch(`${KILO_PROFILE_ENDPOINT}/balance`, {
      headers: withOrganizationHeader(
        {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        organizationId,
      ),
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

async function initiateDeviceAuth(): Promise<DeviceAuthResponse> {
  const response = await fetch(KILO_DEVICE_AUTH_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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

async function pollDeviceAuth(code: string): Promise<DeviceAuthPollResponse> {
  const response = await fetch(`${KILO_DEVICE_AUTH_ENDPOINT}/${code}`);

  if (response.status === 202) return { status: "pending" };
  if (response.status === 403) return { status: "denied" };
  if (response.status === 410) return { status: "expired" };

  if (!response.ok) {
    throw new Error(`Failed to poll device authorization: ${response.status}`);
  }

  return (await response.json()) as DeviceAuthPollResponse;
}

async function loginKilo(
  callbacks: OAuthLoginCallbacks,
): Promise<OAuthCredentials> {
  callbacks.onProgress?.("Initiating device authorization...");
  const authData = await initiateDeviceAuth();
  const { code, verificationUrl, expiresIn } = authData;

  callbacks.onAuth({
    url: verificationUrl,
    instructions: `Enter code: ${code}`,
  });

  callbacks.onProgress?.("Waiting for browser authorization...");

  const deadline = Date.now() + expiresIn * 1000;
  while (Date.now() < deadline) {
    if (callbacks.signal?.aborted) {
      throw new Error("Login cancelled");
    }

    await abortableSleep(POLL_INTERVAL_MS, callbacks.signal);

    const result = await pollDeviceAuth(code);

    if (result.status === "approved") {
      if (!result.token) {
        throw new Error("Authorization approved but no token received");
      }
      callbacks.onProgress?.("Login successful!");
      const organizationId = await selectKiloOrganization(result.token, callbacks);
      return {
        refresh: result.token,
        access: result.token,
        expires: Date.now() + TOKEN_EXPIRATION_MS,
        ...(organizationId ? { accountId: organizationId } : {}),
      };
    }

    if (result.status === "denied") {
      throw new Error("Authorization denied by user.");
    }

    if (result.status === "expired") {
      throw new Error("Authorization code expired. Please try again.");
    }

    const remaining = Math.ceil((deadline - Date.now()) / 1000);
    callbacks.onProgress?.(
      `Waiting for browser authorization... (${remaining}s remaining)`,
    );
  }

  throw new Error("Authentication timed out. Please try again.");
}

async function refreshKiloToken(
  credentials: OAuthCredentials,
): Promise<OAuthCredentials> {
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

function isFreeModel(m: OpenRouterModel): boolean {
  const prompt = parseFloat(m.pricing?.prompt ?? "1");
  const completion = parseFloat(m.pricing?.completion ?? "1");
  if (prompt !== 0 || completion !== 0) return false;
  // Zero pricing alone isn't reliable (some models report "0" but require auth).
  // Use the :free suffix (OpenRouter convention), Kilo-native models (no vendor
  // prefix), or known Kilo/OpenRouter free routers.
  if (m.id === "kilo-auto/free") return true;
  if (m.id.includes(":free")) return true;
  if (!m.id.includes("/")) return true;
  if (m.id.startsWith("kilo/") || m.id.startsWith("openrouter/")) return true;
  return false;
}

type KiloReasoningLevel = "minimal" | "low" | "medium" | "high" | "xhigh";

type KiloModelCompat = {
  cacheControlFormat?: "anthropic";
  requiresReasoningContentOnAssistantMessages?: boolean;
  reasoningEffortMap?: Partial<Record<KiloReasoningLevel, string>>;
};

function getKiloModelCompat(
  m: OpenRouterModel,
): ProviderModelConfig["compat"] | undefined {
  const compat: KiloModelCompat = {};

  // Kilo's gateway is OpenRouter-compatible, but it uses api.kilo.ai so
  // pi-ai's URL/provider auto-detection cannot infer OpenRouter model quirks.
  if (m.id.startsWith("anthropic/")) {
    compat.cacheControlFormat = "anthropic";
  }

  if (m.id === "deepseek/deepseek-v4-flash" || m.id === "deepseek/deepseek-v4-pro") {
    compat.requiresReasoningContentOnAssistantMessages = true;
  }

  if (m.id === "deepseek/deepseek-v4-pro") {
    compat.reasoningEffortMap = { xhigh: "max" };
  }

  return Object.keys(compat).length > 0
    ? (compat as ProviderModelConfig["compat"])
    : undefined;
}

function mapOpenRouterModel(m: OpenRouterModel): ProviderModelConfig {
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
    reasoning: supportsReasoning,
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
  organizationId?: string;
  freeOnly?: boolean;
}): Promise<ProviderModelConfig[]> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": "pi-kilo-provider",
  };
  if (options?.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }
  const organizationId = options?.organizationId;
  const requestHeaders = withOrganizationHeader(headers, organizationId);
  const modelsUrl = organizationId
    ? `${KILO_API_BASE}/api/organizations/${encodeURIComponent(organizationId)}/models`
    : `${KILO_GATEWAY_BASE}/models`;

  const response = await fetch(modelsUrl, {
    headers: requestHeaders,
    signal: AbortSignal.timeout(MODELS_FETCH_TIMEOUT_MS),
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
// Provider Config
// =============================================================================

const KILO_PROVIDER_CONFIG = {
  baseUrl: KILO_GATEWAY_BASE,
  apiKey: "KILO_API_KEY",
  api: "openai-completions" as const,
  headers: {
    "X-KILOCODE-EDITORNAME": "Pi",
    "User-Agent": "pi-kilo-provider",
  },
};

function makeProviderConfig(organizationId?: string) {
  return {
    ...KILO_PROVIDER_CONFIG,
    headers: withOrganizationHeader(KILO_PROVIDER_CONFIG.headers, organizationId),
  };
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default async function (pi: ExtensionAPI) {
  const storedCredentials = readStoredKiloCredentials();
  const startupOrganizationId = getEffectiveOrganizationId(storedCredentials);

  // Fetch models at load time so the provider is immediately usable for
  // --list-models, --model selection, and print mode before session_start fires.
  let freeModels: ProviderModelConfig[] = [];
  let cachedAllModels: ProviderModelConfig[] = [];
  try {
    if (storedCredentials?.access) {
      cachedAllModels = await fetchKiloModels({
        token: storedCredentials.access,
        organizationId: startupOrganizationId,
      });
      freeModels = cachedAllModels.length > 0 ? cachedAllModels : [];
    } else {
      freeModels = await fetchKiloModels({ freeOnly: true });
    }
  } catch (error) {
    console.warn(
      "[kilo] Failed to fetch models at startup:",
      error instanceof Error ? error.message : error,
    );
    if (freeModels.length === 0) {
      try {
        freeModels = await fetchKiloModels({ freeOnly: true });
      } catch {}
    }
  }

  function makeOAuthConfig() {
    return {
      name: "Kilo",
      login: async (callbacks: OAuthLoginCallbacks) => {
        const cred = await loginKilo(callbacks);
        // Cache full models so modifyModels can use them during the
        // modelRegistry.refresh() that runs right after login returns.
        try {
          const organizationId = getEffectiveOrganizationId(cred);
          cachedAllModels = await fetchKiloModels({ token: cred.access, organizationId });
        } catch (error) {
          console.warn(
            "[kilo] Failed to fetch models after login:",
            error instanceof Error ? error.message : error,
          );
        }
        return cred;
      },
      refreshToken: refreshKiloToken,
      getApiKey: (cred: OAuthCredentials) => cred.access,
      // Called by modelRegistry.refresh() when credentials exist.
      // After logout, credentials are removed so this won't be called,
      // leaving only the free models from config.models.
      modifyModels: (models: Model<Api>[], cred: OAuthCredentials) => {
        if (cachedAllModels.length === 0) return models;
        const organizationId = getEffectiveOrganizationId(cred);
        const orgHeaders = organizationId ? { [KILO_ORG_HEADER]: organizationId } : undefined;
        // Use an existing kilo model as a template for provider metadata
        const template = models.find((m) => m.provider === "kilo");
        if (!template) return models;
        const nonKilo = models.filter((m) => m.provider !== "kilo");
        const fullModels = cachedAllModels.map((m) => ({
          ...template,
          id: m.id,
          name: m.name,
          reasoning: m.reasoning,
          input: m.input,
          cost: m.cost,
          contextWindow: m.contextWindow,
          maxTokens: m.maxTokens,
          headers: orgHeaders,
          compat: m.compat,
        }));
        return [...nonKilo, ...fullModels];
      },
    };
  }

  // Always register with free models. modifyModels upgrades to full list
  // when credentials exist, and naturally falls back after logout.
  pi.registerProvider("kilo", {
    ...makeProviderConfig(getEnvOrganizationId()),
    models: freeModels,
    oauth: makeOAuthConfig(),
  });

  // After session starts, pre-fetch all models if already logged in so
  // modifyModels has data to work with. Also fetch and display credits.
  pi.on("session_start", async (_event, ctx) => {
    const cred = ctx.modelRegistry.authStorage.get("kilo");

    // Clear credits if not logged in
    if (cred?.type !== "oauth") {
      ctx.ui.setStatus("kilo-credits", undefined);
      return;
    }

    try {
      cachedAllModels = await fetchKiloModels({
        token: cred.access,
        organizationId: getEffectiveOrganizationId(cred),
      });
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch models at session start:",
        error instanceof Error ? error.message : error,
      );
      return;
    }
    if (cachedAllModels.length > 0) {
      // Re-register to trigger modifyModels with the cached data.
      ctx.modelRegistry.registerProvider("kilo", {
        ...makeProviderConfig(getEffectiveOrganizationId(cred)),
        models: freeModels,
        oauth: makeOAuthConfig(),
      });
    }

    // Fetch and display credits balance
    try {
      const balance = await fetchKiloBalance(cred.access, getEffectiveOrganizationId(cred));
      if (balance !== null) {
        const theme = ctx.ui.theme;
        ctx.ui.setStatus(
          "kilo-credits",
          theme.fg("accent", `💰 ${formatCredits(balance)}`),
        );
      }
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch balance:",
        error instanceof Error ? error.message : error,
      );
    }
  });

  // Update credits display when model changes to a Kilo model
  pi.on("model_select", async (event, ctx) => {
    if (event.model?.provider !== "kilo") return;

    const cred = ctx.modelRegistry.authStorage.get("kilo");
    if (cred?.type !== "oauth") return;

    try {
      const balance = await fetchKiloBalance(cred.access, getEffectiveOrganizationId(cred));
      if (balance !== null) {
        const theme = ctx.ui.theme;
        ctx.ui.setStatus(
          "kilo-credits",
          theme.fg("accent", `💰 ${formatCredits(balance)}`),
        );
      }
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch balance on model select:",
        error instanceof Error ? error.message : error,
      );
    }
  });

  // Refresh credits after each turn
  pi.on("turn_end", async (_event, ctx) => {
    const cred = ctx.modelRegistry.authStorage.get("kilo");
    if (cred?.type !== "oauth") return;

    try {
      const balance = await fetchKiloBalance(cred.access, getEffectiveOrganizationId(cred));
      if (balance !== null) {
        const theme = ctx.ui.theme;
        ctx.ui.setStatus(
          "kilo-credits",
          theme.fg("accent", `💰 ${formatCredits(balance)}`),
        );
      }
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch balance on turn end:",
        error instanceof Error ? error.message : error,
      );
    }
  });

  // On first use of a Kilo model without login, print ToS notice.
  let tosShown = false;

  pi.on("before_agent_start", async (_event, ctx) => {
    if (tosShown) return;
    if (ctx.model?.provider !== "kilo") return;

    const cred = ctx.modelRegistry.authStorage.get("kilo");
    if (cred?.type === "oauth") {
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

  // Use custom footer to show credits inline with token stats
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.setFooter((tui, theme, footerData) => {
      const unsubBranch = footerData.onBranchChange(() => tui.requestRender());

      const formatTokens = (count: number): string => {
        if (count < 1000) return count.toString();
        if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
        if (count < 1000000) return `${Math.round(count / 1000)}k`;
        if (count < 10000000) return `${(count / 1000000).toFixed(1)}M`;
        return `${Math.round(count / 1000000)}M`;
      };

      return {
        dispose() {
          unsubBranch();
        },
        invalidate() {},
        render(width: number): string[] {
          const model = ctx.model;

          // Match built-in footer totals: all assistant messages across all entries
          let totalInput = 0;
          let totalOutput = 0;
          let totalCacheRead = 0;
          let totalCacheWrite = 0;
          let totalCost = 0;
          for (const entry of ctx.sessionManager.getEntries()) {
            if (entry.type === "message" && entry.message.role === "assistant") {
              totalInput += entry.message.usage.input;
              totalOutput += entry.message.usage.output;
              totalCacheRead += entry.message.usage.cacheRead;
              totalCacheWrite += entry.message.usage.cacheWrite;
              totalCost += entry.message.usage.cost.total;
            }
          }

          // Match built-in context usage behavior
          const contextUsage = ctx.getContextUsage();
          const contextWindow = contextUsage?.contextWindow ?? model?.contextWindow ?? 0;
          const contextPercentValue = contextUsage?.percent ?? 0;
          const contextPercent = contextUsage?.percent !== null ? contextPercentValue.toFixed(1) : "?";

          // Build pwd line like built-in (path + branch + session name)
          let pwd = process.cwd();
          const home = process.env.HOME || process.env.USERPROFILE;
          if (home && pwd.startsWith(home)) pwd = `~${pwd.slice(home.length)}`;
          const branch = footerData.getGitBranch();
          if (branch) pwd = `${pwd} (${branch})`;
          const sessionName = ctx.sessionManager.getSessionName();
          if (sessionName) pwd = `${pwd} • ${sessionName}`;

          if (pwd.length > width) {
            const half = Math.floor(width / 2) - 2;
            if (half > 1) {
              pwd = `${pwd.slice(0, half)}...${pwd.slice(-(half - 1))}`;
            } else {
              pwd = pwd.slice(0, Math.max(1, width));
            }
          }

          const statsParts: string[] = [];
          if (totalInput) statsParts.push(`↑${formatTokens(totalInput)}`);
          if (totalOutput) statsParts.push(`↓${formatTokens(totalOutput)}`);
          if (totalCacheRead) statsParts.push(`R${formatTokens(totalCacheRead)}`);
          if (totalCacheWrite) statsParts.push(`W${formatTokens(totalCacheWrite)}`);

          const usingSubscription = model ? ctx.modelRegistry.isUsingOAuth(model) : false;
          if (totalCost || usingSubscription) {
            statsParts.push(`$${totalCost.toFixed(3)}${usingSubscription ? " (sub)" : ""}`);
          }

          const autoIndicator = " (auto)";
          const contextPercentDisplay =
            contextPercent === "?"
              ? `?/${formatTokens(contextWindow)}${autoIndicator}`
              : `${contextPercent}%/${formatTokens(contextWindow)}${autoIndicator}`;

          let contextPercentStr: string;
          if (contextPercentValue > 90) {
            contextPercentStr = theme.fg("error", contextPercentDisplay);
          } else if (contextPercentValue > 70) {
            contextPercentStr = theme.fg("warning", contextPercentDisplay);
          } else {
            contextPercentStr = contextPercentDisplay;
          }
          statsParts.push(contextPercentStr);

          // Inject credits inline on the main stats line
          const creditsStatus = footerData.getExtensionStatuses().get("kilo-credits");
          if (creditsStatus) statsParts.push(creditsStatus);

          let statsLeft = statsParts.join(" ");
          let statsLeftWidth = visibleWidth(statsLeft);

          // Right side: model + thinking + provider like built-in
          const modelName = model?.id || "no-model";
          let rightSideWithoutProvider = modelName;
          if (model?.reasoning) {
            const thinkingLevel = pi.getThinkingLevel() || "off";
            rightSideWithoutProvider =
              thinkingLevel === "off" ? `${modelName} • thinking off` : `${modelName} • ${thinkingLevel}`;
          }

          let rightSide = rightSideWithoutProvider;
          if (footerData.getAvailableProviderCount() > 1 && model) {
            rightSide = `(${model.provider}) ${rightSideWithoutProvider}`;
            if (statsLeftWidth + 2 + visibleWidth(rightSide) > width) {
              rightSide = rightSideWithoutProvider;
            }
          }

          if (statsLeftWidth > width) {
            const plainStatsLeft = statsLeft.replace(/\x1b\[[0-9;]*m/g, "");
            statsLeft = `${plainStatsLeft.substring(0, width - 3)}...`;
            statsLeftWidth = visibleWidth(statsLeft);
          }

          const rightSideWidth = visibleWidth(rightSide);
          const totalNeeded = statsLeftWidth + 2 + rightSideWidth;

          let statsLine: string;
          if (totalNeeded <= width) {
            const padding = " ".repeat(width - statsLeftWidth - rightSideWidth);
            statsLine = statsLeft + padding + rightSide;
          } else {
            const availableForRight = width - statsLeftWidth - 2;
            if (availableForRight > 3) {
              const plainRight = rightSide.replace(/\x1b\[[0-9;]*m/g, "");
              const truncatedRight = plainRight.substring(0, availableForRight);
              const padding = " ".repeat(width - statsLeftWidth - truncatedRight.length);
              statsLine = statsLeft + padding + truncatedRight;
            } else {
              statsLine = statsLeft;
            }
          }

          const dimStatsLeft = theme.fg("dim", statsLeft);
          const remainder = statsLine.slice(statsLeft.length);
          const dimRemainder = theme.fg("dim", remainder);

          return [theme.fg("dim", pwd), dimStatsLeft + dimRemainder];
        },
      };
    });
  });
}
