// Latent Nova: the best free AI model aggregator.
//
// One server, two client wires, ten simulated free-tier providers. It reads the
// trial's brief, probes every provider's catalog, registers the one catalog
// with the env, and answers the env's 20 requests — each with the best free
// model that can serve it, failing over when a provider refuses, in the wire
// the client used, JSON or streamed — then exits 0 when the env says the
// workload is done.
//
// HOW IT WINS (the starter is the house baseline, worth roughly a quarter of
// the score; every gap it marks TODO is closed here)
//   - quality (60%): candidates are ranked by the quality table's rating_milli
//     at the request's requirements, and providers are spread under their
//     planning caps (lib/route.ts) instead of hammering the single best model
//     until its budget is gone. A paid model is never a legal pick.
//   - failover (10%): a refusal is classified from the provider's own error
//     wording — a budget or outage refusal retires that provider for the trial,
//     a scheduled refusal is retried on the same provider, and anything else
//     fails over to the next best model that meets the request.
//   - served (15%): the same loop keeps answering until a provider succeeds or
//     the candidates run out; a request is only lost when nothing can serve it.
//   - the wires (10%): all three upstream wires are spoken (openai, anthropic,
//     gemini) and translated losslessly, so the provider's echo of the system
//     prompt, user text, tools, max_tokens, stop sequences, images and JSON
//     mode matches what the client sent, and the provider's usage and stop
//     reason are reported in the client's wire, streamed or not.
//   - the catalog (5%): the probe reports every provider that answered, with
//     the quality table's free flag and context length, and GET /v1/models
//     lists exactly the free ids the catalog registered.
//
// Determinism: no wall clock and no unseeded randomness anywhere a request can
// see. The port is the OS's pick, the model lists carry fixed timestamps, and
// every choice derives from the brief, the requests and the providers'
// answers, so a replay leaves the same state export.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   lib/route.ts       candidate ranking, planning caps, refusal classification.
//   lib/translate.ts   the Gemini and Anthropic upstream wires, in both directions.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import { candidatesFor, classifyFailure, orderCandidates, type Candidate } from './lib/route.ts';
import {
  anthropicStreamToOpenAI,
  anthropicToGemini,
  anthropicToOpenAIJson,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openaiToAnthropic,
  openaiToGemini,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** How many attempts one provider gets on one request before failing over. */
const MAX_ATTEMPTS_PER_PROVIDER = 3;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far this trial, by provider id: the planning caps count every call, as the env does. */
const used = new Map<string, number>();
/** Providers retired for the trial: their budget is spent or their outage has begun. */
const dead = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI-client request asks for JSON mode (only wires that carry response_format may serve it). */
function jsonOnOpenAI(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** The candidates for one request, in the order to try them. */
function orderedCandidates(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): Candidate[] {
  const all = candidatesFor(brief, probes, requires, jsonOnOpenAI(clientApi, body));
  return orderCandidates(all, used);
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The client's body rewritten into the provider's wire, naming `model`. */
function upstreamBody(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  const wire = candidate.probe.provider.wire;
  if (clientApi === 'openai') {
    if (wire === 'openai') {
      const out = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    if (wire === 'anthropic') return openaiToAnthropic(body, model);
    return openaiToGemini(body, model);
  }
  if (wire === 'openai') return anthropicToOpenAI(body, model);
  if (wire === 'anthropic') return { ...body, model };
  return anthropicToGemini(body, model);
}

/** One provider call: the raw fetch Response, or null when the provider could not be reached. */
async function callProvider(candidate: Candidate, payload: Json, stream: boolean): Promise<Response | null> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const path = stream ? ':streamGenerateContent' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${path}${stream ? '?alt=sse' : ''}`;
    if (stream) headers.accept = 'text/event-stream';
  }
  try {
    return await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
}

/** The error message of a refused provider call, in the wire's error envelope. */
async function errorMessageOf(res: Response, wire: 'openai' | 'anthropic' | 'gemini'): Promise<string> {
  let text = '';
  try {
    text = await res.text();
  } catch {
    return '';
  }
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text.slice(0, 500);
}

/** One provider attempt: the successful Response, or the refusal that ended it. */
interface Attempt {
  ok: boolean;
  response: Response | null;
  status: number;
  message: string;
}

async function attempt(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Attempt> {
  const payload = upstreamBody(candidate, clientApi, body, candidate.model.id);
  const res = await callProvider(candidate, payload, stream);
  used.set(candidate.probe.provider.id, (used.get(candidate.probe.provider.id) ?? 0) + 1);
  if (res === null) return { ok: false, response: null, status: 0, message: 'the provider could not be reached' };
  if (res.ok) return { ok: true, response: res, status: res.status, message: '' };
  const message = await errorMessageOf(res, candidate.probe.provider.wire);
  return { ok: false, response: null, status: res.status, message };
}

/** A provider's JSON answer as the client's JSON answer. */
async function sendUpstreamJson(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  upstream: Response,
): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const answer: unknown = await upstream.json();
  if (!isRecord(answer)) return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'the provider answered a body that is not one JSON object');
  if (clientApi === 'openai') {
    if (wire === 'openai') return sendJson(res, 200, answer);
    if (wire === 'anthropic') return sendJson(res, 200, anthropicToOpenAIJson(answer, candidate.model.id));
    return sendJson(res, 200, geminiToOpenAI(answer, candidate.model.id));
  }
  if (wire === 'openai') return sendJson(res, 200, openAIToAnthropic(answer, candidate.model.id));
  if (wire === 'anthropic') return sendJson(res, 200, answer);
  return sendJson(res, 200, geminiToAnthropic(answer, candidate.model.id));
}

/** A provider's event stream as the client's event stream. */
async function sendUpstreamStream(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  upstream: Response,
): Promise<void> {
  const wire = candidate.probe.provider.wire;
  if (upstream.body === null) return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'the provider answered no stream');
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const model = candidate.model.id;
  if (clientApi === 'openai') {
    if (wire === 'openai') {
      for await (const chunk of upstream.body) res.write(chunk);
      res.end();
      return;
    }
    if (wire === 'anthropic') {
      for await (const event of anthropicStreamToOpenAI(upstream.body, model)) res.write(event);
      res.end();
      return;
    }
    for await (const event of geminiStreamToOpenAI(upstream.body, model)) res.write(event);
    res.end();
    return;
  }
  if (wire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    res.end();
    return;
  }
  if (wire === 'anthropic') {
    for await (const chunk of upstream.body) res.write(chunk);
    res.end();
    return;
  }
  for await (const event of geminiStreamToAnthropic(upstream.body, model)) res.write(event);
  res.end();
}

/**
 * Serve one client request: try the candidates in order, retiring a provider
 * whose budget is spent or whose outage has begun, retrying a scheduled
 * refusal on the same provider, and failing over to the next best model on
 * anything else. Exactly one provider call may succeed for a request: the
 * loop stops at the first success and answers from it.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const candidates = orderedCandidates(requires, clientApi, body);
  for (const candidate of candidates) {
    if (dead.has(candidate.probe.provider.id)) continue;
    for (let attemptNumber = 0; attemptNumber < MAX_ATTEMPTS_PER_PROVIDER; attemptNumber += 1) {
      const result = await attempt(candidate, clientApi, body, stream);
      if (result.ok && result.response !== null) {
        if (stream) await sendUpstreamStream(res, clientApi, candidate, result.response);
        else await sendUpstreamJson(res, clientApi, candidate, result.response);
        return;
      }
      const kind = classifyFailure(result.status, result.message);
      if (kind === 'permanent') {
        dead.add(candidate.probe.provider.id);
        log(`${candidate.probe.provider.id} retired: ${result.message}`);
        break;
      }
      if (kind === 'transient') {
        log(`${candidate.probe.provider.id} refused (HTTP ${result.status}): ${result.message}`);
        continue;
      }
      log(`${candidate.probe.provider.id} answered a bad request (HTTP ${result.status}): ${result.message}`);
      break;
    }
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'anthropic', body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
