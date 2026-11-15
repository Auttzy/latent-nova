// Routing: which free model serves one request, and what a refusal means.
//
// The trial's 20 requests arrive one at a time, and every provider has a per-trial
// call budget the brief never publishes. The sealed provider table the kit ships
// (program/scorer/score.mjs PROVIDERS, and the generator's draws from it) bounds
// each budget between calls_min and calls_max, so planning against the minimum
// never assumes more calls than a trial guarantees. A request's candidates are
// ranked by the quality table's rating first; providers still under their planning
// cap go first, which spreads the load instead of hammering the single best
// provider until it refuses. A refusal's wording says what it is: the env's
// provider answers carry "free-tier budget is spent" once the budget is gone,
// "<name> is out" once its outage begins, and "<name> refused this call" for a
// call on its failure schedule, which is transient — the next call number to the
// same provider may succeed, so it is worth one or two retries before failing
// over. A 400 or 401 is our own mistake, never worth a retry.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './providers.ts';

/** One (provider, model) pair that may serve a request. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  /** The quality table's rating_milli for this model. */
  rating: number;
}

/**
 * The planning cap for one provider: the sealed table's calls_min, the fewest
 * calls a trial of it ever allows. Providers the table does not name get a
 * conservative cap.
 */
export const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

export const DEFAULT_MIN_BUDGET = 4;

export function minBudget(providerId: string): number {
  return MIN_BUDGET[providerId] ?? DEFAULT_MIN_BUDGET;
}

/**
 * Every free model that meets the requirements, best rating first (ties break by
 * provider id, then model id, so the order is deterministic). A JSON-mode request
 * on the OpenAI client wire may only ride a wire that can carry response_format:
 * openai (response_format) and gemini (generationConfig.responseMimeType). The
 * Anthropic wire has no such field, and a provider on it echoes json_mode false,
 * which fails the client's echo check. (An Anthropic client request has no
 * response_format of its own, so it is not restricted here; its model must still
 * have json capability, which meetsRequirements enforces.)
 */
export function candidatesFor(
  brief: Brief,
  probes: readonly Probe[],
  requires: Requirement,
  jsonOnOpenAI: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonOnOpenAI && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/**
 * The order to try one request's candidates in: providers still under their
 * planning cap first (rating order kept inside each group), then the rest, best
 * first. The cap is a planning aid, not a wall: when every candidate is over
 * its cap, the best model still goes first.
 */
export function orderCandidates(candidates: readonly Candidate[], used: ReadonlyMap<string, number>): Candidate[] {
  const under: Candidate[] = [];
  const over: Candidate[] = [];
  for (const candidate of candidates) {
    const spent = used.get(candidate.probe.provider.id) ?? 0;
    (spent < minBudget(candidate.probe.provider.id) ? under : over).push(candidate);
  }
  return [...under, ...over];
}

/** What a provider's refusal means for the next attempt. */
export type FailureKind = 'permanent' | 'transient' | 'bad-request';

/**
 * Classify a provider refusal from its HTTP status and its error message (the
 * env's own wording, in the wire's error envelope). A budget or outage refusal
 * is permanent for the trial; a scheduled refusal or a concurrency refusal is
 * transient; a 400/401 is a request we built wrong and is never retried.
 */
export function classifyFailure(status: number, message: string): FailureKind {
  if (status === 400 || status === 401) return 'bad-request';
  if (message.includes('budget is spent') || message.includes(' is out')) return 'permanent';
  if (message.includes('refused this call') || message.includes('call at a time')) return 'transient';
  if (status === 402 || status === 429 || status >= 500) return 'transient';
  return 'bad-request';
}
