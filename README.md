# Latent Nova

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Auttzy/latent-nova/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Auttzy/latent-nova/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten free-tier model providers, routing each request to the
best free model that can serve it and failing over when a provider refuses.
Built by an AI agent for [make.fun](https://make.fun)'s ["The best free AI
model aggregator"](https://make.fun/bounty/free-model-aggregator-01) bounty,
$500 in USDC.

## How it works

The server reads the trial brief (`$MAKEFUN_PROMPT_FILE`): ten providers with
their wires, auth schemes and keys, the sealed quality table, and the workload
manifest. It probes every provider's catalog route, builds one catalog, and
registers its port and catalog with the env. Each request is then served by
the highest-rated free model meeting its requirements — tools, vision, JSON
mode, context length — with providers spread under their per-trial planning
caps instead of hammering the single best model until its budget is gone. A
refusal is classified from the provider's own error wording: a spent budget or
an outage retires that provider for the trial, a scheduled refusal is retried
on the same provider, and anything else fails over to the next best model.
All three upstream wires are spoken (OpenAI, Anthropic, Gemini) and translated
losslessly, so the provider's echo of the system prompt, user text, tools,
max_tokens, stop sequences, images and JSON mode matches what the client sent,
and the provider's usage and stop reason are reported in the client's wire,
streamed or not.

## Run

```sh
node main.ts
```

`node main.ts --list` prints the catalog document the server registers, and
exits 0. Node 22, standard library only — nothing is installed.
