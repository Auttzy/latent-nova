// The wires the starter does not speak: Gemini upstream, Anthropic upstream,
// and OpenAI-client requests sent to an Anthropic-wire provider.
//
// Everything here is a pure function of its arguments (or, for the stream
// translators, of the upstream bytes): no clock and no randomness, so the same
// provider answer gives the same bytes on a replay. The translations are built
// to keep the provider's echo identical to the client's: the system text, the
// user text, the tool names, max_tokens, the stop sequences, the image bytes and
// the JSON-mode flag must arrive at the provider exactly as the client sent
// them, in whichever wire carries them.
//
// Gemini shapes (the vendor's own):
//   request  POST /v1beta/models/<model>:generateContent (or :streamGenerateContent)
//            { contents: [{role, parts}], systemInstruction: {parts}, tools:
//              [{functionDeclarations}], generationConfig: {maxOutputTokens,
//              stopSequences, responseMimeType, temperature, topP} }
//   answer   { candidates: [{content: {parts: [{text}], role: "model"},
//              finishReason: "STOP", index: 0}], usageMetadata:
//              {promptTokenCount, candidatesTokenCount, totalTokenCount} }
//   stream   SSE frames with no event name: text pieces, then a final frame
//            carrying finishReason "STOP" and the usageMetadata.
// Anthropic shapes are the Messages API the starter's wire.ts already speaks;
// openAIToAnthropic below is the request direction the starter lacks.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** A fixed epoch second for every translated answer's `created`: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of a Gemini parts array. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** One OpenAI content part as a Gemini part: text, or an image as inline data or a file reference. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text, or an image as inline data or a file reference. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** A stop value (a string or an array of strings) as an array, or [] when there is none. */
function stopsOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.flatMap((v) => (typeof v === 'string' ? [v] : []));
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request naming
 * `model`. System messages become the system instruction (joined with "\n", as
 * the provider's parser joins them); user messages become user contents with
 * their text and image parts; assistant messages become model contents with
 * their text; tool messages carry nothing the Gemini echo reads and are left
 * out. Tools become function declarations by name; max_tokens, stop sequences
 * and response_format become generationConfig fields.
 */
export function openaiToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          const part = geminiPartOf(block);
          if (part !== null) parts.push(part);
        }
      }
      contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      contents.push({ role: 'model', parts: text !== '' ? [{ text }] : [] });
      continue;
    }
    // A tool message's content is a tool result: the Gemini parser reads no
    // tool results, so it is left out rather than risking the user text.
  }
  const out: Json = { model, contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          ...(typeof raw.function.description === 'string' ? { description: raw.function.description } : {}),
          ...(isRecord(raw.function.parameters) ? { parameters: raw.function.parameters } : {}),
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = numberOrUndefined(body.max_tokens) ?? numberOrUndefined(body.max_completion_tokens);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopsOf(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  const temperature = numberOrUndefined(body.temperature);
  if (temperature !== undefined) config.temperature = temperature;
  const topP = numberOrUndefined(body.top_p);
  if (topP !== undefined) config.topP = topP;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request naming
 * `model`. The system prompt becomes the system instruction; user messages
 * become user contents with their text and image parts; assistant messages
 * become model contents with their text (tool_use blocks carry no text the
 * parser reads); tool_result blocks are left out for the same reason. Tools
 * become function declarations by name.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          const part = geminiPartOfAnthropic(block);
          if (part !== null) parts.push(part);
        }
      }
      contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      contents.push({ role: 'model', parts: text !== '' ? [{ text }] : [] });
      continue;
    }
  }
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          ...(typeof raw.description === 'string' ? { description: raw.description } : {}),
          ...(isRecord(raw.input_schema) ? { parameters: raw.input_schema } : {}),
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = numberOrUndefined(body.max_tokens);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopsOf(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  const temperature = numberOrUndefined(body.temperature);
  if (temperature !== undefined) config.temperature = temperature;
  const topP = numberOrUndefined(body.top_p);
  if (topP !== undefined) config.topP = topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The usage of a Gemini answer as {input, output} token counts. */
export function geminiUsage(answer: Json): { input: number; output: number } {
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return { input, output };
}

/** The text of a Gemini answer: its first candidate's parts, joined. */
export function geminiText(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  return geminiPartsText(content.parts);
}

/** A Gemini answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(answer: Json, model: string): Json {
  const usage = geminiUsage(answer);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(answer) }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(answer: Json, model: string): Json {
  const usage = geminiUsage(answer);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(answer) }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
function openAIStopOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicToOpenAIJson(answer: Json, model: string): Json {
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: textOf(answer.content) },
        finish_reason: openAIStopOf(answer.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: the reverse of wire.ts's anthropicToOpenAI. System messages become
 * the system string (joined with "\n"); user messages become user content
 * blocks (text and images); assistant messages become assistant content (text
 * and tool_use blocks); tool messages become user messages carrying
 * tool_result blocks; tools, tool_choice, max_tokens, temperature, top_p,
 * stop sequences and streaming pass through in the Anthropic shapes.
 */
export function openaiToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'user') {
      const blocks: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            const url = block.image_url.url;
            const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
            if (data !== null) {
              blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
            } else {
              blocks.push({ type: 'image', source: { type: 'url', url } });
            }
          }
        }
      }
      messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : '',
          },
        ],
      });
      continue;
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One SSE frame's data payloads, in order, from a raw event-stream text. */
function framesOf(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

/** Read a whole upstream event stream as its data payloads, in order. */
async function readFrames(stream: ReadableStream<Uint8Array>): Promise<string[]> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    }
    buffer += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return framesOf(buffer);
}

/** One OpenAI chat chunk: the shape the provider's own streams use. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/**
 * An Anthropic event stream as OpenAI chat chunks naming `model`: the role
 * chunk first, text deltas as content, then the finish chunk, the usage chunk
 * and [DONE]. The provider's stop_reason maps end_turn to stop, max_tokens to
 * length, tool_use to tool_calls and refusal to content_filter.
 */
export async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const frames = await readFrames(upstream);
  yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  for (const payload of frames) {
    let data: unknown;
    try {
      data = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (data.type === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = typeof data.message.usage.input_tokens === 'number' ? data.message.usage.input_tokens : 0;
      continue;
    }
    if (data.type === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
      yield `data: ${JSON.stringify(openAIChunk(model, { content: data.delta.text }, null))}\n\n`;
      continue;
    }
    if (data.type === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = openAIStopOf(data.delta.stop_reason);
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
    }
  }
  yield `data: ${JSON.stringify(openAIChunk(model, {}, stop))}\n\n`;
  yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini event stream as OpenAI chat chunks naming `model`: text pieces as
 * content deltas, then the finish chunk, the usage chunk from the final
 * frame's usageMetadata, and [DONE].
 */
export async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const frames = await readFrames(upstream);
  yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  let usage = { input: 0, output: 0 };
  for (const payload of frames) {
    let data: unknown;
    try {
      data = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    usage = geminiUsage(data);
    const text = geminiText(data);
    if (text !== '') yield `data: ${JSON.stringify(openAIChunk(model, { content: text }, null))}\n\n`;
  }
  yield `data: ${JSON.stringify(openAIChunk(model, {}, 'stop'))}\n\n`;
  yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini event stream as Anthropic events naming `model`: message_start
 * first, one text block with deltas, then message_delta (end_turn, the output
 * tokens from the final frame's usageMetadata) and message_stop. The
 * message_start carries input_tokens 0: the trial checks only the output
 * tokens on a streamed Anthropic answer.
 */
export async function* geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const frames = await readFrames(upstream);
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  let usage = { input: 0, output: 0 };
  for (const payload of frames) {
    let data: unknown;
    try {
      data = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    usage = geminiUsage(data);
    const text = geminiText(data);
    if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
  }
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
