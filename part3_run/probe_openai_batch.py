"""2-request toy probe of the OpenAI Batch API on gpt-6.1-sol (none + high)."""
import io, json, time, sys
from dotenv import load_dotenv
load_dotenv(".env")
import openai
c = openai.OpenAI()
for eff in ("low", "high"):
    r = c.responses.create(model="gpt-6.1-sol", input="Reply with the word ok.",
                           reasoning={"effort": eff}, max_output_tokens=2000)
    print("realtime", eff, repr(r.output_text), r.usage.output_tokens)
lines = [json.dumps({"custom_id": f"p{i}", "method": "POST", "url": "/v1/responses",
                     "body": {"model": "gpt-6.1-sol", "input": "Reply with the word ok.",
                              "reasoning": {"effort": e}, "max_output_tokens": 2000}})
         for i, e in enumerate(("low", "high"))]
buf = io.BytesIO("\n".join(lines).encode()); buf.name = "probe.jsonl"
f = c.files.create(file=buf, purpose="batch")
b = c.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h")
print("batch", b.id, b.status, flush=True)
for _ in range(40):
    time.sleep(15)
    b = c.batches.retrieve(b.id)
    print(b.status, b.request_counts, b.errors, flush=True)
    if b.status in ("completed", "failed", "expired", "cancelled"):
        break
if b.output_file_id:
    print(c.files.content(b.output_file_id).text[:1500])
