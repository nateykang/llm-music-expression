#!/usr/bin/env python3
"""Patch AMAAI-Lab/Music2Emotion's music2emo.py to expose the extra signals we need
that its predict() computes internally but doesn't return: the full 56-mood
probability vector, the audio-derived key, the chord sequence, and the 1536-dim MERT
embedding. Idempotent — run once, from inside the cloned Music2Emotion repo.

The injected lines reference locals that exist in predict() at the point of
`return model_output_dic` (mood_list, probs, key_signature, key_type, chords,
final_embedding_mert). If a future upstream revision renames those, re-inspect
music2emo.py and adjust the injection below.
"""

import sys

f = "music2emo.py"
s = open(f).read()
ANCHOR = "        return model_output_dic"
if "mert_embedding" not in s:
    if ANCHOR not in s:
        sys.exit("PATCH FAILED: injection anchor 'return model_output_dic' not found — "
                 "upstream Music2Emotion changed; re-inspect predict() and update this patch")
    for name in ("mood_list", "probs", "key_signature", "key_type", "chords",
                 "final_embedding_mert"):
        if name not in s:
            sys.exit(f"PATCH FAILED: predict() no longer defines '{name}' — "
                     "upstream renamed a local; update the injection below")
    inj = (
        '        model_output_dic["mood_probs"] = {mood_list[i]: round(float(p),4) for i,p in enumerate(probs.squeeze().tolist())}\n'
        '        model_output_dic["key"] = f"{key_signature} {key_type}"\n'
        '        model_output_dic["chords"] = [c for _,_,c in chords]\n'
        '        model_output_dic["mert_embedding"] = [round(float(x),4) for x in final_embedding_mert.tolist()]\n'
        "        return model_output_dic"
    )
    s = s.replace(ANCHOR, inj, 1)
    open(f, "w").write(s)
    print("PATCHED")
else:
    print("already patched")
