# FactReview — source links

## 1. Overview

| Status | Count |
|---|---:|
| flawed | 0 |
| questioned | 0 |
| unverified | 1 |
| supported | 0 |

### Items requiring attention

No claim is assessed as flawed or questioned.

## 2. Claim list

### claim\_1 — unverified

Original claim 1\.

Location: page 3. Importance: secondary.

Conditions:

- c1: \{&quot;dataset&quot;: &quot;D&quot;, &quot;metric&quot;: &quot;accuracy&quot;, &quot;settings&quot;: \{\}, &quot;description&quot;: &quot;&quot;\}

Evidence needs: Experiments.

Evidence:

- **paper-internal / support**; sufficient: false; covers: c1. Pointer: paper\.md; page 3; line 7; key table\_2. <a id="factreview-evidence-000001"></a>Evidence E0001.
  - <a id="factreview-source-1911b1fde795e01f422509be2a7ba56c64242397e41cbd1af7a85abcbdedfdf0"></a>Source S0001; occurrences: [claim\_1, evidence 1 / E0001](#factreview-evidence-000001), [claim\_1, evidence 2 / E0002](#factreview-evidence-000002).
  - Passage: Unique original result: A 90\.1; B 80\.2\.
  - Detail: Legacy detail preserved\.
- **paper-internal / support**; sufficient: false; covers: c1. Pointer: paper\.md; page 3; line 7; key table\_2. <a id="factreview-evidence-000002"></a>Evidence E0002.
  - Passage: [Source S0001](#factreview-source-1911b1fde795e01f422509be2a7ba56c64242397e41cbd1af7a85abcbdedfdf0) (same exact source).
  - Detail: Legacy detail preserved\.

Questions for authors:

None recorded.

Notes:

- Claim note 1 remains complete\.

## 3. Other findings

No additional findings recorded.

## 4. Execution ledger

### Run 1

```json
{
  "plan_id": "original-plan",
  "command": [
    "python",
    "eval.py"
  ],
  "logs": "full.log"
}
```
