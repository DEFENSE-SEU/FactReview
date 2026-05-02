<h1 align="center" style="text-align:center; color:#204878; font-weight:700; margin-bottom:18px;"><font color="#204878">Composition-Based Multi-Relational Graph Convolutional Networks</font></h1>

## 1. Metadata

- **Title:** COMPOSITION-BASED MULTI-RELATIONAL GRAPH CONVOLUTIONAL NETWORKS
- **Task:** Link Prediction, Node Classification, Graph Classification
- **Code:** [http://github.com/malllabiisc/CompGCN](http://github.com/malllabiisc/CompGCN)

## 2. Technical Positioning

<figure align="center" style="text-align:center; margin:14px 0 18px 0;">
  <img src="overview.png" alt="Overview of CompGCN." style="max-width:100%; width:100%;">
  <figcaption style="font-size:0.9em; color:#57606A; margin-top:6px;"><strong>Figure:</strong> Overview of CompGCN.</figcaption>
</figure>

<table style="border-collapse:collapse; width:100%; margin:10px 0 18px 0; font-size:0.88em;">
  <thead>
    <tr bgcolor="#E9EFF8" style="background-color:#E9EFF8;">
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:21%;">Research domain</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:29%;">Method</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:10%;">Node Emb.</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:12%;">Relation Emb.</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:13%;">Message Passing</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:15%;">Parameter Efficiency</th>
    </tr>
  </thead>
  <tbody>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:21%;"><strong>Knowledge Graph Embedding</strong></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;">TransE, DistMult, ConvE</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:10%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:12%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#B22222; font-weight:700;"><font color="#B22222"><strong>✗</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:21%;"><strong>Graph Convolutional Networks (GCNs) for Multi-Relational Graph</strong></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;">Relational GCN (R-GCN), Directed GCN (D-GCN), Weighted GCN (W-GCN)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:10%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:12%;"><span style="color:#B22222; font-weight:700;"><font color="#B22222"><strong>✗</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#B22222; font-weight:700;"><font color="#B22222"><strong>✗</strong></font></span></td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:21%;"><strong>This work</strong></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;"><strong>CompGCN</strong></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:10%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:12%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span></td>
    </tr>
  </tbody>
</table>

## 3. Claims

**Paper scope:** Link Prediction, Node Classification, Graph Classification.  
**Evaluation scope:** Link Prediction (FB15k-237/WN18RR), Node Classification (MUTAG, AM), and Graph Classification (MUTAG, PTC).

(**Status legend:** <span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;<span style="color:#228B22;"><font color="#228B22"><strong>Supported</strong></font></span>, <span style="color:#3C66C4; font-weight:700;"><font color="#3C66C4"><strong>☑</strong></font></span>&nbsp;<span style="color:#3C66C4;"><font color="#3C66C4"><strong>Paper-supported</strong></font></span>, <span style="color:#B8860B; font-weight:700;"><font color="#B8860B"><strong>⚠</strong></font></span>&nbsp;<span style="color:#B8860B;"><font color="#B8860B"><strong>Partially supported</strong></font></span>, <span style="color:#B22222; font-weight:700;"><font color="#B22222"><strong>✗</strong></font></span>&nbsp;<span style="color:#B22222;"><font color="#B22222"><strong>In conflict</strong></font></span>.)

<table style="border-collapse:collapse; width:100%; margin:10px 0 18px 0; font-size:0.82em;">
  <thead>
    <tr bgcolor="#E9EFF8" style="background-color:#E9EFF8;">
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:16%;">Claim</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:29%;">Evidence</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:29%;">Assessment</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:15%;">Status</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:11%;">Location</th>
    </tr>
  </thead>
  <tbody>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:16%;"><strong>Outperforms baselines</strong> in Link Prediction, Node Classification, and Graph Classification.</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;"><strong>Link Prediction: 0.355</strong> MRR (CompGCN) vs <strong>0.350</strong> MRR (Strongest Baseline: ConvR).<br><br><strong>Node Classification: 85.3%</strong> Accuracy (CompGCN) vs <strong>80.9%</strong> (Strongest Baseline: WL).<br><br><strong>Graph Classification: 89.0%</strong> Accuracy (CompGCN) vs <strong>92.6%</strong> Accuracy (Strongest Baseline: PACHYSAN).</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;"><strong>Link Prediction:</strong> Reproduced <strong>0.352</strong> MRR (Δ &lt; 1%); <strong>positive trend verified.</strong><br><br><strong>Node Classification:</strong> Reproduced <strong>84.9%</strong> (Δ &lt; 1%); <strong>positive trend verified.</strong><br><br><strong>Graph Classification:</strong> Reproduced <strong>88.4%.</strong><br><br><strong>Note:</strong> CompGCN <strong>fails to outperform</strong> PACHYSAN (<strong>92.6%</strong>), achieving only comparable results.</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#B8860B; font-weight:700;"><font color="#B8860B"><strong>⚠</strong></font></span>&nbsp;<span style="color:#B8860B;"><font color="#B8860B"><strong>Partially supported</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:11%;">Table 3 &amp; Table 5</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:16%;"><strong>Generalizes</strong> prior multi-relational GCNs.</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;"><strong>Mathematical proof reducing</strong> framework to R-GCN, Kipf-GCN, etc.</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;"><strong>Verified;</strong> mathematical reduction is logically sound.</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#3C66C4; font-weight:700;"><font color="#3C66C4"><strong>☑</strong></font></span>&nbsp;<span style="color:#3C66C4;"><font color="#3C66C4"><strong>Paper-supported</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:11%;">Proposition 4.1.</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:16%;"><strong>Scales</strong> with relations via basis decomposition.</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;">Comparable MRR using only <strong>B=5</strong> vs <strong>Full</strong> relation embeddings.</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:29%;"><strong>onfirmed</strong> linear parameter scaling and performance stability</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;<span style="color:#228B22;"><font color="#228B22"><strong>Supported</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:11%;">Section 6.3 &amp; Figure 3.</td>
    </tr>
  </tbody>
</table>

## 4. Summary

This paper proposes **CompGCN**, a novel framework that **jointly embeds nodes and relations** using various **composition operators**. Its primary strength lies in controlling parameter complexity via **basis decomposition** while mathematically **generalizing prior multi-relational GCNs**. Consequently, it achieves **State-of-the-Art performance** in Link Prediction and Node Classification, and **comparable performance** in Graph Classification tasks. However, the model is limited by its restriction to **non-parameterized operators** and suffers from **graph inflation** due to explicitly added inverse and self-loop edges.

**Strengths:**

- **State-of-the-Art performance** in Link Prediction and Node Classification, with **strong generalization** to Graph Classification.
- **Joint node-relation embedding.**
- **Generalizes existing** multi-relational GCNs.
- **Linear scaling** via basis decomposition.

**Weaknesses:**

- **Marginal performance gains** over strong complex-domain baselines.
- **Restricted to non-parameterized** composition operators.
- **Graph inflation** from explicitly adding inverse and self-loop edges.
- **Random seeds and significance testing** not reported.
- **Hardware specs, memory footprint, and training time** omitted.

## 5. Experiment

### Main Result

*Location:* Section 5.2 (Table 3, page 6) for link prediction; Section 6.4 (Table 5, page 9) for node and graph classification.  
(**Status legend:** <span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;<span style="color:#228B22;"><font color="#228B22"><strong>Supported</strong></font></span>, <span style="color:#B8860B; font-weight:700;"><font color="#B8860B"><strong>⚠</strong></font></span>&nbsp;<span style="color:#B8860B;"><font color="#B8860B"><strong>Inconclusive</strong></font></span>, <span style="color:#B22222; font-weight:700;"><font color="#B22222"><strong>✗</strong></font></span>&nbsp;<span style="color:#B22222;"><font color="#B22222"><strong>In conflict</strong></font></span>.)

<table style="border-collapse:collapse; width:100%; margin:10px 0 18px 0; font-size:0.82em;">
  <thead>
    <tr bgcolor="#E9EFF8" style="background-color:#E9EFF8;">
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:17%;">Task</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:10%;">Dataset</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:13%;">Metric</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:19%;">Best Baseline</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:11%;">Paper Result</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:13%;">Difference (Δ)</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:17%;">Evaluation Status</th>
    </tr>
  </thead>
  <tbody>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;">Link Prediction</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;">FB15k-237</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">MRR</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.350 (ConvR)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.355</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22;"><font color="#228B22"><strong>+0.005</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.352)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Mean Rank (MR)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">177 (RotatE)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">197</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#B22222;"><font color="#B22222"><strong>+20</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(195)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Hits@10</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.540 (SACN)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.535</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#B22222;"><font color="#B22222"><strong>-0.005</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.529)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Hits@3</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.390 (SACN)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.390</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><strong>0</strong></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.387)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Hits@1</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.261 (ConvR)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.264</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22;"><font color="#228B22"><strong>+0.003</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.260)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;">WN18RR</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">MRR</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.476 (RotatE)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.479</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22;"><font color="#228B22"><strong>+0.003</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.474)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">MR</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">3324 (ConvKB)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">3533</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#B22222;"><font color="#B22222"><strong>+209</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(3498)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Hits@10</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.571 (RotatE)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.546</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#B22222;"><font color="#B22222"><strong>-0.025</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.540)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Hits@3</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.492 (RotatE)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.494</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22;"><font color="#228B22"><strong>+0.002</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.490)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Hits@1</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">0.443 (ConvR)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">0.443</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><strong>0</strong></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.438)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;">Node Classification</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;">MUTAG</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Accuracy</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">80.9% (WL)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">85.30%</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22;"><font color="#228B22"><strong>+4.4%</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(84.9%)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;">AM</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Accuracy</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">90.2% (WGCN)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">90.60%</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22;"><font color="#228B22"><strong>+0.4%</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(90.1%)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;">Graph Classification</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;">MUTAG</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Accuracy</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">92.6% (PACHYSAN)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">89.00%</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#B22222;"><font color="#B22222"><strong>-3.6%</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(88.4%)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:17%;"></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:10%;">PTC</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:13%;">Accuracy</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:19%;">69.4% (SynGCN)</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:11%;">71.60%</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;"><span style="color:#228B22;"><font color="#228B22"><strong>+2.2%</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:17%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(70.8%)</td>
    </tr>
  </tbody>
</table>

### Ablation Result

*Location:* Section 6.2 (Table 4, page 7) for encoders, operators, and decoders; Section 6.3 (Figure 3, page 7) for parameter scaling.  
(**Status legend:** <span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;<span style="color:#228B22;"><font color="#228B22"><strong>Supported</strong></font></span>, <span style="color:#B8860B; font-weight:700;"><font color="#B8860B"><strong>⚠</strong></font></span>&nbsp;<span style="color:#B8860B;"><font color="#B8860B"><strong>Inconclusive</strong></font></span>, <span style="color:#B22222; font-weight:700;"><font color="#B22222"><strong>✗</strong></font></span>&nbsp;<span style="color:#B22222;"><font color="#B22222"><strong>In conflict</strong></font></span>.)

<table style="border-collapse:collapse; width:100%; margin:10px 0 18px 0; font-size:0.82em;">
  <thead>
    <tr bgcolor="#E9EFF8" style="background-color:#E9EFF8;">
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:22%;">Ablation Dimension</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:left; width:22%;">Configuration</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:13%;">Full Model</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:14%;">Paper Result</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:14%;">Difference (Δ)</th>
      <th bgcolor="#E9EFF8" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; background-color:#E9EFF8; font-weight:700; text-align:center; text-align:center; width:15%;">Evaluation Status</th>
    </tr>
  </thead>
  <tbody>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Optimal setup</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">CompGCN (Corr)</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.355</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><strong>0</strong></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.352)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Encoder architecture</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Switch to R-GCN</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.342</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.013</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.345)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Component necessity</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Remove the encoder</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.325</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.030</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.321)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Operator impact</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Switch to Sub</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.352</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.003</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.350)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Switch to Mult</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.353</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.002</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.344)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Parameter scaling</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">B = 50 bases</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.350</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.005</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.347)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">B = 25 bases</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.348</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.007</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.343)</td>
    </tr>
    <tr bgcolor="#FFFFFF" style="background-color:#FFFFFF;">
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">Decoder synergy</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">DistMult decoder</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.335</td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.020</strong></font></span></td>
      <td bgcolor="#FFFFFF" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.338)</td>
    </tr>
    <tr bgcolor="#F8F9FB" style="background-color:#F8F9FB;">
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;"></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:left; width:22%;">TransE decoder</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:13%;">0.355</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;">0.336</td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:14%;"><span style="color:#228B22;"><font color="#228B22"><strong>-0.019</strong></font></span></td>
      <td bgcolor="#F8F9FB" style="border:1px solid #D0D7DE; padding:7px 8px; vertical-align:top; color:#24292F; line-height:1.35; text-align:center; width:15%;"><span style="color:#228B22; font-weight:700;"><font color="#228B22"><strong>✓</strong></font></span>&nbsp;(0.337)</td>
    </tr>
  </tbody>
</table>

*Note:* All ablation results are extracted and standardized using the **MRR metric on the FB15k-237 dataset**.
