# Reliability & Maintenance Standards — Working Guide

> **Why this file looks like this, not a folder of PDFs**: ISO/IEC/SAE/API standards are commercial, copyrighted documents sold directly by their issuing bodies (roughly USD 80–400 each). There is no free official full text, and sites offering "free PDF" copies are almost always unauthorized. So instead of copies of the documents, this is a **working guide written in our own words**: what each standard is actually for, the points worth remembering day-to-day, and where it touches this codebase. It is not a substitute for the licensed text — treat it as an orientation map, and pull the real document (links at the end of each entry) before you need exact clause wording or a compliance audit.

> **How to use this guide**: these standards are not a simultaneous compliance checklist. They come from different problem domains and different industries (some are oil & gas–specific, some are generic), and adopting one doesn't obligate adopting the rest. Below they're grouped by **what question they actually answer**, so you can see which ones genuinely overlap/reinforce each other and which ones are different enough that they should be evaluated on their own, independently of the others.

---

## Group 1 — "Why does it fail, and what maintenance policy do we choose?" (RCM / FMEA core)

These four standards are tightly coupled: FMEA/FMECA supplies the failure-mode inventory that RCM's decision logic consumes, and the two RCM documents are literally a standard + its own official application guide.

### SAE JA1011 — *Evaluation Criteria for Reliability-Centered Maintenance (RCM) Processes*
Current: 2009 revision widely in force; SAE also lists a newer JA1011_202411 (Nov 2024) edition — confirm which one your license/subscription actually grants before citing a specific date.
- It does **not** describe how to run an RCM analysis — it defines the **minimum bar** a process must clear to be allowed to call itself "RCM". Use it as an audit checklist against a process, not as a how-to.
- Everything hinges on answering **7 questions** per asset, in order: functions & performance standards → functional failures → failure modes → failure effects → failure consequences → proactive tasks (& their applicability/effectiveness) → default actions (redesign/run-to-failure) when no proactive task qualifies.
- It insists on the **operating context** — the same failure mode can warrant a different decision on a duplicated vs. a single critical asset.
- It classifies consequences into 4 buckets (hidden-failure, safety/environmental, operational, non-operational) — this classification is what should drive task selection, not severity alone.
- Official: https://www.sae.org/standards/content/ja1011/

### SAE JA1012 — *A Guide to the Reliability-Centered Maintenance (RCM) Standard*
Current edition in force since 2011.
- Companion to JA1011: expands on *why* each of the 7 questions matters and how to interpret edge cases (e.g. what counts as a "hidden function", how to handle multiple failure effects).
- Worth reading if the RCM assistant needs to justify *why* it asked a particular follow-up question, not just *that* it did.
- Official: https://saemobilus.sae.org/standards/ja1012_201108-a-guide-reliability-centered-maintenance-rcm-standard

### IEC 60300-3-11:2009 — *Dependability management — Application guide: Reliability centred maintenance*
2nd edition (2009), current.
- The IEC "sibling" of SAE JA1011/12, part of the broader IEC 60300 dependability-management series. More prescriptive about the actual **decision-tree algorithm** (a specific flowchart, not just the 7 questions) and about documentation/audit trail requirements.
- Useful specifically for the *decision-logic diagram* itself — if you ever want to hardcode/visualize the RCM decision tree (rather than only prompting an LLM with the 7 questions), this is the document with the actual flowchart.
- Official: https://webstore.iec.ch/en/publication/1296

### IEC 60812:2018 — *Failure modes and effects analysis (FMEA and FMECA)*
3rd edition (2018), replaced the 2006 edition.
- Distinguishes FMEA (qualitative) from FMECA (adds a Criticality step) — a lot of real-world tools blur this; worth being explicit about which one is actually being computed.
- The 2018 edition explicitly added: alternative RPN calculation schemes (not just Severity×Occurrence×Detection), a **criticality-matrix-based method** as an alternative to RPN, and worked examples for software and process FMEAs, not just hardware.
- Severity/Occurrence/Detection are meant to be scored against **defined ranking tables** (not raw arbitrary 1–10 guesses) — the tables themselves are industry/company-specific (AIAG-VDA is the most common reference in automotive, but IEC 60812's own annexes give generic ones too).
- Official: https://webstore.iec.ch/en/publication/26359

**Where this touches the code**: `execute_rcm_node` / `services/llm.py` implement JA1011's 7-question shape; `fmea_calculate_rpn` (`backend/api/analysis.py:1703-1721`) hardcodes RPN bands (<50/<150/<300) with no cited ranking table — the 2018 edition's alternative criticality-matrix method is a direct, low-effort complement here.

---

## Group 2 — "A failure already happened — what caused it, and how do the logic paths combine?"

Different question from Group 1 (a posteriori investigation vs. a priori maintenance planning), but the three standards here share vocabulary (events, gates, cut sets) and are often used together on the same incident.

### IEC 62740:2015 — *Root cause analysis (RCA)*
1st edition (2015).
- Deliberately **method-agnostic**: it doesn't mandate 5-Whys or Ishikawa specifically — it catalogs several RCA techniques (including fault tree–based RCA, barrier analysis, change analysis) and gives criteria for picking one based on the type of event.
- Explicitly scoped to **a posteriori** analysis only (something already happened) — it's not a predictive tool, which is a useful distinction to keep in mind vs. FTA/RBD below.
- Emphasizes that RCA should identify root causes at a level where a **corrective action is actually feasible**, not just "the deepest cause you can name" — stopping too shallow (blame the operator) or too deep (physics) are both flagged as common failure modes of RCA itself.
- Official: https://webstore.iec.ch/en/publication/21810

### IEC 61025:2006 — *Fault tree analysis (FTA)*
2nd edition (2006) — still the current edition, not superseded.
- FTA is top-down: start from an undesired top event, decompose into intermediate/basic events through logic gates (AND, OR, and less common ones like NOT, XOR, k-out-of-n voting gates).
- The standard's core deliverables beyond "compute the top-event probability" are **minimal cut sets** (the smallest combinations of basic events that cause the top event) and **importance measures** (Birnbaum, Fussell-Vesely, risk achievement/reduction worth) — these tell you which basic events actually matter for risk reduction, which a bare probability number doesn't.
- Has a standardized symbol set (gates, event shapes) — useful if a visual fault-tree editor is ever built beyond the current flat list.
- Official: https://webstore.iec.ch/en/publication/4311

### IEC 61078:2016 — *Reliability block diagrams (RBD)*
3rd edition (2016), replaced the 2006 edition.
- RBD is the complementary, bottom-up counterpart to FTA: model the system as blocks in series/parallel/k-out-of-n, and compute availability/reliability/failure-frequency for the whole configuration.
- The 2016 edition specifically added "non-coherent" and "dynamic" RBDs, an electrical-analogy method, and annexes for Boolean-algebra calculation and importance factors — i.e., it now formally covers most of what a real redundancy model needs.
- The standard explicitly documents the mathematical relationship between RBD and FTA (they're dual representations for coherent systems) — meaning the two can share underlying logic rather than being separate features.
- Official: https://webstore.iec.ch/en/publication/25647

**Where this touches the code**: `execute_fta_node` (`backend/api/workbench.py:576-612`) only supports a single AND/OR gate over a flat event list — no nested gates, no minimal cut sets, no importance measures, and no RBD/redundancy modeling exists anywhere in the app. This is the most "standard in name only" corner of the current implementation.

---

## Group 3 — Quantitative life-data & repairable-system statistics

The two standards actually behind the math the app already runs (Weibull and Kijima) — no new capability needed, just closer alignment with what's already computed under the hood.

### IEC 61649:2008 — *Weibull analysis*
2nd edition (2008).
- Covers 2- and 3-parameter Weibull fitting via three method families: graphical (probability plotting), analytical (MLE, which is what this app already uses via the `reliability` Python package), and WeiBayes (when sample size is too small to fit shape+scale independently and a prior/assumed β is used).
- Explicitly expects **confidence bounds on β and η** and a **goodness-of-fit test** to be reported alongside the point estimates — not just the point estimates alone.
- Gives interpretation guidance for β itself (β<1 infant mortality, β≈1 random/electronic-style failures, β>1 wear-out) — useful as user-facing copy next to a fitted β value.
- Official: https://webstore.iec.ch/en/publication/5698

### IEC 61164:2004 — *Reliability growth — Statistical test and estimation methods*
2nd edition (2004), current; companion to IEC 61014 (*Programmes for reliability growth*).
- Not currently referenced anywhere in the app, but it's the actual standard **behind repairable-system / imperfect-repair modeling** — the same problem space Kijima I/II addresses, using NHPP-family models (e.g. the power-law/Crow-AMSAA process) instead of virtual age.
- Gives confidence-interval and goodness-of-fit procedures specifically for **repeated-failure (repairable)** data, which is exactly what the Kijima module fits today without reporting any parameter uncertainty.
- Worth citing next to the Kijima module as "the closest thing to an official standard for what this feature does", even though Kijima I/II themselves come from academic literature (Kijima & Sumita, 1986) rather than from this or any other standard.
- Official (via IEC webstore, search "IEC 61164"): https://webstore.iec.ch

**Where this touches the code**: `ReliabilityFitter.fit_weibull` (`backend/src/reliability_analysis/analysis/models.py:122-207`) already uses censored MLE — correct per IEC 61649 — but doesn't surface the confidence intervals or Anderson-Darling statistic that the underlying `reliability` package already computes internally. `KijimaModel` (`backend/src/reliability_analysis/analysis/kijima_model.py`) reports point estimates only, no CIs — same gap, no standard reference at all today.

---

## Group 4 — System-level availability / production assurance

### ISO 20815:2018 — *Petroleum, petrochemical and natural gas industries — Production assurance and reliability management*
2nd edition (2018), reconfirmed current in 2024.
- Despite the oil & gas–specific title, its production-assurance framework (define regularity targets → model the system with RBDs/Markov/simulation → run sensitivity on the drivers → feed decisions back into design/operations) is broadly applicable to any plant availability study.
- Expects genuine **stochastic simulation** (typically Monte Carlo over TBF/TTR distributions, replicated to produce a distribution of outcomes — P10/P50/P90, not a single number) and explicit **system configuration modeling** (redundancy, spares logistics as lead-time distributions, not flat multipliers).
- Distinguishes reliability, availability and maintainability as separate, individually trackable metrics feeding into a combined "production assurance" figure — not a single blended percentage.
- Official: https://www.iso.org/standard/69983.html

### NORSOK Z-016 — *Regularity management & reliability technology*
Historical note, not a new recommendation.
- The Norwegian offshore-industry "sibling" of ISO 20815, commonly cited alongside it in older RAM literature. Its scope has been effectively absorbed into ISO 20815's later editions.
- Not recommended as an additional reference for this project — citing it next to ISO 20815 today would be redundant rather than complementary; if in doubt, check current withdrawal status directly at standard.no.

**Where this touches the code**: `compute_ram_simulation` (`backend/api/analysis.py:1724-1826`) is a single deterministic formula — the only randomness in it is cosmetic noise on a display chart, not a real Monte Carlo. No redundancy/system-configuration modeling exists. This is the largest gap between an "ISO 20815" label and what's actually implemented — worth either building a real stochastic engine (Groups 2+4 combined: RBD feeding a Monte Carlo) or softening the label until it does.

---

## Group 5 — Risk-based inspection & asset risk ranking

### API RP 580 — *Risk-Based Inspection Program Elements*
4th edition (2023).
- Defines the minimum program elements for an RBI program: a formal risk assessment (qualitative, semi-quantitative or quantitative), inspection planning driven by that risk, and a management-of-change process to keep the risk assessment current as conditions change.
- Explicitly separates **Probability of Failure (PoF)** — driven by degradation mechanisms, inspection effectiveness, damage rate — from **Consequence of Failure (CoF)** — driven by fluid inventory, safety/environmental/financial impact — and expects them to be combined in a defined risk matrix, not averaged or conflated.

### API RP 581 — *Risk-Based Inspection Technology*
4th edition (January 2025).
- The quantitative companion to RP 580: gives the actual damage-mechanism-specific PoF calculation methods (thinning, cracking, brittle fracture, etc.) and CoF calculation methods (release/consequence modeling), plus how to combine them into a numeric risk value and prioritize inspection intervals/methods.
- A defined, industry-standard **risk matrix with fixed severity/likelihood bands** is central to this standard — this is the concrete alternative to a dataset-self-normalized scatter plot.
- Official (both): https://www.api.org/products-and-services/standards/purchase

**Where this touches the code**: `compute_criticality` (`backend/api/analysis.py:177-275`) splits into quadrants using the **mean of the loaded dataset itself** as the split point — so the same failure history is "critical" or not depending on what else happens to be loaded alongside it, which isn't comparable across plants/datasets the way a fixed-band RBI matrix is.

---

## Group 6 — Asset-management umbrella (optional, broader scope than analysis alone)

### ISO 55000:2024 / ISO 55001:2024 — *Asset management — Vocabulary, overview and principles / Requirements*
2nd editions, both launched July 2024 (replacing the 2014 first editions).
- ISO 55000 gives vocabulary/principles; ISO 55001 gives the auditable "shall" requirements for an asset management **system** (like ISO 9001 does for quality) — RCM, RAM and criticality outputs would sit *inside* this system as inputs to decisions, not be the system itself.
- The 2024 edition added a new clause 4.5 "Asset Management Decision-making" — explicitly formalizing how value, risk and cost trade-offs should be weighed when deciding between competing asset actions — arguably the natural home for tying RCM + RAM + Criticality outputs into one recommendation, if the product's ambition grows past standalone analyses.
- Only worth pursuing if the product's scope is meant to expand toward being a decision/management system, not just a set of analysis tools — otherwise it's a much bigger commitment than the other standards here.
- Official: https://www.iso.org/standard/83053.html (55000) · https://www.iso.org/standard/83054.html (55001)

---

## Group 7 — Functional safety (a genuinely different domain — evaluate on its own)

Flagging this group separately, as requested: it answers a different question ("is a safety instrumented function trustworthy enough to rely on") than everything above, has its own vocabulary (SIL, PFD, SIF), and pulling it in only makes sense if the data actually includes safety-instrumented systems — it shouldn't be evaluated against the same criteria as Groups 1–6.

### IEC 61508 (Ed. 2, 2010) — *Functional safety of electrical/electronic/programmable electronic safety-related systems*
- The umbrella functional-safety standard across all industries; defines Safety Integrity Levels (SIL 1–4) and the full safety lifecycle (hazard analysis → safety requirements → design → validation → maintenance).

### IEC 61511 — *Functional safety — Safety instrumented systems for the process industry*
- The process-industry–specific application of IEC 61508's concepts; this is the one actually cited in oil & gas / process plants rather than 61508 directly.
- Central quantity is **Probability of Failure on Demand (PFD)** for a Safety Instrumented Function, which determines whether it meets its required SIL — this would be a genuinely new capability (not an extension of RAM/Weibull), only worth adding if the product's scope is meant to cover safety instrumented systems specifically.
- Official (all parts, search "IEC 61508" / "IEC 61511"): https://webstore.iec.ch

---

## Suggested priority (impact vs. effort), unchanged from the earlier pass

1. **IEC 61649 — surface CI/AD in the Weibull output.** Low effort: the `reliability` package already computes these internally; the fix is exposing them in `fit_weibull()`'s return dict. High payoff for technical credibility.
2. **API RP 580/581 — fixed-band criticality matrix.** Medium effort, high impact: today's result changes depending on what else is loaded in the same dataset, which isn't comparable across plants.
3. **ISO 14224 — data taxonomy.** High effort (schema change), but the highest-leverage structural fix — it would upgrade FMECA, RCM, and cross-equipment benchmarking all at once. (Full write-up of this one is still in the standards research from the previous pass — ping if you want it folded into this file too.)
4. **IEC 61078 (RBD) + an honest pass on the ISO 20815 label.** High effort, but the only way to make the RAM Simulator's stochastic/redundancy claims match its label.
