# Community reranker playbook

## Status

**Training shelved; 0.8B inference active.** Use the installed Jev-Style 0.8B model for the all-community scoring path. Do not train a 150M student unless observed misses or context cost justify it.

The current implementation scores every eligible community summary, then asks the generation model for partial answers only for summaries above the relevance threshold. It records the full score list. No model training or database change is part of this approach.

### Runtime and batching

- Use Jev-Style's typed `noul` probability for each independent query–summary relevance judgment. Stock Ollama chat can generate text from the GGUF, but it does not expose this model's typed decision scores; use the Jev-Style runtime endpoint (`/v1/systemone`).
- Send 64–128 community summaries per request (default 64), with one independent NOUL question per community in the shared state. Score every eligible summary before generating any partial answers.
- Allow one scoring request on the GPU at a time and pre-serialize one successor request in a bounded CPU queue. This gives at most two in-flight batches: one processing and one ready.
- Keep the score cutoff configurable. The default is 0.5; tune it only against the query/community judgments and evidence-coverage objective below.
- If an input/token or GPU-memory error occurs, split that batch and retry serially. Never truncate a community summary or silently omit a score.

## Goal

If the current workflow needs improvement, test whether a smaller pair scorer can select query-relevant **community summaries and sections** with less context while preserving the evidence needed by the final answer.

Treat these as two related but distinct decisions:

1. **Community lens:** given a query and a community summary, should this community be retained for global context?
2. **Section lens:** given a query and a section (optionally with its community summary), should this section be read or passed to answer synthesis?

The model is useful only if it improves the downstream evidence set. A high pair-classification score by itself is not the goal.

## Task and teacher boundaries

The intended general-purpose 0.8B decision model is [`chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF`](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF). The local Ollama installation also contains [`jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev-scorer-v2-gguf`](https://huggingface.co/jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev-scorer-v2-gguf), which is an NVSH shell-operation candidate scorer with task-specific training and evaluations. Do not treat the NVSH model's candidate-relative next-token scores as general independent query–summary relevance probabilities. The active `/v1/systemone` path uses Jev-Style's typed NOUL decisions from its model runtime; verify the served checkpoint matches the general Jev-Style 0.8B model before running comparisons.

Therefore:

- Continue using the installed 0.8B model as it is currently prompted and integrated; first verify that the local Ollama model is this exact artifact or record the actual model and Modelfile if it differs.
- Treat its community-summary judgments as a baseline to measure, not as established relevance truth.
- Do not reinterpret its candidate-relative option probabilities as independent probabilities that a query–summary or query–section pair is relevant.
- Teacher-generated labels for community/section relevance may be tested only after the teacher is prompted for that exact decision and its agreement with human judgments is measured.
- MS MARCO is query–passage answerability/relevance ranking data, not an entailment dataset. A passage can help answer a query without entailing its wording. Its labels should not be pooled with NVSH option-choice labels as one undifferentiated binary target.

## Evidence required before training

Build a fixed, representative evaluation set from actual GraphRAG questions and the current section corpus. Each query should have judgments at both levels:

- relevant communities, based on whether their summaries contribute evidence to answering the query;
- relevant sections, based on whether the section contains useful evidence, a necessary distinction, or a material counterexample.

Include mixed and difficult cases: multiple relevant communities, relevant evidence buried among similar communities, broad questions, questions about methods/taxonomies, and sections that are topically similar but do not answer the question. Preserve the current retrieval candidates and also record the gold evidence found outside the current candidate set where feasible. A reranker cannot recover evidence that candidate generation never surfaces.

Split by query and group paraphrases or related query variants together before splitting. Do not let the same question, paraphrase, or templated variant appear across training, validation, and test. Keep the test set sealed until model and threshold choices are complete.

For an initial pilot, target at least a few hundred distinct query cases if the corpus and annotation budget permit. Double-review a sample of labels, adjudicate disagreements, and report the number and provenance of judgments. If there are fewer cases, report the wider uncertainty rather than treating a noisy point estimate as conclusive.

### Primary measurements

Evaluate selection at fixed context/token budgets, since the purpose is to control what enters synthesis:

- **Evidence recall at budget:** fraction of judged-relevant sections retained within the available token budget. This is the primary metric.
- **Community recall at budget:** fraction of judged-relevant communities retained.
- **Context cost:** tokens and number of communities/sections retained.
- **Ranking quality:** nDCG or MRR over candidate communities and sections.
- **Miss rate:** fraction of queries where a necessary community or section is dropped.
- **Downstream answer quality:** completeness and support/citation correctness when the same answer process receives each system's selected evidence.

Report metrics by query type and granularity as well as overall. Choose the allowed miss rate and token budget before evaluating the final test set. For this application, evidence recall and missed-answer-critical sections should take priority over maximizing precision.

## Baselines

Run every baseline against identical queries and candidate pools:

1. Current production workflow and current 0.8B JEV prompt/integration.
2. Existing retrieval ranking before any JEV filtering, if available.
3. A task-matched pretrained query–passage CrossEncoder, such as an MS MARCO MiniLM model, as a reference point. It is trained for web passage ranking, so it is not presumed to fit research-community summaries.
4. GPT-X3-150M with a freshly initialized, untrained classification head as a chance-level sanity check only.
5. GPT-X3-150M with a trained head and frozen backbone.
6. GPT-X3-150M with LoRA/adapters and a trained relevance head.

The untrained head is not a useful quality baseline: it gives random scores. The frozen-backbone trained-head condition distinguishes “the existing representation is enough” from “the backbone must adapt.” Compare the 150M conditions to the 0.8B model and the task-matched reranker on the same held-out cases. Smaller parameter count is a latency/memory advantage, not evidence of higher relevance quality.

## Data construction if the pilot justifies training

Construct two labeled datasets, with explicit task names and separate evaluation:

### Community relevance

Input: query plus community summary. Label whether the community should be retained as useful evidence for answering that query. Prefer graded labels if reviewers can distinguish directly useful, supporting/contextual, and irrelevant communities; otherwise use a clearly defined binary target.

### Section relevance

Input: query plus section text, optionally including the community summary/title as context. Label whether the section should be read or passed to synthesis. Include near-miss negatives and counterevidence, not only random unrelated sections.

Use real query-candidate groups. For MS MARCO-derived training, use query groups with positive passages and realistic hard negatives; do not assume every unlabeled passage is a true negative. Keep MS MARCO examples as a separate training source/task and measure them separately. They may provide general relevance warm-up, but only the community/section evaluation establishes success for this project.

If the 0.8B model is tested as a teacher, store its exact prompt, raw response/logits if available, model revision, and candidate set. Have humans audit a stratified sample. Train on teacher outputs only if the teacher demonstrates useful agreement on the relevant community/section decisions. Preserve human-labeled test cases as an independent check against teacher imitation.

## Model and packaging options

Sentence Transformers supports CrossEncoders that score paired inputs jointly. A standard pair-classification or regression head can serve the relevance task. A decoder can also be wrapped as a CrossEncoder, but the input format and score extraction must match the model architecture.

For a future 150M experiment, compare:

- a trained classification head with the base frozen;
- LoRA plus the relevance head;
- optionally, full fine-tuning only if the hardware/time budget supports it and it is justified by pilot results.

Keep community and section labels distinguishable. Start with separate heads or task-specific adapters if both levels are trained; share the base only if held-out results support sharing. Calibrate community and section scores separately. Save the Transformers checkpoint and tokenizer as the training source; produce a GGUF quantized deployment artifact only after the model and inference path are selected.

## Training budget and plateau rule

Budget: **at most two hours of training wall time**, if possible. This budget excludes data preparation and human labeling, which must be reported separately.

1. Profile a short run on the intended hardware and sequence lengths.
2. Estimate the number of examples/tokens and optimizer steps that fit the remaining budget. “2k–32k records” is a data-size range, not a step count; one query with many candidate pairs is not equivalent to one independent query.
3. Evaluate on validation data at regular intervals. Track the primary evidence-recall-at-budget metric, context cost, ranking quality, and downstream answer measurements where available. Training loss is diagnostic, not the stopping criterion.
4. Stop after three consecutive validation checks show no improvement larger than evaluation uncertainty on the primary metric, or sooner if the two-hour cap is reached. Keep the best validation checkpoint, not automatically the final checkpoint.
5. Run the sealed test set once for the selected configuration and compare paired query-level results with confidence intervals. If the test result is inconclusive, report it as inconclusive rather than tuning against it.

Record the model/tokenizer revision, data version and hashes, split manifest, labels and provenance, hardware, software versions, seed, precision/quantization, LoRA/head settings, sequence length, batch/gradient accumulation, optimizer and learning rate, checkpoints, elapsed time, examples/tokens seen, and every validation score. The plateau record must name the best step, wall time, metric values, and uncertainty interval.

## Decision rule

Training is justified only if the selected 150M model:

- improves evidence recall at the same context budget, or preserves recall while materially reducing context cost;
- does not increase answer-critical evidence misses beyond the predeclared tolerance;
- maintains or improves downstream answer completeness and support;
- shows a repeatable held-out gain over the current 0.8B workflow, with uncertainty reported;
- fits the intended runtime and training budgets.

If it does not meet these conditions, keep the current 0.8B model or disable the reranking filter. A quality plateau below the current workflow is a valid stopping result, not a reason to continue spending training time.

## References

- [Jev-Style 0.8B Decision v3 model card](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF)
- [NVSH-specific 0.8B scorer model card](https://huggingface.co/jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev-scorer-v2-gguf)
- [Microsoft MS MARCO ranking task and data](https://github.com/microsoft/MSMARCO/blob/master/Datasets.md)
- [Sentence Transformers CrossEncoder training guide](https://sbert.net/docs/cross_encoder/training_overview.html)
- [Sentence Transformers custom CrossEncoder models](https://sbert.net/docs/cross_encoder/usage/custom_models.html)
- [Sentence Transformers MS MARCO training example](https://www.sbert.net/examples/cross_encoder/training/ms_marco/README.html)
