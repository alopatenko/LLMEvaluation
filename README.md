# LLM Evaluation
# Compendium of LLM Evaluation methods

---
### Reviews and Surveys

### Leaderboards and Arenas
#### LMSys Arena https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
#### MTEB https://huggingface.co/spaces/mteb/leaderboard
#### SWE Bench https://www.swebench.com/
#### Gorilla, Berkeley function calling Leaderboard https://gorilla.cs.berkeley.edu/leaderboard.html https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
#### WildBench WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild https://huggingface.co/spaces/allenai/WildBench
#### Enterprise Scenarios, Patronus https://huggingface.co/blog/leaderboard-patronus
#### Vectara Halluzinations https://github.com/vectara/hallucination-leaderboard

### Evaluation Software
#### MTEB https://huggingface.co/spaces/mteb/leaderboard
#### OpenICL Framework https://arxiv.org/abs/2303.02913
#### RAGAS https://docs.ragas.io/en/stable/
#### EleutherAI LLM Evaluation Harness https://github.com/EleutherAI/lm-evaluation-harness
#### OpenAI Evals https://github.com/openai/evals
#### ML Flow Evaluate https://mlflow.org/docs/latest/llms/llm-evaluate/index.html
#### MosaicML Composer https://github.com/mosaicml/composer
#### TruLens https://github.com/truera/trulens/
#### BigCode Evaluation Harness https://github.com/bigcode-project/bigcode-evaluation-harness
﻿
### LLM Evaluation articles in tech media and blog posts from companies
#### https://techcrunch-com.cdn.ampproject.org/c/s/techcrunch.com/2024/03/23/why-its-impossible-to-review-ais-and-why-techcrunch-is-doing-it-anyway/amp/
#### https://blog.mozilla.ai/exploring-llm-evaluation-at-scale-with-the-neurips-large-language-model-efficiency-challenge/
#### https://www.nytimes.com/2024/04/15/technology/ai-models-measurement.html
#### 

### Large benchmarks
#### SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks EMNLP 2022, https://aclanthology.org/2022.emnlp-main.340.pdf
#### MEASURING MASSIVE MULTITASK LANGUAGE UNDERSTANDING, ICLR, 2021,  https://arxiv.org/pdf/2009.03300.pdf

### Specific benchmarks
#### OpenEQA: From word models to world models, Meta, Apr 2024, Understanding physical spaces by Models,  https://ai.meta.com/blog/openeqa-embodied-question-answering-robotics-ar-glasses/?utm_source=twitter&utm_medium=organic_social&utm_content=video&utm_campaign=dataset
#### ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models, Apr 2024, https://arxiv.org/pdf/2403.20262.pdf

### Evaluation theory, evaluation methods, analysis of evaluation
#### Are We on the Right Way for Evaluating Large Vision-Language Models?, Apr 2024, https://arxiv.org/pdf/2403.20330.pdf
#### Elo Uncovered: Robustness and Best Practices in Language Model Evaluation, Nov 2023 https://arxiv.org/abs/2311.17295
#### Are Emergent Abilities of Large Language Models a Mirage? Apr 23 https://arxiv.org/abs/2304.15004
#### Don't Make Your LLM an Evaluation Benchmark Cheater nov 2023 https://arxiv.org/abs/2311.01964
#### (RE: stat methods ) Prediction-Powered Inference jan 23 https://arxiv.org/abs/2301.09633  PPI++: Efficient Prediction-Powered Inference nov 23, https://arxiv.org/abs/2311.01453

### HITL (Human in the Loop)
### LLM as Judge
Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena Jun 2023, https://arxiv.org/abs/2306.05685
---
## LLM Evaluation
### Embeddings
#### MTEB: Massive Text Embedding Benchmark Oct 2022 https://arxiv.org/abs/2210.07316 Leaderboard https://huggingface.co/spaces/mteb/leaderboard
### In Context Learning
#### HellaSwag,  HellaSwag: Can a Machine Really Finish Your Sentence? 2019, https://arxiv.org/abs/1905.07830 Paper + code + dataset https://rowanzellers.com/hellaswag/ 
####  The LAMBADA dataset: Word prediction requiring a broad discourse context 2016, https://arxiv.org/abs/1606.06031 
### Hallucinations
### Multi Turn
#### LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models Nov 2023, https://arxiv.org/abs/2311.18232
### Reasoning
#### Comparing Humans, GPT-4, and GPT-4V On Abstraction and Reasoning Tasks 2023, [arxiv](https://arxiv.org/abs/2311.09247)
### Multi-Lingual
#### The Invalsi Benchmark: measuring Language Models Mathematical and Language understanding in Italian, Mar 2024, https://arxiv.org/pdf/2403.18697.pdf
### Multi-Modal
### Instruction Following
#### Evaluating Large Language Models at Evaluating Instruction Following Oct 2023, https://arxiv.org/abs/2310.07641
### Ethical AI
#### Evaluating the Moral Beliefs Encoded in LLMs,  Jul 23 https://arxiv.org/abs/2307.14324
#### AI Deception: A Survey of Examples, Risks, and Potential Solutions Aug 23 https://arxiv.org/abs/2308.14752
### Biases
#### FairPair: A Robust Evaluation of Biases in Language Models through Paired Perturbations, Apr 2024 https://arxiv.org/abs/2404.06619v1

### Safe AI
#### Gradient-Based Language Model Red Teaming Jan 24, https://arxiv.org/abs/2401.16656
### Code Generating LLMs
#### Evaluating Large Language Models Trained on Code HumanEval Jul 2022 https://arxiv.org/abs/2107.03374
#### CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation Feb 21 https://arxiv.org/abs/2102.04664
#### Copilot Evaluation Harness: Evaluating LLM-Guided Software Programming Feb 24 https://arxiv.org/abs/2402.14261
#### SWE Bench SWE-bench: Can Language Models Resolve Real-World GitHub Issues? Feb 2024 https://arxiv.org/abs/2402.05863 https://www.cognition-labs.com/post/swe-bench-technical-report
#### Gorilla Functional Calling Leaderboard, Berkeley https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html

### Various unclassified tasks
#### Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study Mar 24, WSDM 24, https://www.microsoft.com/en-us/research/publication/table-meets-llm-can-large-language-models-understand-structured-table-data-a-benchmark-and-empirical-study/
####  LLM Comparative Assessment: Zero-shot NLG Evaluation through Pairwise Comparisons using Large Language Models, jul 2023 https://arxiv.org/abs/2307.07889v3

---

## LLM Systems
### RAG Evaluation
#### RAGAS: Automated Evaluation of Retrieval Augmented Generation Jul 23, https://arxiv.org/abs/2309.15217
#### ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems Nov 23, https://arxiv.org/abs/2311.09476 
### Conversational systems
#### https://towardsdatascience.com/how-to-make-the-most-out-of-llm-production-data-simulated-user-feedback-843c444febc7
### Copilots
### Search and Recommendation Engines
### Task Utility
#### Towards better Human-Agent Alignment: Assessing Task Utility in LLM-Powered Applications, Feb 2024, https://arxiv.org/abs/2402.09015


