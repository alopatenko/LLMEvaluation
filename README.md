
# Evaluation of LLM and LLM based Systems
# Compendium of LLM Evaluation methods
---
### Introduction
The aim of this compendium is to assist academics and industry professionals in creating effective evaluation suites tailored to their specific needs. It does so by reviewing the top industry practices for assessing large language models (LLMs) and their applications. This work goes beyond merely cataloging benchmarks and evaluation studies; it encompasses a comprehensive overview of all effective and practical evaluation techniques, including those embedded within papers that primarily introduce new LLM methodologies and tasks. I plan to periodically update this survey with any noteworthy and shareable evaluation methods that I come across.
I aim to create a resource that will enable anyone with queries—whether it's about evaluating a large language model (LLM) or an LLM application for specific tasks, determining the best methods to assess LLM effectiveness, or understanding how well an LLM performs in a particular domain—to easily find all the relevant information needed for these tasks. Additionally, I want to highlight various methods for evaluating the evaluation tasks themselves, to ensure that these evaluations align effectively with business or academic objectives.

My view on LLM Evaluation: [Deck](LLMEvaluation.pdf), and [SF Big Analytics and AICamp](https://www.youtube.com/watch?v=dW89BHjLA4M) [video Analytics Vidhya](https://community.analyticsvidhya.com/c/datahour/evaluating-llms-and-llm-systems-pragmatic-approach) ([Data Phoenix Mar 5](https://www.youtube.com/watch?v=spgVnMgvLSw)) (by [Andrei Lopatenko](https://www.linkedin.com/in/lopatenko/))

## [The github repository](https://github.com/alopatenko/LLMEvaluation)

![Evals are surprisingly often all you need](greg.png)

# Table of contents
- [Reviews and Surveys](#reviews-and-surveys)
- [Leaderboards and Arenas](#leaderboards-and-arenas)
- [Evaluation Software](#evaluation-software)
- [LLM Evaluation articles in tech media and blog posts from companies](#llm-evaluation-articles-in-tech-media-and-blog-posts-from-companies)
- [Large benchmarks](#large-benchmarks)
- [Evaluation of evaluation, Evaluation theory, evaluation methods, analysis of evaluation](#evaluation-of-evaluation-evaluation-theory-evaluation-methods-analysis-of-evaluation)
- [Long Comprehensive Studies](#long-comprehensive-studies)
- [HITL (Human in the Loop)](#hitl-human-in-the-loop)
- [LLM as Judge](#llm-as-judge)
- [LLM Evaluation](#llm-evaluation)
    - [Embeddings](#embeddings)
    - [In Context Learning](#in-context-learning)
    - [Hallucinations](#hallucinations)
    - [Question Answering](#question-answering)
    - [Multi Turn](#multi-turn)
    - [Reasoning](#reasoning)
    - [Multi-Lingual](#multi-lingual)
    - [Multi-Modal](#multi-modal)
    - [Instruction Following](#instruction-following)
    - [Ethical AI](#ethical-ai)
    - [Biases](#biases)
    - [Safe AI](#safe-ai)
    - [Cybersecurity](#cybersecurity)
    - [Code Generating LLMs](#code-generating-llms)
    - [Summarization](#summarization)
    - [LLM  quality (generic methods: overfitting, redundant layers etc)](#llm--quality-generic-methods-overfitting-redundant-layers-etc)
    - [Inference Performance](#inference-performance)
    - [Agent LLM architectures](#agent-llm-architectures)
    - [Long Text Generation](#long-text-generation)
    - [Graph Understandings](#graph-understanding)
    - [Reward Models](#reward-models)
    - [Various unclassified tasks](#various-unclassified-tasks)
- [LLM Systems](#llm-systems)
    - [RAG Evaluation](#rag-evaluation)
    - [Conversational systems](#conversational-systems)
    - [Copilots](#copilots)
    - [Search and Recommendation Engines](#search-and-recommendation-engines)
    - [Task Utility](#task-utility)
  - [Verticals](#verticals)
    - [Healthcare and medicine](#healthcare-and-medicine)
    - [Law](#law)
    - [Science (generic)](#science)
    - [Financial](#financial)
- [Other collections](#other-collections)
- [Citation](#citation)
---
### Reviews and Surveys
- Benchmark Evaluations, Applications, and Challenges of Large Vision Language Models: A Survey, UMD,  Jan 2025, [arxiv](https://www.arxiv.org/abs/2501.02189)
- AI Benchmarks and Datasets for LLM Evaluation, Dec 2024, [arxiv](https://arxiv.org/abs/2412.01020), a survey of many LLM benchmarks
- LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods, Dec 2024, [arxiv](https://arxiv.org/abs/2412.05579)
- A Systematic Survey and Critical Review on Evaluating Large Language Models: Challenges, Limitations, and Recommendations, EMNLP 2024, [ACLAnthology](https://aclanthology.org/2024.emnlp-main.764/)
- A Survey on Evaluation of Multimodal Large Language Models, aug 2024, [arxiv](https://arxiv.org/abs/2408.15769)
- A Survey of Useful LLM Evaluation, Jun 2024, [arxiv](https://arxiv.org/abs/2406.00936)
- Evaluating Large Language Models: A Comprehensive Survey , Oct 2023 [arxiv:](https://arxiv.org/abs/2310.19736)
- A Survey on Evaluation of Large Language Models Jul 2023 [arxiv:](https://arxiv.org/abs/2307.03109)
- Through the Lens of Core Competency: Survey on Evaluation of Large Language Models, Aug 2023 , [arxiv:](https://arxiv.org/abs/2308.07902)
- for industry-specific surveys of evaluation methods for industries such as medical, see in respective parts of this compendium

---
### Leaderboards and Arenas
-  New Hard Leaderboard by HuggingFace [leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) [description, blog post](https://huggingface.co/spaces/open-llm-leaderboard/blog)
- The FACTS Grounding Leaderboard: Benchmarking LLMs' Ability to Ground Responses to Long-Form Input, DeepMind, Jan 2025, [arxiv](https://arxiv.org/abs/2501.03200) [Leaderboard](https://www.kaggle.com/facts-leaderboard)
- [LMSys Arena]( https://chat.lmsys.org/?leaderboard) ([explanation:]( https://lmsys.org/blog/2023-05-03-arena/))
- Aider Polyglot, code edit benchmark, [Aider Polyglot](https://aider.chat/docs/leaderboards/)
- Salesforce's Contextual Bench leaderboard [hugging face](https://huggingface.co/spaces/Salesforce/ContextualBench-Leaderboard)  an overview of how different LLMs perform across a variety of contextual tasks,
- [ArenaHard Leaderboard](https://github.com/lmarena/arena-hard-auto?tab=readme-ov-file#leaderboard) Paper: From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline, UC Berkeley, Jun 2024, [arxiv](https://arxiv.org/abs/2406.11939) [github repo](https://github.com/lmarena/arena-hard-auto)   ArenaHard benchmark
- OpenGPT-X Multi- Lingual European LLM Leaderboard, [evaluation of LLMs for many European languages - on HuggingFace](https://huggingface.co/spaces/openGPT-X/european-llm-leaderboard)
- [AllenAI's ZeroEval LeaderBoard](https://huggingface.co/spaces/allenai/ZeroEval)  benchmark: [ZeroEval from AllenAI](https://github.com/WildEval/ZeroEval)  unified framework for evaluating (large) language models on various reasoning tasks  
- [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [MTEB ](https://huggingface.co/spaces/mteb/leaderboard)
- [SWE Bench ](https://www.swebench.com/)
- [AlpacaEval leaderboard](https://tatsu-lab.github.io/alpaca_eval/) Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators, Apr 2024, [arxiv](https://arxiv.org/abs/2404.04475)  [code](https://github.com/tatsu-lab/alpaca_eval)
- [Open Medical LLM Leaderboard from HF](https://huggingface.co/blog/leaderboard-medicalllm) [Explanation](https://huggingface.co/blog/leaderboard-medicalllm)
- [Gorilla, Berkeley function calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) [Explanation ](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- [WildBench WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild ](https://huggingface.co/spaces/allenai/WildBench)
- [Enterprise Scenarios, Patronus ](https://huggingface.co/blog/leaderboard-patronus)
- [Vectara Hallucination Leaderboard ]( https://github.com/vectara/hallucination-leaderboard)
- [Ray/Anyscale's LLM Performance Leaderboard]( https://github.com/ray-project/llmperf-leaderboard) ([explanation:]( https://www.anyscale.com/blog/comparing-llm-performance-introducing-the-open-source-leaderboard-for-llm))
- Hugging Face LLM Performance [hugging face leaderboard](https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard)
- [Multi-task Language Understanding on MMLU](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu)
  
---
### Evaluation Software
- [EleutherAI LLM Evaluation Harness ](https://github.com/EleutherAI/lm-evaluation-harness)
- Eureka, Microsoft, A framework for standardizing evaluations of large foundation models, beyond single-score reporting and rankings. [github](https://github.com/microsoft/eureka-ml-insights) Sep 2024 [arxiv](https://arxiv.org/abs/2409.10566)
- [OpenAI Evals]( https://github.com/openai/evals)
- [Phoenix Arize AI LLM observability and evaluation platform](https://github.com/Arize-ai/phoenix)
- [MTEB](https://huggingface.co/spaces/mteb/leaderboard)
- [OpenICL Framework ](https://arxiv.org/abs/2303.02913)
- [RAGAS]( https://docs.ragas.io/en/stable/)
- [Confident-AI DeepEval The LLM Evaluation Framework](https://github.com/confident-ai/deepeval) (unittest alike evaluation of LLM outputs)
- [ML Flow Evaluate ](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [MosaicML Composer ](https://github.com/mosaicml/composer)
- [Microsoft Prompty](https://github.com/microsoft/prompty) 
- [NVidia Garac evaluation of LLMs vulnerabilities](https://github.com/NVIDIA/garak) Generative AI Red-teaming & Assessment Kit
- [Toolkit from Mozilla AI for LLM as judge evaluation](https://blog.mozilla.ai/local-llm-as-judge-evaluation-with-lm-buddy-prometheus-and-llamafile/) tool: [lm-buddy eval tool](https://github.com/mozilla-ai/lm-buddy?ref=blog.mozilla.ai) model: [Prometheus](https://kaistai.github.io/prometheus/)
- [ZeroEval from AllenAI](https://github.com/WildEval/ZeroEval)  unified framework for evaluating (large) language models on various reasoning tasks  [LeaderBoard](https://huggingface.co/spaces/allenai/ZeroEval)
- [TruLens ](https://github.com/truera/trulens/)
- [Promptfoo](https://www.promptfoo.dev/)
- [BigCode Evaluation Harness ](https://github.com/bigcode-project/bigcode-evaluation-harness)
- [LangFuse LLM Engineering platform with observability and evaluation tools ](https://langfuse.com/)
- [LLMeBench]( https://github.com/qcri/LLMeBench/) see [LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking](https://arxiv.org/abs/2308.04945)
- [ChainForge](https://chainforge.ai/)
- [Ironclad Rivet](https://rivet.ironcladapp.com/)
- LM-PUB-QUIZ: A Comprehensive Framework for Zero-Shot Evaluation of Relational Knowledge in Language Models, [arxiv pdf](https://arxiv.org/abs/2408.15729) [github repository](https://lm-pub-quiz.github.io/)

﻿---
### LLM Evaluation articles in tech media and blog posts from companies
- A Framework for Building Micro Metrics for LLM System Evaluation, Jan 2025, [InfoQ](https://www.infoq.com/articles/micro-metrics-llm-evaluation/)
- Evaluate LLMs using Evaluation Harness and Hugging Face TGI/vLLM, Sep 2024, [blog](https://www.philschmid.de/evaluate-llms-with-lm-eval-and-tgi-vllm)
- The LLM Evaluation guidebook ⚖️ from HuggingFace, Oct 2024,  [Hugging Face Evaluation guidebook](https://github.com/huggingface/evaluation-guidebook)
- Let's talk about LLM Evaluation, HuggingFace, [article](https://huggingface.co/blog/clefourrier/llm-evaluation)
- Using LLMs for Evaluation LLM-as-a-Judge and other scalable additions to human quality ratings. Aug 2024, [Deep Learning Focus](https://cameronrwolfe.substack.com/p/llm-as-a-judge)
- Examining the robustness of LLM evaluation to the distributional assumptions of benchmarks, Microsoft, Aug 2024, [ACL 2024](https://aclanthology.org/2024.acl-long.560.pdf)
- Introducing SimpleQA, OpenAI, Oct 2024 [OpenAI](https://openai.com/index/introducing-simpleqa/)
- Catch me if you can! How to beat GPT-4 with a 13B model, [LM sys org](https://lmsys.org/blog/2023-11-14-llm-decontaminator/)
-  [Why it’s impossible to review AIs, and why TechCrunch is doing it anyway Techcrun mat 2024](https://techcrunch-com.cdn.ampproject.org/c/s/techcrunch.com/2024/03/23/why-its-impossible-to-review-ais-and-why-techcrunch-is-doing-it-anyway/amp/)
- [A.I. has a measurement problem, NY Times, Apr 2024](https://www.nytimes.com/2024/04/15/technology/ai-models-measurement.html)
- [Beyond Accuracy: The Changing Landscape Of AI Evaluation, Forbes, Mar 2024](https://www.forbes.com/sites/sylvainduranton/2024/03/14/beyond-accuracy-the-changing-landscape-of-ai-evaluation/?sh=34576ff61e3d)
- [Mozilla AI Exploring LLM Evaluation at scale](https://blog.mozilla.ai/exploring-llm-evaluation-at-scale-with-the-neurips-large-language-model-efficiency-challenge/)
- Evaluation part of [How to Maximize LLM Performance](https://humanloop.com/blog/optimizing-llms)
- Mozilla AI blog published multiple good articles in [Mozilla AI blog](https://blog.mozilla.ai/)
- Andrej Karpathy on evaluation [X](https://twitter.com/karpathy/status/1795873666481402010)
- From Meta on evaluation of Llama 3 models [github](https://github.com/meta-llama/llama3/blob/main/eval_details.md)
- DeepMind AI Safety evaluation June 24 [deepmind blog, Introducing Frontier Safety Framework](https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/)
- AI Snake Oil, June 2024, [AI leaderboards are no longer useful. It's time to switch to Pareto curves.](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful)
- Hamel Dev March 2024, [Your AI Product Needs Eval. How to construct domain-specific LLM evaluation systems](https://hamel.dev/blog/posts/evals/)
  
---
### Large benchmarks
- MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs, Sep 2024, Audesk AI, [arxiv](https://arxiv.org/abs/2409.02257)
- MMLU Pro Massive Multitask Language Understanding - Pro version, Jun 2024, [arxiv](https://arxiv.org/abs/2406.01574)
- Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks EMNLP 2022, [pdf](https://aclanthology.org/2022.emnlp-main.340.pdf)
- Measuring Massive Multitask Language Understanding,  MMLU, ICLR, 2021, [arxiv](https://arxiv.org/pdf/2009.03300.pdf) [MMLU dataset](https://github.com/hendrycks/test)
- BigBench: Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models, 2022, [arxiv](https://arxiv.org/abs/2206.04615),  [datasets](https://github.com/google/BIG-bench)
- Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them, Oct 2022, [arxiv](https://arxiv.org/abs/2210.09261)
  
---
### Evaluation of evaluation, Evaluation theory, evaluation methods, analysis of evaluation
- The LLM Evaluation guidebook ⚖️ from HuggingFace, Oct 2024,  [Hugging Face Evaluation guidebook](https://github.com/huggingface/evaluation-guidebook)
- Evaluating the Evaluations: A Perspective on Benchmarks, Opinion paper, Amazon, Jan 2025, [SIGIR](https://www.sigir.org/wp-content/uploads/2025/01/p18.pdf)
- Inherent Trade-Offs between Diversity and Stability in Multi-Task Benchmarks, Max Planck Institute for Intelligent Systems, Tübingen, May 2024, ICML 2024, [arxiv](https://arxiv.org/abs/2405.01719)
- A Systematic Survey and Critical Review on Evaluating Large Language Models: Challenges, Limitations, and Recommendations, EMNLP 2024, [ACLAnthology](https://aclanthology.org/2024.emnlp-main.764/)
- Re-evaluating Automatic LLM System Ranking for Alignment with Human Preference, Dec 2024, [arxiv](https://arxiv.org/abs/2501.00560)
- Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations, Nov 2024, Anthropic, [arxiv](https://arxiv.org/abs/2411.00640)
- Lessons from the Trenches on Reproducible Evaluation of Language Models, May 2024, [arxiv](https://arxiv.org/abs/2405.14782)
- Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat, Nov 2024, [arxiv](https://arxiv.org/abs/2411.14483)
- Towards Evaluation Guidelines for Empirical Studies involving LLMs, Nov 2024, [arxiv](https://arxiv.org/abs/2411.07668)
- Sabotage Evaluations for Frontier Models, Anthropic, Nov 2024, [paper](https://assets.anthropic.com/m/377027d5b36ac1eb/original/Sabotage-Evaluations-for-Frontier-Models.pdf)  [blog post](https://www.anthropic.com/research/sabotage-evaluations)
- AI Benchmarks and Datasets for LLM Evaluation, Dec 2024, [arxiv](https://arxiv.org/abs/2412.01020), a survey of many LLM benchmarks
- Lessons from the Trenches on Reproducible Evaluation of Language Models, May 2024, [arxiv](https://arxiv.org/abs/2405.14782)
- Examining the robustness of LLM evaluation to the distributional assumptions of benchmarks, Aug 2024, [ACL 2024](https://aclanthology.org/2024.acl-long.560/)
- Synthetic data in evaluation*, see Chapter 3 in Best Practices and Lessons Learned on Synthetic Data for Language Models, Apr 2024, [arxiv](https://arxiv.org/abs/2404.07503)
- From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline, UC Berkeley, Jun 2024, [arxiv](https://arxiv.org/abs/2406.11939) [github repo](https://github.com/lmarena/arena-hard-auto)
- When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards, National Center for AI (NCAI), Feb 2024, [arxiv](https://arxiv.org/abs/2402.01781)
- Lifelong Benchmarks: Efficient Model Evaluation in an Era of Rapid Progress, Feb 2024, [arxiv](https://arxiv.org/abs/2402.19472)
- Are We on the Right Way for Evaluating Large Vision-Language Models?, Apr 2024, [arxiv](https://arxiv.org/pdf/2403.20330.pdf)
- What Are We Measuring When We Evaluate Large Vision-Language Models? An Analysis of Latent Factors and Biases, Apr 2024, [arxiv](https://arxiv.org/abs/2404.02415)
- Detecting Pretraining Data from Large Language Models, Oct 2023, [arxiv](https://arxiv.org/abs/2310.16789)
- Revisiting Text-to-Image Evaluation with Gecko: On Metrics, Prompts, and Human Ratings, Apr 2024, [arxiv](https://arxiv.org/abs/2404.16820)
- Faithful model evaluation for model-based metrics, EMNLP 2023, [amazon science](https://www.amazon.science/publications/faithful-model-evaluation-for-model-based-metrics)
- AI Snake Oil, June 2024, [AI leaderboards are no longer useful. It's time to switch to Pareto curves.](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful)
- State of What Art? A Call for Multi-Prompt LLM Evaluation , Aug 2024, [Transactions of the Association for Computational Linguistics (2024) 12](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00681/123885)
- Data Contamination Through the Lens of Time, Abacus AI etc, Oct 2023, [arxiv](https://arxiv.org/abs/2310.10628)
- Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models, ICML 2023, [mlr press](https://proceedings.mlr.press/v202/liu23ao.html)
- Are Emergent Abilities of Large Language Models a Mirage? Apr 23 [arxiv](https://arxiv.org/abs/2304.15004)
- Don't Make Your LLM an Evaluation Benchmark Cheater nov 2023 [arxiv](https://arxiv.org/abs/2311.01964)
- Holistic Evaluation of Text-to-Image Models, Stanford etc NeurIPS 2023, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/dd83eada2c3c74db3c7fe1c087513756-Paper-Datasets_and_Benchmarks.pdf)
- Model Spider: Learning to Rank Pre-Trained Models Efficiently, Nanjing University etc [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2c71b14637802ed08eaa3cf50342b2b9-Abstract-Conference.html)
- Evaluating Open-QA Evaluation, 2023, [arxiv](https://arxiv.org/abs/2305.12421)
- (RE: stat methods ) Prediction-Powered Inference Jan 23 [arxiv](https://arxiv.org/abs/2301.09633)  PPI++: Efficient Prediction-Powered Inference nov 23, [arxiv](https://arxiv.org/abs/2311.01453)
- Elo Uncovered: Robustness and Best Practices in Language Model Evaluation, Nov 2023 [arxiv](https://arxiv.org/abs/2311.17295)
- A Theory of Dynamic Benchmarks, ICLR 2023, University of California, Berkeley, [arxiv](https://arxiv.org/abs/2210.03165)
- Holistic Evaluation of Language Models, Center for Research on Foundation Models (CRFM), Stanford, Oct 2022, [arxiv](https://arxiv.org/abs/2211.09110)
- What Will it Take to Fix Benchmarking in Natural Language Understanding?, NY University , Google Brain, Oct 2022, [arxiv](https://arxiv.org/abs/2104.02145)
- Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models, Jun 2022, [arxiv](https://arxiv.org/abs/2206.04615)
- Evaluating Question Answering Evaluation, 2019, [ACL](https://aclanthology.org/D19-5817/)

---
### Long Comprehensive Studies
- Evaluation of OpenAI o1: Opportunities and Challenges of AGI, University of Alberta etc, Sep 2024, [arxiv](https://arxiv.org/abs/2409.18486)
- TrustLLM: Trustworthiness in Large Language Models, Jan 2024, [arxiv](https://arxiv.org/abs/2401.05561)
- Evaluating AI systems under uncertain ground truth: a case study in dermatology, Jul 2023, Google DeepMind etc, [arxiv](https://arxiv.org/abs/2307.02191)
  
---
### HITL (Human in the Loop)
- Developing a Framework for Auditing Large Language Models Using Human-in-the-Loop, Univ of Washington, Stanford, Amazon AI etc, Feb 2024, [arxiv](https://arxiv.org/abs/2402.09346)
- Which Prompts Make The Difference? Data Prioritization For Efficient Human LLM Evaluation, Cohere, Nov 2023, [arxiv](https://arxiv.org/abs/2310.14424)
- Evaluating Question Answering Evaluation, 2019, [ACL](https://aclanthology.org/D19-5817/)

---
### LLM as Judge
- LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods, Tsinghua University, Dec 2024, [arxiv](https://arxiv.org/abs/2412.05579)
- Are LLM-Judges Robust to Expressions of Uncertainty? Investigating the effect of Epistemic Markers on LLM-based Evaluation, Seoul National University , Naver etc Oct 2024, [arxiv](https://www.arxiv.org/pdf/2410.20774)
- JudgeBench: A Benchmark for Evaluating LLM-based Judges, UC Berkeley, Oct 2024, [arxiv](https://arxiv.org/abs/2410.12784)
- From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline, Oct 2024, UC Berkeley, [arxiv](https://arxiv.org/abs/2406.11939)
- Using LLMs for Evaluation LLM-as-a-Judge and other scalable additions to human quality ratings. Aug 2024, [Deep Learning Focus](https://cameronrwolfe.substack.com/p/llm-as-a-judge)
- Language Model Council: Democratically Benchmarking Foundation Models on Highly Subjective Tasks, Jun 2024, [arxiv](https://arxiv.org/abs/2406.08598)
- Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges, University of Massachusetts Amherst, Meta, Jun 2024, [arxiv](https://arxiv.org/abs/2406.12624)
- Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators, Stanford University, Apr 2024, [arxiv](https://arxiv.org/abs/2404.04475) [leaderboard](https://tatsu-lab.github.io/alpaca_eval/) [code](https://github.com/tatsu-lab/alpaca_eval)
- Large Language Models are Inconsistent and Biased Evaluators, Grammarly Duke Nvidia, May 2024, [arxiv](https://arxiv.org/abs/2405.01724)
- Report Cards: Qualitative Evaluation of Language Models Using Natural Language Summaries, University of Toronto and Vector Institute, Sep 2024, [arxiv](https://arxiv.org/abs/2409.00844)
- Evaluating LLMs at Detecting Errors in LLM Responses, Penn State University, Allen AI etc, Apr 2024, [arxiv](https://arxiv.org/abs/2404.03602)
- Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models, Cohere, Apr 2024, [arxiv](https://arxiv.org/abs/2404.18796)
- Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators, Mar 2024, [arxiv](https://arxiv.org/abs/2403.16950)
- LLM Evaluators Recognize and Favor Their Own Generations, Apr 2024, [pdf](https://drive.google.com/file/d/19H7-BNqccOw_IN3h-0WEz_zzc5ak3nyW/view)
- Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12272)
- The Generative AI Paradox on Evaluation: What It Can Solve, It May Not Evaluate, Feb 2024, [arxiv](https://arxiv.org/abs/2402.06204)
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena Jun 2023, [arxiv](https://arxiv.org/abs/2306.05685)
- Discovering Language Model Behaviors with Model-Written Evaluations, Dec 2022, [arxiv](https://arxiv.org/abs/2212.09251)
- Benchmarking Foundation Models with Language-Model-as-an-Examiner, 2022, [NEURIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html)
- Red Teaming Language Models with Language Models, Feb 2022, [arxiv](https://arxiv.org/abs/2202.03286)
- ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate, Aug 2023, [arxiv](https://arxiv.org/abs/2308.07201)
- ALLURE: Auditing and Improving LLM-based Evaluation of Text using Iterative In-Context-Learning, Sep 2023, [arxiv](https://arxiv.org/abs/2309.13701)
- Style Over Substance: Evaluation Biases for Large Language Models, Jul 2023, [arxiv](https://arxiv.org/abs/2307.03025)
- Large Language Models Are State-of-the-Art Evaluators of Translation Quality, Feb 2023, [arxiv](https://arxiv.org/abs/2302.14520)
- Large Language Models Are State-of-the-Art Evaluators of Code Generation, Apr 2023, [researchgate](https://www.researchgate.net/publication/370338371_Large_Language_Models_Are_State-of-the-Art_Evaluators_of_Code_Generation)
  
---
## LLM Evaluation
### Embeddings
- MTEB: Massive Text Embedding Benchmark Oct 2022 [arxiv](https://arxiv.org/abs/2210.07316 Leaderboard) [Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Marqo embedding benchmark for eCommerce [at Huggingface](https://huggingface.co/spaces/Marqo/Ecommerce-Embedding-Benchmarks), text to image and category to image tasks 
- The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding, [openreview pdf](https://openreview.net/pdf/f5f1953a9c798ec61bb050e62bc7a94037fd4fab.pdf)
- MMTEB: Community driven extension to MTEB [repository](https://github.com/embeddings-benchmark/mteb/blob/main/docs/mmteb/readme.md)
- Chinese MTEB C-MTEB [repository](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
- French MTEB [repository](https://github.com/Lyon-NLP/mteb-french)
  
---
### In Context Learning
- HellaSwag,  HellaSwag: Can a Machine Really Finish Your Sentence? 2019, [arxiv](https://arxiv.org/abs/1905.07830) Paper + code + dataset https://rowanzellers.com/hellaswag/ 
- The LAMBADA dataset: Word prediction requiring a broad discourse context 2016, [arxiv](https://arxiv.org/abs/1606.06031)
   
---
### Hallucinations
- The FACTS Grounding Leaderboard: Benchmarking LLMs' Ability to Ground Responses to Long-Form Input, DeepMind, Jan 2025, [arxiv](https://arxiv.org/abs/2501.03200) [Leaderboard](https://www.kaggle.com/facts-leaderboard)
- Introducing SimpleQA, OpenAI, Oct 2024 [OpenAI](https://openai.com/index/introducing-simpleqa/)
- A Survey of Hallucination in Large Visual Language Models, Oct 2024, See Chapter IV, Evaluation of Hallucinations [arxiv](https://arxiv.org/pdf/2410.15359#page=9.46)
- Long-form factuality in large language models, Google DeepMind etc, Mar 2024, [arxiv](https://arxiv.org/abs/2403.18802)
- TRUSTLLM: TRUSTWORTHINESS IN LARGE LANGUAGE MODELS: A PRINCIPLE AND BENCHMARK, Lehigh University, University of Notre Dame, MS Research,  etc,  Jan 2024, [arxiv](https://arxiv.org/abs/2401.05561), 
- INVITE: A testbed of automatically generated invalid questions to evaluate large language models for hallucinations, Amazon Science, EMNLP 2023, [amazon science](https://www.amazon.science/publications/invite-a-testbed-of-automatically-generated-invalid-questions-to-evaluate-large-language-models-for-hallucinations)
- Generating Benchmarks for Factuality Evaluation of Language Models, Jul 2023, [arxiv](https://arxiv.org/abs/2307.06908)
- AlignScore: Evaluating Factual Consistency with a Unified Alignment Function, May 2023, [arxiv](https://arxiv.org/abs/2305.16739)
- ChatGPT as a Factual Inconsistency Evaluator for Text Summarization, Mar 2023, [arxiv](https://arxiv.org/abs/2303.15621)
- HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models, Dec 2023,  [ACL](https://aclanthology.org/2023.emnlp-main.397.pdf) 
- Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models, Tencent AI lab etc, Sep 2023, [arxiv](https://arxiv.org/abs/2309.01219)
- Measuring Faithfulness in Chain-of-Thought Reasoning,  Anthropic etc,  Jul 2023, [[arxiv](https://arxiv.org/abs/2307.13702)
- FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation, University of Washington etc,  May 2023, [arxiv](https://arxiv.org/abs/2305.14251) [repository](https://github.com/shmsw25/FActScore)
- TRUE: Re-evaluating Factual Consistency Evaluation, Apt 2022, [arxiv](https://arxiv.org/abs/2204.04991)
  
---
### Question answering
QA is used in many vertical domains, see Vertical section bellow
- CoReQA: Uncovering Potentials of Language Models in Code Repository Question Answering, Jan 2025, [arxiv](https://arxiv.org/abs/2501.03447)
- Unveiling the power of language models in chemical research question answering, Jan 2025, [Nature, communication chemistry](https://www.nature.com/articles/s42004-024-01394-x) ScholarChemQA Dataset
- Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses, Oct 2024, Salesforce, [arxiv](https://arxiv.org/abs/2410.22349) [Answer Engine (RAG) Evaluation Repository](https://github.com/SalesforceAIResearch/answer-engine-eval)
- Introducing SimpleQA, OpenAI, Oct 2024 [OpenAI](https://openai.com/index/introducing-simpleqa/)
- Are Large Language Models Consistent over Value-laden Questions?, Jul 2024, [arxiv](https://arxiv.org/abs/2407.02996)
- CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge, Jun 2019, [ACL](https://aclanthology.org/N19-1421/) 
- Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering, Sep 2018, [arxiv](https://arxiv.org/abs/1809.02789) [OpenBookQA dataset at AllenAI](https://allenai.org/data/open-book-qa)
- Jin, Di, et al. "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams., 2020, [arxiv](https://arxiv.org/abs/2009.13081) [MedQA](https://paperswithcode.com/dataset/medqa-usmle)
- Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge, 2018, [arxiv](https://arxiv.org/abs/1803.05457)  [ARC Easy dataset](https://leaderboard.allenai.org/arc_easy/submissions/get-started) [ARC dataset](https://allenai.org/data/arc)
- BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions, 2019, [arxiv](https://arxiv.org/abs/1905.10044) [BoolQ dataset](https://huggingface.co/datasets/google/boolq)
- HellaSwag,  HellaSwag: Can a Machine Really Finish Your Sentence? 2019, [arxiv](https://arxiv.org/abs/1905.07830) Paper + code + dataset https://rowanzellers.com/hellaswag/
- PIQA: Reasoning about Physical Commonsense in Natural Language, Nov 2019, [arxiv](https://arxiv.org/abs/1911.11641)
[PIQA dataset](https://github.com/ybisk/ybisk.github.io/tree/master/piqa)
- Crowdsourcing Multiple Choice Science Questions [arxiv](https://arxiv.org/abs/1707.06209) [SciQ dataset](https://allenai.org/data/sciq)
- WinoGrande: An Adversarial Winograd Schema Challenge at Scale, 2017, [arxiv](https://arxiv.org/abs/1907.10641) [Winogrande dataset](https://www.tensorflow.org/datasets/catalog/winogrande)
- TruthfulQA: Measuring How Models Mimic Human Falsehoods, Sep 2021, [arxiv](https://arxiv.org/abs/2109.07958)
- TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages, 2020, [arxiv](https://arxiv.org/abs/2003.05002)  [data](https://github.com/google-research-datasets/tydiqa)
- Natural Questions: A Benchmark for Question Answering Research, [Transactions ACL 2019](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question) 
  
---
### Multi Turn
- LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models Nov 2023, [arxiv](https://arxiv.org/abs/2311.18232)
- MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues Feb 24 [arxiv](https://arxiv.org/abs/2402.14762)
- How Well Can LLMs Negotiate? NEGOTIATIONARENA Platform and Analysis Feb 2024 [arxiv](https://arxiv.org/abs/2402.05863)
- Parrot: Enhancing Multi-Turn Instruction Following for Large Language Models, Oct 2023, [arxiv](https://arxiv.org/abs/2310.07301)
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS 2023, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
- MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback, Sep 2023, [arxiv](https://arxiv.org/abs/2309.10691)

---
### Reasoning
- See 5.3 Evaluations chapter of DeepSeek R3 tech report on how new frontier models are evaluated Dec 2024 [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v1) and 3.1. DeepSeek-R1 Evaluation Chapter of DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning Jan 2025 [arxiv](https://arxiv.org/abs/2501.12948)
- Evaluating Generalization Capability of Language Models across Abductive, Deductive and Inductive Logical Reasoning, Jan 2025, [Proceedings of the 31st International Conference on Computational Linguistics](https://aclanthology.org/2025.coling-main.330/))
- FrontierMath at EpochAI, [FrontierAI page](https://epoch.ai/frontiermath), FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI, Nov 2024,  [arxiv](https://arxiv.org/abs/2411.04872)
- Easy Problems That LLMs Get Wrong, May 2024, [arxiv](https://arxiv.org/abs/2405.19616v2), a comprehensive Linguistic Benchmark designed to evaluate the limitations of Large Language Models (LLMs) in domains such as logical reasoning, spatial intelligence, and linguistic understanding
- Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning, NeurIPS 2024 Track Datasets and Benchmarks Spotlight, Sep 2024, [OpenReview](https://openreview.net/forum?id=aXeiCbMFFJ)
- Comparing Humans, GPT-4, and GPT-4V On Abstraction and Reasoning Tasks 2023, [arxiv](https://arxiv.org/abs/2311.09247)
- LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models, [arxiv](https://arxiv.org/abs/2404.05221)
- Evaluating LLMs' Mathematical Reasoning in Financial Document Question Answering, Feb 24, [arxiv](https://arxiv.org/abs/2402.11194v2) 
- Competition-Level Problems are Effective LLM Evaluators, Dec 23, [arxiv](https://arxiv.org/abs/2312.02143)
- Eyes Can Deceive: Benchmarking Counterfactual Reasoning Capabilities of Multimodal Large Language Models, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12966)
- MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning, Oct 2023, [arxiv](https://arxiv.org/abs/2310.16049)
  
---
### Multi-Lingual
- Mexa: Multilingual Evaluation of English-Centric LLMs via Cross-Lingual Alignment, (ICLR 2025 submission) [open review](https://openreview.net/forum?id=hsMkpzr9Oy)
- Multilingual Large Language Models: A Systematic Survey, Nov 2024, see Evaluation chapter about details of evaluation of multi-lingual large language models
 [Evaluation chapter, arxiv](https://arxiv.org/pdf/2411.11072#page=32.71)
- Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models, Taobao & Tmall Group of Alibaba, Nov 2024, [arxiv](https://arxiv.org/pdf/2411.07140)
- Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs, Oct 2024, [arxiv](https://arxiv.org/abs/2410.13394)
- Towards Multilingual LLM Evaluation for European Languages, TU Dresden etc, Oct 2024, [arxiv](https://arxiv.org/abs/2410.08928)
- MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models, Oct 2024, [arxiv](https://arxiv.org/abs/2410.17578)
- LLMzSzŁ: a comprehensive LLM benchmark for Polish, Jan 2024, [arxiv](https://arxiv.org/abs/2501.02266)
- AlGhafa Evaluation Benchmark for Arabic Language Models Dec 23, ACL Anthology [ACL pdf](https://aclanthology.org/2023.arabicnlp-1.21.pdf) [article](https://aclanthology.org/2023.arabicnlp-1.21/)
- CALAMITA: Challenge the Abilities of LAnguage Models in ITAlian, Dec 2024, [Tenth Italian Conference on Computational Linguistics,](https://ceur-ws.org/Vol-3878/116_calamita_preface_long.pdf)
- Evaluating and Advancing Multimodal Large Language Models in Ability Lens, Nov 2024, [arxiv](https://arxiv.org/abs/2411.14725)
- Introducing the Open Ko-LLM Leaderboard: Leading the Korean LLM Evaluation Ecosystem [HF blog](https://huggingface.co/blog/leaderboard-upstage)
- Heron-Bench: A Benchmark for Evaluating Vision Language Models in Japanese , Apr 2024 [arxiv](https://arxiv.org/abs/2404.07824)
- BanglaQuAD: A Bengali Open-domain Question Answering Dataset, Oct 2024, [arxiv](https://arxiv.org/abs/2410.10229)
- MultiPragEval: Multilingual Pragmatic Evaluation of Large Language Models, Jun 2024, [arxiv](https://arxiv.org/abs/2406.07736)
- Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?, Mar 2024, [Findings of the Association for Computational Linguistics: EACL 2024](https://aclanthology.org/2024.findings-eacl.71/)
- The Invalsi Benchmark: measuring Language Models Mathematical and Language understanding in Italian, Mar 2024, [arxiv](https://arxiv.org/pdf/2403.18697.pdf)
- MEGA: Multilingual Evaluation of Generative AI, Mar 2023, [arxiv](https://arxiv.org/abs/2303.12528)
- Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian Language?, Apr 2024, [arxiv](https://arxiv.org/abs/2404.06644)
- Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model, Cohere, Feb 2024, [arxiv](https://arxiv.org/abs/2402.07827) see [Evaluation chapter](https://arxiv.org/pdf/2402.07827#page=13.36) with details how to evaluate multi lingual model capabilities
- M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models, 2023, [NIPS website](https://proceedings.neurips.cc/paper_files/paper/2023/hash/117c5c8622b0d539f74f6d1fb082a2e9-Abstract-Datasets_and_Benchmarks.html)
- LAraBench: Benchmarking Arabic AI with Large Language Models, May 23, [arxiv](https://arxiv.org/abs/2305.14982)
- AlignBench: Benchmarking Chinese Alignment of Large Language Models, Nov 2023, [arxiv](https://arxiv.org/abs/2311.18743)
- CLUE: A Chinese Language Understanding Evaluation Benchmark, Apr 2020, [arxiv](https://arxiv.org/abs/2407.16931) [CLUEWSC(Winograd Scheme Challenge)](https://github.com/CLUEbenchmark/CLUEWSC2020)

---
#### Multi-Lingual Embedding tasks
- The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding, [openreview pdf](https://openreview.net/pdf/f5f1953a9c798ec61bb050e62bc7a94037fd4fab.pdf)
- Chinese MTEB C-MTEB [repository](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
- French MTEB [repository](https://github.com/Lyon-NLP/mteb-french)
- C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models, May 2023, [arxiv](https://arxiv.org/abs/2305.08322)
  
---
### Multi-Modal
- Benchmark Evaluations, Applications, and Challenges of Large Vision Language Models: A Survey, Jan 2025, [arxiv](https://www.arxiv.org/abs/2501.02189)
- LVLM-EHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models, Nov 2024, [IEEE](https://ieeexplore.ieee.org/abstract/document/10769058)
- ScImage: How Good Are Multimodal Large Language Models at Scientific Text-to-Image Generation?, Dec 2024, [arxiv](https://arxiv.org/abs/2412.02368)
- RealWorldQA, Apr 2024, [HuggingFace](https://huggingface.co/blog/KennyUTC/realworldqa)
- Image2Struct: Benchmarking Structure Extraction for Vision-Language Models, Oct 2024, [arxiv](https://arxiv.org/abs/2410.22456)
- MMBench: Is Your Multi-modal Model an All-Around Player?, Oct 2024 [springer ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72658-3_13)
- MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models, Oct 2024, [arxiv](https://arxiv.org/abs/2410.10139)
- MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI, Apr 2024, [arxiv](https://arxiv.org/abs/2404.16006)
- MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI, CVPR 2024, [CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.html)
- ConvBench: A Multi-Turn Conversation Evaluation Benchmark with Hierarchical Ablation Capability for Large Vision-Language Models, Dec 2024, [open review](https://openreview.net/pdf?id=PyTf2jj0SH) [github for the benchmark and evaluation framework](https://github.com/shirlyliu64/ConvBench)
- Careless Whisper: Speech-to-Text Hallucination Harms, FAccT '24, [ACM](https://dl.acm.org/doi/abs/10.1145/3630106.3658996)
- AutoBench-V: Can Large Vision-Language Models Benchmark Themselves?, Oct 2024 [arxiv](https://arxiv.org/abs/2410.21259)
- HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning, Oct 2024,  [Computer Vision – ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72980-5_17) 
- VHELM: A Holistic Evaluation of Vision Language Models, Oct 2024, [arxiv](https://arxiv.org/abs/2410.07112)
- Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models, Reka AI, May 2024 [arxiv](https://arxiv.org/abs/2405.02287) [dataset](https://github.com/reka-ai/reka-vibe-eval)  [blog post](https://www.reka.ai/news/vibe-eval)
- Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis, Aug 2024, [arxiv](https://arxiv.org/abs/2409.00106)
- CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models, Jun 2024, [arxiv](https://arxiv.org/abs/2406.06007)
- EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models, Jun 2024, [arxiv](https://arxiv.org/abs/2406.05756)
- MFC-Bench: Benchmarking Multimodal Fact-Checking with Large Vision-Language Models, Jun 2024, [arxiv](https://arxiv.org/abs/2406.11288)
- Holistic Evaluation of Text-to-Image Models Nov 23 [arxiv](https://arxiv.org/abs/2311.04287)
- VBench: Comprehensive Benchmark Suite for Video Generative Models Nov 23 [arxiv](https://arxiv.org/abs/2311.04287)
- Evaluating Text-to-Visual Generation with Image-to-Text Generation, Apr 2024, [arxiv](https://arxiv.org/abs/2404.01291)
- What Are We Measuring When We Evaluate Large Vision-Language Models? An Analysis of Latent Factors and Biases, Apr 2024, [arxiv](https://arxiv.org/abs/2404.02415)
- Are We on the Right Way for Evaluating Large Vision-Language Models?, Apr 2024, [arxiv](https://arxiv.org/pdf/2403.20330.pdf)
- MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning, Nov 2023, [arxiv](https://arxiv.org/abs/2311.10774)
- BLINK: Multimodal Large Language Models Can See but Not Perceive, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12390) [github](https://zeyofu.github.io/blink/)
- Eyes Can Deceive: Benchmarking Counterfactual Reasoning Capabilities of Multimodal Large Language Models, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12966)
- Revisiting Text-to-Image Evaluation with Gecko: On Metrics, Prompts, and Human Ratings, Apr 2024, [arxiv](https://arxiv.org/abs/2404.16820)
- VALOR-EVAL: Holistic Coverage and Faithfulness Evaluation of Large Vision-Language Models, Apr 2024, [arxiv](https://arxiv.org/abs/2404.13874v1)
- MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts, Oct 2023, [arxiv](MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts)
- Evaluation part of https://arxiv.org/abs/2404.18930, Apr 2024, [arxiv](https://arxiv.org/abs/2404.18930), [repository](https://github.com/showlab/Awesome-MLLM-Hallucination)
- VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use, Aug 2023. [arxiv](https://arxiv.org/abs/2308.06595)
- MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities, Aug 2023, [arxiv](https://arxiv.org/abs/2308.02490)
- SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension, Jul 2023, [arxiv](https://arxiv.org/abs/2307.16125)
- LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark, NeurIPS 2023, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/548a41b9cac6f50dccf7e63e9e1b1b9b-Abstract-Datasets_and_Benchmarks.html)
- Holistic Evaluation of Text-to-Image Models, Stanford etc NeurIPS 2023, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/dd83eada2c3c74db3c7fe1c087513756-Paper-Datasets_and_Benchmarks.pdf) 
- mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality, Apr 2023 [arxiv](https://arxiv.org/abs/2304.14178)
  
---
### Instruction Following
- Evaluating Large Language Models at Evaluating Instruction Following Oct 2023, [arxiv](https://arxiv.org/abs/2310.07641)
- Find the INTENTION OF INSTRUCTION: Comprehensive Evaluation of Instruction Understanding for Large Language Models, Dec 2024, [arxiv](https://arxiv.org/abs/2412.19450)
- HREF: Human Response-Guided Evaluation of Instruction Following in Language Models, Dec 2024, [arxiv](https://arxiv.org/abs/2412.15524)
- CFBench: A Comprehensive Constraints-Following Benchmark for LLMs. Aug 2024, [arxiv](https://arxiv.org/abs/2408.01122)
- Instruction-Following Evaluation for Large Language Models, IFEval, Nov 2023, [arxiv](https://arxiv.org/abs/2311.07911)
- FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets, Jul 2023, [arxiv](https://arxiv.org/abs/2307.10928) , [FLASK dataset](https://github.com/kaistAI/FLASK)
- DINGO: Towards Diverse and Fine-Grained Instruction-Following Evaluation, Mar 2024, [aaai](https://ojs.aaai.org/index.php/AAAI/article/view/29768), [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/29768/31322)
- LongForm: Effective Instruction Tuning with Reverse Instructions, Apr 2023, [arxiv](https://arxiv.org/abs/2304.08460) [dataset](https://github.com/akoksal/LongForm)
  
---
### Ethical AI
- Evaluating the Moral Beliefs Encoded in LLMs,  Jul 23 [arxiv](https://arxiv.org/abs/2307.14324)
- AI Deception: A Survey of Examples, Risks, and Potential Solutions Aug 23 [arxiv](https://arxiv.org/abs/2308.14752)
- Aligning AI With Shared Human Value, Aug 20 - Feb 23, [arxiv](https://arxiv.org/abs/2008.02275) Re: ETHICS benchmark
- What are human values, and how do we align AI to them?, Mar 2024, [pdf](https://static1.squarespace.com/static/65392ca578eee444c445c9de/t/6606f95edb20e8118074a344/1711733370985/human-values-and-alignment-29MAR2024.pdf)
- TrustLLM: Trustworthiness in Large Language Models, Jan 2024, [arxiv](https://arxiv.org/abs/2401.05561)
- Helpfulness, Honesty, Harmlessness (HHH) framework from Antrhtopic, introduced in A General Language Assistantas a Laboratory for Alignment, 2021, [arxiv](https://arxiv.org/pdf/2112.00861), it's in BigBench now [bigbench](https://github.com/google/BIG-bench)
- WorldValuesBench: A Large-Scale Benchmark Dataset for Multi-Cultural Value Awareness of Language Models, April 2024, [arxiv](https://arxiv.org/abs/2404.16308)
- Chapter 19 in The Ethics of Advanced AI Assistants, Apr 2024, Google DeepMind, [pdf at google](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/ethics-of-advanced-ai-assistants/the-ethics-of-advanced-ai-assistants-2024-i.pdf)
- BEHONEST: Benchmarking Honesty of Large Language Models, June 2024, [arxiv](https://arxiv.org/abs/2406.13261)
  
---
### Biases
- FairPair: A Robust Evaluation of Biases in Language Models through Paired Perturbations, Apr 2024 [arxiv](https://arxiv.org/abs/2404.06619v1)
- BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation, 2021, [arxiv](https://arxiv.org/abs/2101.11718), [dataset](https://github.com/amazon-science/bold)
- “I’m fully who I am”: Towards centering transgender and non-binary voices to measure biases in open language generation, ACM FAcct 2023, [amazon science](https://www.amazon.science/publications/im-fully-who-i-am-towards-centering-transgender-and-non-binary-voices-to-measure-biases-in-open-language-generation)
- This Land is {Your, My} Land: Evaluating Geopolitical Biases in Language Models, May 2023, [arxiv](https://arxiv.org/abs/2305.14610)
  
---
### Safe AI
- Lessons From Red Teaming 100 Generative AI Products, Microsoft, Jan 2025, [arxiv](https://arxiv.org/abs/2501.07238)
- Trading Inference-Time Compute for Adversarial Robustness, OpenAI, Jan 2025, [arxiv](https://openai.com/index/trading-inference-time-compute-for-adversarial-robustness/)
- Medical large language models are vulnerable to data-poisoning attacks, New York University, Jan 2025, [Nature Medicine](https://www.nature.com/articles/s41591-024-03445-1)
- Benchmark for general-purpose AI chat model, December 2024, AILuminate from ML Commons, [mlcommons website](https://ailuminate.mlcommons.org/benchmarks/)
- Fooling LLM graders into giving better grades through neural activity guided adversarial prompting, Stanford University, Dec 2024, [arxiv](https://www.arxiv.org/abs/2412.15275)
- ECCV 2024 MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models, Shanghai AI Laboratory, etc, Jan 2024, [github](https://github.com/isXinLiu/MM-SafetyBench) [arxiv nov 2023](https://arxiv.org/abs/2311.17600)
- Introducing v0.5 of the AI Safety Benchmark from MLCommons, ML Commons, Google Research etc Apr 2024, [arxiv](https://arxiv.org/abs/2404.12241)
- SecCodePLT: A Unified Platform for Evaluating the Security of Code GenAI, Virtue AI, etc, Oct 2024, [arxiv](https://arxiv.org/abs/2410.11096)
- Beyond Prompt Brittleness: Evaluating the Reliability and Consistency of Political Worldviews in LLMs , University of Stuttgart, etc, Nov 2024, [MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00710/125176)
- Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?, Center for AI Safety etc, Jul 2024. [arxiv](https://arxiv.org/abs/2407.21792)
- Risk Taxonomy, Mitigation, and Assessment Benchmarks of Large Language Model Systems, Zhongguancun Laboratory, etc,  Jan 2024, [arxiv](https://arxiv.org/abs/2401.05778)
- LLMSecCode: Evaluating Large Language Models for Secure Coding, Chalmers University of Technology,  Aug 2024, [arxiv](https://arxiv.org/abs/2408.16100)
- Attack Atlas: A Practitioner's Perspective on Challenges and Pitfalls in Red Teaming GenAI, IBM Research etc, Sep 2024, [arxiv](https://arxiv.org/abs/2409.15398)
- DetoxBench: Benchmarking Large Language Models for Multitask Fraud & Abuse Detection, Amazon.com, Sep 2024, [arxiv](https://arxiv.org/abs/2409.06072)
- Purple Llama, an umbrella project from Meta, Meta, [Purple Llama repository](https://github.com/meta-llama/PurpleLlama)
- How Many Are in This Image A Safety Evaluation Benchmark for Vision LLMs, ECCV 2024, [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72983-6_3)
- Explore, Establish, Exploit: Red Teaming Language Models from Scratch, MIT CSAIL etc, Jun 2023, [arxiv](https://arxiv.org/abs/2306.09442)
- Rethinking Backdoor Detection Evaluation for Language Models, University of Southern California, Aug 2024, [arxiv pdf](https://arxiv.org/abs/2409.00399)
- Gradient-Based Language Model Red Teaming, Google Research & Anthropic, Jan 24, [arxiv](https://arxiv.org/abs/2401.16656)
- JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models, University of Pennsylvania, ETH Zurich, etc, , Mar 2024, [arxiv](https://arxiv.org/abs/2404.01318)
- Announcing a Benchmark to Improve AI Safety MLCommons has made benchmarks for AI performance—now it's time to measure safety, Apr 2024 [IEEE Spectrum](https://spectrum.ieee.org/ai-safety-benchmark)
- Model evaluation for extreme risks, Google DeepMind, OpenAI, etc, May 2023, [arxiv](https://arxiv.org/abs/2305.15324)
- A StrongREJECT for Empty Jailbreaks, Center for Human-Compatible AI, UC Berkeley, Feb 2024, [arxiv](https://arxiv.org/abs/2402.10260) [StrongREJECT Benchmark](https://strong-reject.readthedocs.io/en/latest/)
- Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training, Anthropic, Redwood Research etc Jan 2024, [arxiv](https://arxiv.org/abs/2401.05566)
- How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs, Nov 2023, [arxiv](https://arxiv.org/abs/2311.16101)
- On Evaluating Adversarial Robustness of Large Vision-Language Models, NeurIPS 2023, [NeuriPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a97b58c4f7551053b0512f92244b0810-Abstract-Conference.html)
- How Robust is Google's Bard to Adversarial Image Attacks?, Sep 2023, [arxiv](https://arxiv.org/abs/2309.11751)
  
---
### Cybersecurity
- CYBERSECEVAL 3: Advancing the Evaluation of Cybersecurity Risks and Capabilities in Large Language Models, July 2023, [Meta](https://ai.meta.com/research/publications/cyberseceval-3-advancing-the-evaluation-of-cybersecurity-risks-and-capabilities-in-large-language-models/) [arxiv](https://arxiv.org/abs/2408.01605)
- CYBERSECEVAL 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models, Apr 2024, [Meta](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/) [arxiv](https://arxiv.org/abs/2404.13161)
- Benchmarking OpenAI o1 in Cyber Security, Oct 2024, [arxiv](https://arxiv.org/abs/2410.21939)
- Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models, Aug 2024, [arxiv](https://arxiv.org/abs/2408.08926)
  
---
### Code Generating LLMs
- SecCodePLT: A Unified Platform for Evaluating the Security of Code GenAI, Oct 2024, [arxiv](https://arxiv.org/abs/2410.11096)
- Aider Polyglot, code editing benchmark [Aider polyglot site](https://aider.chat/docs/benchmarks.html#the-benchmark)
- A Survey on Evaluating Large Language Models in Code Generation Tasks, Peking University etc, Aug 2024, [arxiv](https://arxiv.org/abs/2408.16498)
- LLMSecCode: Evaluating Large Language Models for Secure Coding, Aug 2024, [arxiv](https://arxiv.org/abs/2408.16100)
- Copilot Evaluation Harness: Evaluating LLM-Guided Software Programming Feb 24 [arxiv](https://arxiv.org/abs/2402.14261)
- LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code, Berkeley, MIT, Cornell, Mar 2024, [arxiv](https://arxiv.org/abs/2403.07974)
- Introducing SWE-bench Verified, OpenAI, Aug 2024, [OpenAI](https://openai.com/index/introducing-swe-bench-verified/)
- SWE Bench SWE-bench: Can Language Models Resolve Real-World GitHub Issues? Feb 2024 [arxiv](https://arxiv.org/abs/2402.05863) [Tech Report](https://www.cognition-labs.com/post/swe-bench-technical-report)
- Gorilla Functional Calling Leaderboard, Berkeley [Leaderboard]( https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- DevBench: A Comprehensive Benchmark for Software Development, Mar 2024,[arxiv](https://arxiv.org/abs/2403.08604)
- Evaluating Large Language Models Trained on Code HumanEval Jul 2022 [arxiv](https://arxiv.org/abs/2107.03374)
- CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation Feb 21 [arxiv](https://arxiv.org/abs/2102.04664)
- MBPP (Mostly Basic Python Programming) benchmark, introduced in Program Synthesis with Large Language Models
, 2021 [papers with code](https://paperswithcode.com/paper/program-synthesis-with-large-language-models) [data](https://huggingface.co/datasets/mbpp)
- CodeMind: A Framework to Challenge Large Language Models for Code Reasoning, Feb 2024, [arxiv](https://arxiv.org/abs/2402.09664)
- CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution, Jan 2024, [arxiv](https://arxiv.org/abs/2401.03065)
- CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning, Jul 2022, [arxiv](https://arxiv.org/abs/2207.01780) [code at salesforce github](https://github.com/salesforce/CodeRL)
  
---
### Summarization
- Evaluation & Hallucination Detection for Abstractive Summaries, [online blog article](https://eugeneyan.com/writing/abstractive/)
- A dataset and benchmark for hospital course summarization with adapted large language models, Dec 2024, [Journal of the American Medical Informatics Association](https://academic.oup.com/jamia/advance-article-abstract/doi/10.1093/jamia/ocae312/7934937?redirectedFrom=fulltext)
- Benchmarking Large Language Models for News Summarization , Jan 2024, [Transactions of ACL](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276)
- SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation, May 2023, [arxiv](https://arxiv.org/abs/2305.13194) [benchmark data](https://github.com/google-research-datasets/seahorse?tab=readme-ov-file)
- Human-like Summarization Evaluation with ChatGPT, Apr 2023, [arxiv](https://arxiv.org/abs/2304.02554)
- Evaluating the Factual Consistency of Large Language Models Through News Summarization, Nov 2022, [arxiv](https://arxiv.org/abs/2211.08412)
- USB: A Unified Summarization Benchmark Across Tasks and Domains, May 2023, [arxiv](https://arxiv.org/abs/2305.14296)
- QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization, Jan 2021, [arxiv](https://arxiv.org/abs/2112.08542)
- SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization, Nov 2021, [arxiv](https://arxiv.org/abs/2111.09525) [github data](https://github.com/tingofurro/summac)
- WikiAsp: A Dataset for Multi-domain Aspect-based Summarization, 2021, [Transactions ACL](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00362/98088/WikiAsp-A-Dataset-for-Multi-domain-Aspect-based) [dataset](https://huggingface.co/datasets/wiki_asp)
  
---
### LLM  quality (generic methods: overfitting, redundant layers etc)
- [WeightWatcher](https://calculatedcontent.com/2024/01/23/evaluating-fine-tuned-llms-with-weightwatcher/)
  
---
### Inference Performance
- Ray/Anyscale's LLM Performance [Leaderboard](https://github.com/ray-project/llmperf-leaderboard) ([explanation:](https://www.anyscale.com/blog/comparing-llm-performance-introducing-the-open-source-leaderboard-for-llm))
- MLCommons MLPerf benchmarks (inference) [MLPerf announcement of the LLM track](https://mlcommons.org/2023/09/mlperf-results-highlight-growing-importance-of-generative-ai-and-storage/)
  
---
### Agent LLM Architectures
- Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making, Oct 2024, [arxiv](https://arxiv.org/abs/2410.07166)
- LLM4RL: Enhancing Reinforcement Learning with Large Language Models, Aug 2024, [IEEE Explore](https://ieeexplore.ieee.org/document/10667224)
- Put Your Money Where Your Mouth Is: Evaluating Strategic Planning and Execution of LLM Agents in an Auction Arena, Oct 2023, [arxiv](https://arxiv.org/abs/2310.05746)
- Chapter 4 4 LLM-based autonomous agent evaluation in A survey on large language model based autonomous agents, Front. Comput. Sci., 2024, [Front. Comput. Sci., 2024, at Springer](https://link.springer.com/content/pdf/10.1007/s11704-024-40231-1.pdf#page=16.50)
- LLM-Deliberation: Evaluating LLMs with Interactive Multi-Agent Negotiation Games, Sep 2023,[arxiv](https://arxiv.org/abs/2309.17234)
- AgentBench: Evaluating LLMs as Agents, Aug 2023, [arxiv](https://arxiv.org/abs/2308.03688)
- How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments, Mar 2024, [arxiv](https://arxiv.org/abs/2403.11807)
- R-Judge: Benchmarking Safety Risk Awareness for LLM Agents, Jan 2024, [arxiv](https://arxiv.org/abs/2401.10019)
- ProAgent: Building Proactive Cooperative Agents with Large Language Models, Aug 2023, [arxiv](https://arxiv.org/abs/2308.11339)
- Towards A Unified Agent with Foundation Models, Jul 2023, [arxiv](https://arxiv.org/abs/2307.09668)
- RestGPT: Connecting Large Language Models with Real-World RESTful APIs, Jun 2023, [arxiv](https://arxiv.org/abs/2306.06624)
- Large Language Models Are Semi-Parametric Reinforcement Learning Agents, Jun 2023, [arxiv](https://arxiv.org/abs/2306.07929)

  
---
### Long Text Generation
- Suri: Multi-constraint Instruction Following for Long-form Text Generation, Jun 2024, [arxiv](https://arxiv.org/abs/2406.19371)
- LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs, Aug 2024, [arxiv](https://arxiv.org/abs/2408.07055)
- LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding, Aug 2023, [arxiv](https://arxiv.org/abs/2308.14508)
- HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models, Sep 2024, [arxiv](https://arxiv.org/abs/2409.16191)
  
---
### Graph understanding
-  GPT4Graph: Can Large Language Models Understand Graph Structured Data ? An Empirical Evaluation and Benchmarking, May 2023, [arxiv](https://arxiv.org/abs/2305.15066)
- LLM4DyG: Can Large Language Models Solve Spatial-Temporal Problems on Dynamic Graphs? Oct 2023, [arxiv](https://arxiv.org/abs/2310.17110)
- Talk like a Graph: En Graphs for Large Language Models, Oct 2023, [arxiv](https://arxiv.org/abs/2310.04560)
- Open Graph Benchmark: Datasets for Machine Learning on Graphs, [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html)
- Can Language Models Solve Graph Problems in Natural Language? [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/622afc4edf2824a1b6aaf5afe153fa93-Abstract-Conference.html)
- Evaluating Large Language Models on Graphs: Performance Insights and Comparative Analysis, Aug 2023, [https://arxiv.org/abs/2308.11224]
## Reward Models 
- RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style, Oct 2024, [arxiv](https://arxiv.org/abs/2410.16184)
- HelpSteer2: Open-source dataset for training top-performing reward models, Aug 2024, [arxiv](https://arxiv.org/abs/2406.08673)
- RewardBench: Evaluating Reward Models for Language Modeling, Mar 2024, [arxiv](https://arxiv.org/abs/2403.13787v1)
- MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues Feb 24 [arxiv](https://arxiv.org/abs/2402.14762)
---
### Various unclassified tasks
(TODO as there are more than three papers per class, make a class a separate chapter in this Compendium)
- From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline, UC Berkeley, Jun 2024, [arxiv](https://arxiv.org/abs/2406.11939) [github repo](https://github.com/lmarena/arena-hard-auto)   ArenaHard benchmark
- Fooling LLM graders into giving better grades through neural activity guided adversarial prompting, Dec 2024, [arxiv](https://www.arxiv.org/abs/2412.15275)
- LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation, Jan 2025, [arxiv](https://arxiv.org/abs/2501.05414)
- DesignQA: A Multimodal Benchmark for Evaluating Large Language Models’ Understanding of Engineering Documentation, Dec 2024, [The American Society of Mechanical Engineers](https://asmedigitalcollection.asme.org/computingengineering/article-abstract/25/2/021009/1210215/DesignQA-A-Multimodal-Benchmark-for-Evaluating?redirectedFrom=fulltext)
- OmniEvalKit: A Modular, Lightweight Toolbox for Evaluating Large Language Model and its Omni-Extensions, Dec 2024, [arxiv](https://arxiv.org/abs/2412.06693)
- Holmes ⌕ A Benchmark to Assess the Linguistic Competence of Language Models , Dec 2024, [MIT Press Transactions of ACL, 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00718/125534)
- BENCHAGENTS: Automated Benchmark Creation with Agent Interaction, Oct 2024, [arxiv](https://arxiv.org/abs/2410.22584)
- EscapeBench: Pushing Language Models to Think Outside the Box, Dec 2024, [arxiv](https://arxiv.org/abs/2412.13549)
- OLMES: A Standard for Language Model Evaluations, Jun 2024, [arxiv](https://arxiv.org/abs/2406.08446)
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training, Nov 2024, [arxiv](https://arxiv.org/abs/2411.15124) see 7.1 Open Language Model Evaluation System (OLMES) and AllenAI Githib rep for [Olmes](http://github.com/allenai/olmes)
- Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making, Oct 2024, [arxiv](https://arxiv.org/abs/2410.07166)
- Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks, Nov 2024, [arxiv](https://arxiv.org/abs/2411.05821)
- Evaluating Superhuman Models with Consistency Checks, Apr 2024, [IEEE](https://ieeexplore.ieee.org/abstract/document/10516635)
- To the Globe (TTG): Towards Language-Driven Guaranteed Travel Planning, Meta AI, Oct 2024, [arxiv](https://arxiv.org/abs/2410.16456) evaluation for tasks of travel planning
- Assessing the Performance of Human-Capable LLMs -- Are LLMs Coming for Your Job?, Oct 2024, [arxiv](https://arxiv.org/abs/2410.16285),  SelfScore, a novel benchmark designed to assess the performance of automated Large Language Model (LLM) agents on help desk and professional consultation task
- Should We Really Edit Language Models? On the Evaluation of Edited Language Models, Oct 2024, [arxiv](https://arxiv.org/abs/2410.18785)
- DyKnow: Dynamically Verifying Time-Sensitive Factual Knowledge in LLMs, EMNLP 2024, Oct 2024, [arxiv](https://arxiv.org/abs/2404.08700), [Repository for DyKnow](https://github.com/sislab-unitn/DyKnow)
- Jeopardy dataset at HuggingFace, [huggingface](https://huggingface.co/datasets/jeopardy-datasets/jeopardy)
- A framework for few-shot language model evaluation, Zenodo, Jul 2024, [Zenodo](https://zenodo.org/records/5371629)
- ORAN-Bench-13K: An Open Source Benchmark for Assessing LLMs in Open Radio Access Networks, Jul 2024 [arxiv](https://arxiv.org/abs/2407.06245)
- AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models, Aug 2023, [arxiv](https://arxiv.org/abs/2304.06364)
- Evaluation of Response Generation Models: Shouldn’t It Be Shareable and Replicable?, Dec 2022, [Proceedings of the 2nd Workshop on Natural Language Generation, Evaluation, and Metrics (GEM)](https://aclanthology.org/2022.gem-1.12/) [Github repository for Human Evaluation Protocol](https://github.com/sislab-unitn/Human-Evaluation-Protocol)
- From Babbling to Fluency: Evaluating the Evolution of Language Models in Terms of Human Language Acquisition, Oct 2024, [arxiv](https://arxiv.org/abs/2410.13259)
- DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph, June 2024, [arxiv](https://arxiv.org/abs/2406.17271)
- RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style, Oct 2024, [arxiv](https://arxiv.org/abs/2410.16184)
- Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study Mar 24, WSDM 24, [ms blog](https://www.microsoft.com/en-us/research/publication/table-meets-llm-can-large-language-models-understand-structured-table-data-a-benchmark-and-empirical-study/)
- How Much are Large Language Models Contaminated? A Comprehensive Survey and the LLMSanitize Library, Nanyang Technological University, Mar 2024, [arxiv](https://arxiv.org/abs/2404.00699)
-  LLM Comparative Assessment: Zero-shot NLG Evaluation through Pairwise Comparisons using Large Language Models, jul 2023 [arxiv](https://arxiv.org/abs/2307.07889v3)
- OpenEQA: From word models to world models, Meta, Apr 2024, Understanding physical spaces by Models,  [Meta AI blog](https://ai.meta.com/blog/openeqa-embodied-question-answering-robotics-ar-glasses/)
- Is Your LLM Outdated? Benchmarking LLMs & Alignment Algorithms for Time-Sensitive Knowledge. Apr 2024, [arxiv](https://arxiv.org/abs/2404.08700)
- ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models, Apr 2024, [arxiv](https://arxiv.org/pdf/2403.20262.pdf)
- LongEmbed: Extending Embedding Models for Long Context Retrieval, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12096), benchmark for long context tasks, [repository for LongEmbed](https://github.com/dwzhu-pku/LongEmbed)
- LoTa-Bench: Benchmarking Language-oriented Task Planners for Embodied Agents, Feb 2024, [arxiv](https://arxiv.org/abs/2402.08178)
- Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT, Feb 2024, [arxiv](https://arxiv.org/abs/2402.07440), LoCoV1 benchmark for long context LLM,
- A User-Centric Benchmark for Evaluating Large Language Models, Apr 2024, [arxiv](https://arxiv.org/abs/2404.13940), [data of user centric benchmark at github](https://github.com/Alice1998/URS)
- Evaluating Quantized Large Language Models, Tsinghua University etc, International Conference on Machine Learning, PMLR 2024, [arxiv](https://arxiv.org/abs/2402.18158)
- RACE: Large-scale ReAding Comprehension Dataset From Examinations, 2017, [arxiv](https://arxiv.org/abs/1704.04683) [RACE dataset at CMU](https://www.cs.cmu.edu/~glai1/data/race/)
- CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models, 2020, [arxiv](https://arxiv.org/abs/2010.00133) [CrowS-Pairs dataset](https://github.com/nyu-mll/crows-pairs)
- DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs, Jun 2019, [ACL](https://aclanthology.org/N19-1246/) [data](https://allenai.org/data/drop)
- RewardBench: Evaluating Reward Models for Language Modeling, Mar 2024, [arxiv](https://arxiv.org/abs/2403.13787v1)
- Toward informal language processing: Knowledge of slang in large language models, EMNLP 2023, [amazon science](https://www.amazon.science/publications/invite-a-testbed-of-automatically-generated-invalid-questions-to-evaluate-large-language-models-for-hallucinations)
- FOFO: A Benchmark to Evaluate LLMs' Format-Following Capability, Feb 2024, [arxiv](https://arxiv.org/abs/2402.18667)
- Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs, 05 2024,Bird, a big benchmark for large-scale database grounded in text-to-SQL tasks, containing 12,751 pairs of text-to-SQL data and 95 databases with a total size of 33.4 GB, spanning 37 professional domain [arxiv](https://arxiv.org/abs/2305.03111) [data and leaderboard](https://bird-bench.github.io/)
- MuSiQue: Multihop Questions via Single-hop Question Composition, Aug 2021, [arxiv](https://arxiv.org/abs/2108.00573)
- Evaluating Copyright Takedown Methods for Language Models, June 2024, [arxiv](https://arxiv.org/abs/2406.18664)
  
---
## LLM Systems
### RAG Evaluation
- Google Frames Dataset for evaluation of RAG systems, Sep 2024, [arxiv paper: Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation
- Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses, Oct 2024, Salesforce, [arxiv](https://arxiv.org/abs/2410.22349) [Answer Engine (RAG) Evaluation Repository](https://github.com/SalesforceAIResearch/answer-engine-eval)
](https://arxiv.org/abs/2409.12941) [Hugging Face, dataset](https://huggingface.co/datasets/google/frames-benchmark)
- RAGAS: Automated Evaluation of Retrieval Augmented Generation Jul 23, [arxiv](https://arxiv.org/abs/2309.15217)
- ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems Nov 23, [arxiv](https://arxiv.org/abs/2311.09476)
- Evaluating Retrieval Quality in Retrieval-Augmented Generation, Apr 2024, [arxiv](https://arxiv.org/abs/2404.13781)
- IRSC: A Zero-shot Evaluation Benchmark for Information Retrieval through Semantic Comprehension in Retrieval-Augmented Generation Scenarios, Sep 2024, [arxiv](https://arxiv.org/abs/2409.15763)
  
---
### Conversational systems
And Dialog systems
- A survey on chatbots and large language models: Testing and evaluation techniques, Jan 2025, [Natural Language Processing Journal Mar 2025](https://www.sciencedirect.com/science/article/pii/S2949719125000044)
- How Well Can Large Language Models Reflect? A Human Evaluation of LLM-generated Reflections for Motivational Interviewing Dialogues, Jan 2025, [Proceedings of the 31st International Conference on Computational Linguistics COLING](https://aclanthology.org/2025.coling-main.135/) 
- Benchmark for general-purpose AI chat model, December 2024, AILuminate from ML Commons, [mlcommons website](https://ailuminate.mlcommons.org/benchmarks/)
- Comparative Analysis of Finetuning Strategies and Automated Evaluation Metrics for Large Language Models in Customer Service Chatbots, Aug 2024, [preprint](https://www.researchsquare.com/article/rs-4895456/v1)
- Introducing v0.5 of the AI Safety Benchmark from MLCommons, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12241)
- Foundation metrics for evaluating effectiveness of healthcare conversations powered by generative AI Feb 24, [Nature](https://www.nature.com/articles/s41746-024-01074-z.epdf)
- CausalScore: An Automatic Reference-Free Metric for Assessing Response Relevance in Open-Domain Dialogue Systems, Jun 2024, [arxiv](https://arxiv.org/abs/2406.17300)
- Simulated user feedback for the LLM production, [TDS](https://towardsdatascience.com/how-to-make-the-most-out-of-llm-production-data-simulated-user-feedback-843c444febc7)
- How Well Can LLMs Negotiate? NEGOTIATIONARENA Platform and Analysis Feb 2024 [arxiv](https://arxiv.org/abs/2402.05863)
- Rethinking the Evaluation of Dialogue Systems: Effects of User Feedback on Crowdworkers and LLMs, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12994)
- A Two-dimensional Zero-shot Dialogue State Tracking Evaluation Method using GPT-4, Jun 2024, [arxiv](https://arxiv.org/abs/2406.11651)
  
---
### Copilots
- Tutor CoPilot: A Human-AI Approach for Scaling Real-Time Expertise, Stanford, Oct 2024, [arxiv](https://arxiv.org/abs/2410.03017)
- From Interaction to Impact: Towards Safer AI Agents Through Understanding and Evaluating UI Operation Impacts, University of Washington , Apple, Oct 2024, [arxiv](https://arxiv.org/abs/2410.09006)
- Copilot Evaluation Harness: Evaluating LLM-Guided Software Programming Feb 24 [arxiv](https://arxiv.org/abs/2402.14261)
- ELITR-Bench: A Meeting Assistant Benchmark for Long-Context Language Models, Apr 2024, [arxiv](https://arxiv.org/pdf/2403.20262.pdf)
  
---
### Search and Recommendation Engines
- Investigating Users' Search Behavior and Outcome with ChatGPT in Learning-oriented Search Tasks, SIGIR-AP 2024, [ACM](https://dl.acm.org/doi/abs/10.1145/3673791.3698406)
- Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation,[RecSys 2023](https://dl.acm.org/doi/abs/10.1145/3604915.3608860)
- Is ChatGPT a Good Recommender? A Preliminary Study Apr 2023 [arxiv](https://arxiv.org/abs/2304.10149)
- IRSC: A Zero-shot Evaluation Benchmark for Information Retrieval through Semantic Comprehension in Retrieval-Augmented Generation Scenarios, Sep 2024, [arxiv](https://arxiv.org/abs/2409.15763)
- LLMRec: Benchmarking Large Language Models on Recommendation Task, Aug 2023, [arxiv](https://arxiv.org/abs/2308.12241)
- OpenP5: Benchmarking Foundation Models for Recommendation, Jun 2023, [researchgate](https://www.researchgate.net/publication/371727972_OpenP5_Benchmarking_Foundation_Models_for_Recommendation)
- Marqo embedding benchmark for eCommerce [at Huggingface](https://huggingface.co/spaces/Marqo/Ecommerce-Embedding-Benchmarks), text to image and category to image tasks 
- LaMP: When Large Language Models Meet Personalization, Apr 2023,  [arxiv](https://arxiv.org/abs/2304.11406)
- Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses, Oct 2024, Salesforce, [arxiv](https://arxiv.org/abs/2410.22349) [Answer Engine (RAG) Evaluation Repository](https://github.com/SalesforceAIResearch/answer-engine-eval)
- BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives, Feb 2024, [arxiv](https://arxiv.org/abs/2402.14151)
- Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents, Apr 2023, [arxiv](https://arxiv.org/abs/2304.09542)
-  BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models, Oct 2021, [arxiv](https://arxiv.org/abs/2104.08663)
-  BENCHMARK : LoTTE,  Long-Tail Topic-stratified Evaluation for IR that features 12 domain-specific search tests, spanning StackExchange communities and using queries from GooAQ, [ColBERT repository wth the benchmark data](https://github.com/stanford-futuredata/ColBERT)
- LongEmbed: Extending Embedding Models for Long Context Retrieval, Apr 2024, [arxiv](https://arxiv.org/abs/2404.12096), benchmark for long context tasks, [repository for LongEmbed](https://github.com/dwzhu-pku/LongEmbed)
- Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT, Feb 2024, [arxiv](https://arxiv.org/abs/2402.07440), LoCoV1 benchmark for long context LLM,
-  STARK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases, Apr 2024, [arxiv](https://arxiv.org/abs/2404.13207) [code github](https://github.com/snap-stanford/stark)
-  Constitutional AI: Harmlessness from AI Feedback, Sep 2022 [arxiv](https://arxiv.org/abs/2212.08073) (See Appendix B Identifying and Classifying Harmful Conversations, other parts)
- 
---
### Task Utility
- Towards Effective GenAI Multi-Agent Collaboration: Design and Evaluation for Enterprise Applications, Dec 2024, [arxiv](https://arxiv.org/abs/2412.05449)
- Assessing and Verifying Task Utility in LLM-Powered Applications, May 2024, [arxiv](https://arxiv.org/abs/2405.02178)
- Towards better Human-Agent Alignment: Assessing Task Utility in LLM-Powered Applications, Feb 2024, [arxiv](https://arxiv.org/abs/2402.09015)
  
---
## Verticals
### Healthcare and medicine
- MedSafetyBench: Evaluating and Improving the Medical Safety of Large Language Models, Dec 2024, [openreview](https://openreview.net/pdf?id=cFyagd2Yh4) [arxiv](https://arxiv.org/abs/2403.03744) [benchmark code and data at github](https://github.com/AI4LIFE-GROUP/med-safety-bench)
- Evaluation of LLMs accuracy and consistency in the registered dietitian exam through prompt engineering and knowledge retrieval, Nature, Jan 2025, [Scientific reporta Nature](https://www.nature.com/articles/s41598-024-85003-w)
- Medical large language models are vulnerable to data-poisoning attacks, Jan 2025, [Nature Medicine](https://www.nature.com/articles/s41591-024-03445-1)
- A dataset and benchmark for hospital course summarization with adapted large language models, Dec 2024, [Journal of the American Medical Informatics Association](https://academic.oup.com/jamia/advance-article-abstract/doi/10.1093/jamia/ocae312/7934937?redirectedFrom=fulltext)
- MedQA-CS: Benchmarking Large Language Models Clinical Skills Using an AI-SCE Framework, Oct 2024, [arxiv](https://arxiv.org/abs/2410.01553)
- A framework for human evaluation of large language models in healthcare derived from literature review, September 2024, [Nature Digital Medicine](https://www.nature.com/articles/s41746-024-01258-7) 
- Evaluation and mitigation of cognitive biases in medical language models, Oct 2024 [Nature](https://www.nature.com/articles/s41746-024-01283-6)
- Foundation metrics for evaluating effectiveness of healthcare conversations powered by generative AI Feb 24, [Nature](https://www.nature.com/articles/s41746-024-01074-z.epdf)
- Evaluation and mitigation of the limitations of large language models in clinical decision-making, July 2024, [Nature Medicine](https://www.nature.com/articles/s41591-024-03097-1)
- Evaluating Generative AI Responses to Real-world Drug-Related Questions, June 2024, [Psychiatry Research](https://www.sciencedirect.com/science/article/abs/pii/S0165178124003433)
- MedExQA: Medical Question Answering Benchmark with Multiple Explanations, Jun 2024, [arxiv](https://arxiv.org/abs/2406.06331)
- Clinical Insights: A Comprehensive Review of Language Models in Medicine, Aug 2024, [arxiv](https://arxiv.org/abs/2408.11735) See table 2 for evaluation
- Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data Jan 2024 [arxiv](https://arxiv.org/abs/2401.06866)
- Evaluating LLM -- Generated Multimodal Diagnosis from Medical Images and Symptom Analysis, Jan 2024, [arxiv](https://arxiv.org/abs/2402.01730)
- A Comprehensive Survey on Evaluating Large Language Model Applications in the Medical Industry, May 2024, [arxiv](https://arxiv.org/abs/2404.15777)
- Evaluating large language models in medical applications: a survey, May 2024, [arxiv](https://arxiv.org/abs/2405.07468)
- MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering, 2022, [PMLR](https://proceedings.mlr.press/v174/pal22a.html)
- What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams, MedQA benchmark, Sep 2020, [arxiv](https://arxiv.org/abs/2009.13081)
- PubMedQA: A Dataset for Biomedical Research Question Answering, 2019, [acl](https://aclanthology.org/D19-1259/)
- [Open Medical LLM Leaderboard from HF](https://huggingface.co/blog/leaderboard-medicalllm) [Explanation](https://huggingface.co/blog/leaderboard-medicalllm)
- Evaluating Large Language Models on a Highly-specialized Topic, Radiation Oncology Physics, Apr 2023, [arxiv](https://arxiv.org/abs/2304.01938)
- Assessing the Accuracy of Responses by the Language Model ChatGPT to Questions Regarding Bariatric Surgery, Apr 2023, [pub med](https://pubmed.ncbi.nlm.nih.gov/37106269/)
- Can LLMs like GPT-4 outperform traditional AI tools in dementia diagnosis? Maybe, but not today, Jun 2023, [arxiv](https://arxiv.org/abs/2306.01499)
- Evaluating the use of large language model in identifying top research questions in gastroenterology, Mar 2023, [nature](https://www.nature.com/articles/s41598-023-31412-2)
- Evaluating AI systems under uncertain ground truth: a case study in dermatology, Jul 2023, [arxiv](https://arxiv.org/abs/2307.02191)
- MedDialog: Two Large-scale Medical Dialogue Datasets, Apr 2020, [arxiv](https://arxiv.org/abs/2004.03329)
- An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition, 2015, [article html](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0564-6)
- DrugBank 5.0: a major update to the DrugBank database for 2018, 2018, [paper html](https://academic.oup.com/nar/article/46/D1/D1074/4602867)]
- A Dataset for Evaluating Contextualized Representation of Biomedical Concepts in Language Models, May 2024, [nature](https://www.nature.com/articles/s41597-024-03317-w), [dataset](https://github.com/hrouhizadeh/BioWiC)
- MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records, Aug 2023, [arxiv](https://arxiv.org/abs/2308.14089)

---  
### Law
- LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models, [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html)
- LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain, [EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.200/)
- Multi-LexSum: Real-world Summaries of Civil Rights Lawsuits at Multiple Granularities [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/552ef803bef9368c29e53c167de34b55-Abstract-Datasets_and_Benchmarks.html)

---
### Science
- Unveiling the power of language models in chemical research question answering, Jan 2025, [Nature, communication chemistry](https://www.nature.com/articles/s42004-024-01394-x)
- SciRepEval: A Multi-Format Benchmark for Scientific Document Representations, 2022, [arxiv](https://arxiv.org/abs/2211.13308)
- What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks, NeurIPS 2023, [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bbb330189ce02be00cf7346167028ab1-Abstract-Datasets_and_Benchmarks.html)
- GPQA: A Graduate-Level Google-Proof Q&A Benchmark, Nov 2023, [arxiv](https://arxiv.org/abs/2311.12022) [gpqa benchmark dataset](https://huggingface.co/datasets/Idavidrein/gpqa)
- MATH Mathematics Aptitude Test of Heuristics, Measuring Mathematical Problem Solving With the MATH Dataset, Nov 2021 [arxiv](https://arxiv.org/abs/2103.03874)

---
### Math
-  How well do large language models perform in arithmetic tasks?, Mar 2023, [arxiv](https://arxiv.org/abs/2304.02015)
- FrontierMath at EpochAI, [FrontierAI page](https://epoch.ai/frontiermath), FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI, Nov 2024,  [arxiv](https://arxiv.org/abs/2411.04872)
-   Cmath: Can your language model pass chinese elementary school math test?, Jun 23, [arxiv](https://arxiv.org/abs/2306.16636)
-   GSM8K [paperwithcode](https://paperswithcode.com/dataset/gsm8k) [repository github](https://github.com/openai/grade-school-math)

---
### Financial
- Evaluating LLMs' Mathematical Reasoning in Financial Document Question Answering, Feb 24, [arxiv](https://arxiv.org/abs/2402.11194v2)
- PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance, Jun 2023, [arxiv](https://arxiv.org/abs/2306.05443)
- BloombergGPT: A Large Language Model for Finance (see Chapter 5 Evaluation), Mar 2023, [arxiv](https://arxiv.org/abs/2303.17564)
- FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets, Oct 2023, [arxiv](https://arxiv.org/abs/2310.04793)

---  
### Other
- Understanding the Capabilities of Large Language Models for Automated Planning, May 2023, [arxiv](https://arxiv.org/abs/2305.16151)

 --- 
## Other Collections 
- [LLM/VLM Benchmarks by Aman Chadha](https://aman.ai/primers/ai/benchmarks/)
- [Awesome LLMs Evaluation Papers](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers), a list of papers mentioned in the [Evaluating Large Language Models: A Comprehensive Survey](https://arxiv.org/pdf/2310.19736), Nov 2023

---  

## Citation

```
@article{Lopatenko2024CompendiumLLMEvaluation,
  title   = {Compendium of LLM Evaluation methods},
  author  = {Lopatenko, Andrei},
  year    = {2024},
  note    = {\url{https://github.com/alopatenko/LLMEvaluation}}
}
```



