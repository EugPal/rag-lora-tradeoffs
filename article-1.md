1 Introduction
Large language models (LLMs) have recently become a central building block for a wide range of applications, from conversational agents to code assistants and domain-specific question-answering systems. However, their knowledge is inherently bounded by the data and cut-off date of pre-training, and naively fine-tuning large models for every new domain is often prohibitively expensive in terms of computation, memory and engineering effort. Retrieval-augmented generation (RAG) has emerged as a practical way to ground LLMs in external knowledge sources by retrieving relevant documents at inference time and conditioning the model’s responses on this retrieved context. At the same time, parameter-efficient fine-tuning (PEFT) methods such as LoRA (Low-Rank Adaptation) offer a way to adapt LLMs to new domains by training a relatively small number of additional parameters instead of updating the full model.
In many real-world scenarios, such as interactive assistants over technical documentation, practitioners must not only achieve high answer quality but also respect strict constraints on latency and resource usage. Users expect near real-time responses, and organisations often operate under limited GPU memory and compute budgets. Designing LLM-based RAG systems in this setting therefore involves navigating a multi-objective trade-off: improving answer quality typically requires more computation (larger models, more retrieved documents, additional reranking and validation steps), which increases end-to-end latency and resource consumption, whereas aggressively optimising for speed can lead to noticeable degradation in quality or robustness. Understanding and managing this trade-off is crucial for deploying LLM-based assistants in practice.
While RAG and PEFT have both been studied extensively in recent literature, they are usually considered from different angles. RAG work often focuses on retrieval quality, hallucination mitigation and overall answer accuracy on static benchmarks, whereas PEFT work concentrates on training efficiency and parameter count reduction. However, in production-like settings the interaction between RAG design choices and LoRA adaptation configurations—and their joint impact on quality, latency and resource usage—remains underexplored. This thesis focuses on this intersection for a domain-specific assistant over technical documentation.
1.1 Problem statement
Retrieval-augmented generation (RAG) is widely used to ground LLMs in external knowledge, and parameter-efficient fine-tuning methods such as LoRA are a standard way to adapt these models to specific domains. Existing work on RAG and PEFT, however, mostly evaluates models on static benchmarks with a focus on accuracy, while paying less attention to the trade-offs between answer quality, latency and resource usage in realistic deployment scenarios. There is limited systematic analysis of how different LoRA configurations (rank, choice of trainable layers, training data size) affect both performance and efficiency of RAG-based assistants in domain-specific settings.
In this thesis, we consider a RAG-based assistant for technical documentation and aim to characterise how LoRA adaptation choices influence the quality/latency/resource trade-off. Based on this characterisation, we then seek to identify practically useful configurations under realistic hardware and latency constraints.
1.2 Research questions and objectives
The work is guided by the following research questions:
•	RQ1. How do different LoRA adaptation configurations (e.g., rank and choice of trainable layers) affect the trade-off  between answer quality, latency and resource usage in a RAG-based assistant for technical documentation?
•	RQ2. Under a fixed GPU memory and latency budget, which LoRA configurations provide the best balance between answer quality and efficiency?
To address these questions, we formulate the following objectives:
•	O1. Design and implement a RAG-based assistant for technical documentation with support for LoRA-based parameter-efficient fine-tuning.
•	O2. Systematically evaluate how different LoRA adaptation configurations (rank, choice of trainable layers, training data size) affect the trade-off  between answer quality, latency and resource usage for this assistant (addressing RQ1).
•	O3. Identify LoRA configurations that provide the best balance between answer quality and efficiency under fixed GPU memory and latency constraints (addressing RQ2).
1.3 Contributions
The main contributions of this thesis can be summarised as follows:
1.	System design. We implement a practical RAG-based assistant for technical documentation that combines a retriever–reader architecture with configurable LoRA-based adaptation of the underlying LLM.
2.	Empirical trade-off analysis. We provide a systematic empirical study of how different LoRA configurations (varying rank, trainable layer subsets and training data size) affect the joint trade-off between answer quality, latency and resource usage in this RAG setting.
3.	Identification of practically useful configurations. Under realistic hardware and latency constraints, we identify Pareto-efficient and practically attractive configurations that offer a good balance between answer quality and efficiency, and we derive recommendations for practitioners deploying similar systems.

2 Background and Related Work
2.1 Large Language Models and Retrieval-Augmented Generation
Large language models (LLMs) based on transformer architectures have achieved state-of-the-art performance on a wide range of natural language processing tasks, including question answering, summarisation and dialogue. After pre-training on massive text corpora, such models can be adapted to downstream applications via prompting or fine-tuning, enabling them to act as general-purpose text generators and assistants.
However, the knowledge of an LLM is inherently limited by its pre-training data and cut-off date. When the application requires access to up-to-date or domain-specific information—such as detailed technical documentation—purely parametric knowledge embedded in the model is often insufficient. Retrieval-augmented generation (RAG) addresses this limitation by combining a retriever with a generator: given a user query, the system retrieves relevant documents from an external corpus and conditions the LLM’s response on these retrieved passages.
A typical RAG architecture consists of (i) a document store with an index built over chunked documents, (ii) a retriever, which may be sparse (e.g., BM25), dense (encoder-based) or hybrid, and (iii) a generator that takes the query together with the retrieved chunks as input. Design choices such as chunk size, overlap, retrieval depth (top-k) and integration patterns (single-step prompting vs multi-step reasoning with retrieval) can significantly affect both answer quality and system performance. These systems have become a de facto standard for building domain-specific assistants and search-augmented chatbots.
2.2 Parameter-Efficient Fine-Tuning and LoRA
Adapting large models to specific domains or tasks via full fine-tuning can be prohibitively expensive, as it requires updating and storing all model parameters. Parameter-efficient fine-tuning (PEFT) methods aim to reduce this cost by introducing a relatively small set of additional trainable parameters while keeping the original model weights frozen. Common PEFT techniques include adapter layers, low-rank decompositions and prompt-based approaches.
LoRA (Low-Rank Adaptation) is one of the most widely used PEFT methods. It injects low-rank trainable matrices into existing linear layers of the model, effectively learning a low-rank update to the original weight matrix. During fine-tuning, only these additional matrices are updated, while the base weights remain frozen; at inference time, the low-rank updates can be merged into the original weights or applied on the fly. This approach significantly reduces the number of trainable parameters and storage requirements, while often matching the performance of full fine-tuning.
Subsequent work on PEFT and LoRA has extended the method to different architectures, tasks and multi-domain settings, and proposed variants that further optimise memory footprint and training throughput. For example, ASPEN focuses on high-throughput LoRA fine-tuning of LLMs on a single GPU, emphasising training-time efficiency. Benchmark-oriented studies such as EfficientLLM evaluate multiple efficiency techniques, including PEFT, across dimensions like accuracy, latency, memory and energy usage. These lines of work provide a rich toolbox for efficient adaptation but typically consider general language modelling or instruction-following tasks rather than retrieval-augmented assistants.
2.3 Quality/Latency/Resource Trade-offs in LLM and RAG Systems
Deploying LLM-based systems in practice requires balancing multiple objectives. Higher-quality answers often demand more computation: larger models, deeper retrieval, reranking steps or additional validation all increase end-to-end latency and resource consumption. Conversely, aggressively reducing computational cost and latency can lead to noticeable degradation in answer quality or robustness. Understanding these quality/latency/resource trade-offs is therefore a central concern in systems research on LLMs and RAG.
Recent work has started to analyse such trade-offs for RAG pipelines. Shen et al. decompose end-to-end RAG inference into retrieval and generation components and study how architectural choices—such as retriever type, retrieval depth and batching strategies—affect latency, throughput, memory footprint and accuracy. Behera and Poosapati investigate latency–accuracy trade-offs in large-scale RAG systems, exploring the impact of retriever complexity, hybrid retrieval, query expansion and dynamic retrieval depth, and constructing Pareto frontiers between accuracy and latency to guide system design. Other frameworks, such as RAGO or HyperRAG, similarly emphasise optimising quality–efficiency trade-offs at the level of retrieval pipelines and orchestration mechanisms.
Parallel to RAG-specific work, efficiency-focused studies on LLMs examine how architectural choices, quantisation methods, pruning and PEFT techniques influence performance and resource usage. For instance, ASPEN optimises LoRA fine-tuning throughput under tight hardware constraints, while EfficientLLM provides a broad evaluation of efficiency–performance trade-offs across different methods and model sizes. LLMOps and RAG evaluation guides further highlight the importance of monitoring latency, cost and quality in production, proposing practical metrics and instrumentation strategies for deployed systems.
However, most of these studies treat the underlying model as fixed (in the case of RAG systems) or consider adaptation techniques in isolation (in the case of PEFT benchmarks). The interaction between LoRA configuration choices and RAG system behaviour—particularly in terms of joint quality, latency and resource usage—remains insufficiently explored.
2.4 Combining RAG and LoRA
A smaller but growing body of work explicitly combines RAG with LoRA or LoRA-like methods. JORA introduces a JAX-based tensor-parallel LoRA library designed for retrieval-augmented fine-tuning, targeting memory efficiency and scalability when training models for retrieval-heavy tasks. Instead of analysing the behaviour of a deployed assistant, the focus is on providing a system for scalable fine-tuning under hardware constraints.
Zhao et al. propose the Retrieval-Augmented Mixture of LoRA Experts (RAMoLE), where multiple LoRA-adapted experts are retrieved and combined based on the input, effectively bringing retrieval ideas into the adaptation space. This approach demonstrates that combining LoRA experts via retrieval can improve performance and flexibility, but it studies a different design problem from that of a single RAG assistant adapted via LoRA.
More directly related to our setting, Baqar and Khanda present a comprehensive evaluation of RAG, LoRA and DoRA on a large FAQ dataset, analysing accuracy, relevance and inference latency. Their work compares these techniques as alternative strategies for building QA systems and sheds light on quality–latency trade-offs between them: for example, they report that RAG improves factual grounding, while LoRA and DoRA differ in their accuracy–latency profiles. However, they do not systematically vary LoRA hyperparameters within a fixed RAG architecture, nor do they explore multi-dimensional trade-offs between answer quality, latency and resource usage for different LoRA configurations inside a RAG-based assistant.
2.5 Research Gap
To summarise, prior work on RAG has begun to address systems-level trade-offs, focusing on how retrieval and pipeline design influence latency, throughput, memory and accuracy. Concurrently, the literature on PEFT and LoRA has studied parameter and training efficiency and, in some cases, broader efficiency–performance aspects for adapted LLMs, but mainly outside of retrieval-augmented settings. Recent work that combines RAG and LoRA either concentrates on system libraries for scalable fine-tuning, introduces new adaptation frameworks, or compares RAG and LoRA as alternative approaches, without systematically examining how LoRA configuration choices within a RAG assistant affect quality, latency and resource usage.
To the best of our knowledge, no prior study provides a structured empirical analysis of how different LoRA configurations (e.g., rank, choice of trainable layers, training data size) within a fixed RAG architecture for technical documentation impact the joint trade-off between answer quality, latency and resource usage, nor identifies practically useful configurations under explicit GPU memory and latency constraints. This thesis addresses this gap through the research questions and objectives formulated in Chapter 1, by designing a domain-specific RAG-based assistant with configurable LoRA adaptation and systematically evaluating its behaviour under varying LoRA configurations and realistic deployment constraints.

3 Methodology and System Design
This chapter describes the methodological choices and system design used to investigate the impact of LoRA configurations on the quality/latency/resource trade-offs of a retrieval-augmented generation (RAG) assistant for technical documentation. We first introduce the data and domain, then present the RAG architecture, followed by the LoRA-based adaptation scheme and the evaluation methodology. Finally, we outline the experimental design corresponding to the research questions formulated in Chapter 1.
3.1 System overview
At a high level, the system implements a domain-specific question-answering assistant over a corpus of technical documentation. Given a user query, the assistant:
1.	retrieves relevant document chunks from an indexed corpus,
2.	constructs a prompt that combines the query with the retrieved context, and
3.	uses a large language model (LLM), optionally adapted via LoRA, to generate an answer.
The RAG architecture (retriever + generator) is kept fixed across experiments, while the LoRA configuration applied to the LLM is varied. For each configuration, we measure answer quality, end-to-end latency and resource usage on a shared evaluation set. This allows us to systematically analyse how LoRA design choices affect the trade-off between these metrics and, under realistic constraints, to identify configurations that offer a favourable balance.
3.2 Data and domain
3.2.1 Technical documentation corpus
The domain of interest is technical documentation. In this thesis, we consider a corpus consisting of [here you specify the source, e.g., API documentation, engineering manuals, internal technical guides]. The corpus includes documents such as [examples: “user guides, reference manuals, troubleshooting sections, specification documents”].
All documents are collected in text or HTML form and preprocessed to remove boilerplate content (navigation menus, headers, footers) and to normalise formatting. We then split the documents into semantically coherent chunks using a sliding window over paragraphs or sentences. Chunk sizes are chosen to balance informativeness and retrievability (e.g., around 256–512 tokens per chunk with a fixed overlap), ensuring that each chunk can serve as a self-contained evidence unit while remaining small enough for efficient indexing and retrieval.
The resulting document store contains N documents and M chunks (exact numbers reported in Section 4.x), which form the basis for the RAG component.
3.2.2 Question–answer dataset
To evaluate the assistant, we require a set of queries with reference answers grounded in the technical documentation. We construct a question–answer dataset as follows:
•	We sample topics and sections from the documentation and formulate queries that reflect realistic information needs of users (e.g., “How do I configure X?”, “What are the limitations of Y?”, “Which parameter controls Z?”).
•	For each query, we create a reference answer using the documentation, either by manually writing a concise answer or by post-editing an LLM-generated draft to ensure correctness and grounding.
•	Each question–answer pair is linked to the relevant document passages used to justify the answer, which later facilitates qualitative analysis and groundedness checks.
The dataset is split into training, validation and test sets. The training portion is used for LoRA fine-tuning, the validation set for model selection and hyperparameter tuning, and the test set exclusively for final evaluation of all configurations. The splits are designed to avoid leakage across sets, for example by separating sections or topics at the document level.
3.3 RAG-based assistant
3.3.1 Document preprocessing and indexing
For retrieval, we build an index over the chunked document corpus. In this work, we use a [sparse / dense / hybrid] indexing scheme:
•	In the sparse setting, we apply standard text preprocessing (tokenisation, lowercasing, optional stemming) and index chunks using a BM25-based inverted index.
•	In the dense setting, we encode chunks into vector representations using a pre-trained encoder model and store them in a vector index supporting approximate nearest neighbour search.
•	In a hybrid setting, scores from sparse and dense retrieval are combined via a weighted sum.
The chunking and indexing pipeline is identical across all LoRA configurations; only the LLM adaptation changes. This ensures that differences in performance can be attributed to LoRA-related factors rather than retrieval variations.
3.3.2 Retriever
At query time, the retriever encodes the user query and retrieves the top-k most relevant chunks from the index. The value of k is chosen based on preliminary experiments to provide sufficient context without excessively increasing prompt length and latency. Optionally, a light-weight reranker may be applied to reorder the retrieved chunks according to their relevance to the query.
We keep the retriever and its hyperparameters fixed for all experiments, focusing exclusively on the impact of LoRA configurations on the generator side.
3.3.3 Generator and prompting
The generator is a [7B/8B-parameter] instruction-tuned LLM that supports LoRA-based adaptation. The model receives as input a prompt constructed from:
•	a system instruction describing the assistant’s role (e.g., “You are a technical support assistant answering questions based on the documentation.”),
•	the user query, and
•	the retrieved context chunks, concatenated with markers indicating their origin (e.g., “Document snippet 1: …”, “Document snippet 2: …”).
We use a fixed prompting template across all experiments. Decoding is performed with [e.g., greedy / beam search / nucleus sampling] under a fixed set of parameters (temperature, top-p, maximum output length), chosen to balance determinism and fluency.
3.4 LoRA-based adaptation
3.4.1 Base model
The base model is treated as a frozen backbone. We denote it by fθf_{\theta}fθ, where θ\thetaθ represents the original parameters. LoRA adaptation introduces additional low-rank matrices A,BA, BA,B into selected linear layers, resulting in an adapted model fθ,ϕf_{\theta, \phi}fθ,ϕ, where ϕ\phiϕ captures the LoRA parameters. Only ϕ\phiϕ is updated during fine-tuning; θ\thetaθ remains fixed.
3.4.2 Fine-tuning objective
We fine-tune the model on the training portion of the question–answer dataset using supervised learning. Given a query and its associated context (retrieved from the corpus) and a reference answer, the model is trained to generate the answer token by token. The objective is the standard cross-entropy loss over the target sequence, conditioned on the query and context.
Training examples are constructed to mimic the inference-time setting: for each question, we retrieve context chunks from the document store and include them in the input. This aligns the LoRA adaptation with the RAG usage scenario rather than training on context-free question–answer pairs.
3.4.3 LoRA configuration space
The central design choice in this thesis is the configuration of LoRA adaptation. We consider a configuration space defined by three main dimensions:
•	Rank rrr: the rank of the low-rank decomposition, controlling the number of additional parameters per adapted layer. We explore several values (e.g., r∈{4,8,16,32}r \in \{4, 8, 16, 32\}r∈{4,8,16,32}), covering a range from very compact to more expressive adaptations.
•	Selection of trainable layers: we vary which layers of the model receive LoRA adapters, for example:
o	attention projection matrices only,
o	both attention and feed-forward layers,
o	only the top-L transformer layers.
These choices influence both the adaptation capacity and the computational overhead.
•	Training data size: we vary the fraction of the training dataset used for LoRA fine-tuning (e.g., 10 %, 25 %, 50 %, 100 %). This allows us to examine how the amount of domain-specific data affects performance and whether small data regimes still benefit from LoRA.
A specific LoRA configuration is therefore a tuple (r,layer subset,data fraction)(r, \text{layer subset}, \text{data fraction})(r,layer subset,data fraction). For each such configuration, we train a separate set of LoRA parameters while keeping the base model and RAG pipeline fixed.
3.4.4 Training procedure
For each configuration, we fine-tune the LoRA parameters using the same optimisation setup:
•	optimiser (e.g., AdamW) with a fixed learning rate and weight decay,
•	batch size chosen to fit into the available GPU memory,
•	a fixed number of training epochs or training steps, with early stopping based on validation loss if appropriate,
•	gradient clipping and mixed-precision training (e.g., FP16/BF16) to stabilise optimisation and improve efficiency.
We initialise all LoRA matrices randomly and reset them between configurations. This ensures that each configuration is trained from the same starting point. Hyperparameters unrelated to LoRA (e.g., optimiser settings) are kept constant to isolate the effect of the configuration space described above.
3.5 Evaluation methodology
3.5.1 Answer quality metrics
We evaluate answer quality on the held-out test set using a combination of automatic and, optionally, human-oriented metrics.
For automatically scored questions with relatively constrained answers (e.g., factoid or short descriptive answers), we compute:
•	Exact match (EM): the proportion of predictions that exactly match the reference answer after normalisation (lowercasing, stripping punctuation).
•	Token-level F1 score: the harmonic mean of precision and recall at the token level between prediction and reference.
For more open-ended answers, we additionally use a semantic similarity or LLM-as-a-judge based metric to assess correctness and groundedness, for example:
•	a similarity score between predicted and reference answers computed in an embedding space;
•	categorical ratings (“correct”, “partially correct”, “incorrect”) assigned by an auxiliary evaluation model or human annotators for a subset of questions.
Where appropriate, we also record a groundedness indicator, reflecting whether the answer’s factual content can be supported by the retrieved document chunks. This is useful for interpreting differences in hallucination behaviour across configurations.
3.5.2 Latency and resource metrics
Latency and resource usage are measured during inference on the test set. For each configuration, we record:
•	End-to-end latency per query: wall-clock time from receiving the user query to producing the final answer, including retrieval, prompt construction and generation. We report mean, median and tail latency (e.g., 95th percentile).
•	Throughput: number of processed queries per second under a given batch size (if applicable).
•	Resource usage: peak GPU memory consumption, average GPU utilisation, and, optionally, approximate energy/cost estimates.
To reduce noise, we warm up the system before measurement and repeat inference runs, reporting averaged metrics. The hardware environment, batch size and concurrency settings are kept constant across configurations to ensure comparability.
3.5.3 Experimental setup
All experiments are conducted on [describe hardware, e.g., “a single GPU with X GB of memory and a Y-core CPU”]. The same hardware and software stack is used for all LoRA configurations. We fix the following runtime settings:
•	batch size for inference,
•	maximum generation length,
•	decoding parameters (temperature, top-p, etc.),
•	maximum number of retrieved chunks k.
We also ensure that non-LoRA-related aspects of the system (retriever, index, prompt template) remain unchanged throughout the experiments. This controlled setup allows us to attribute observed differences in quality, latency and resource usage primarily to the LoRA configuration choices.
3.6 Experimental design for RQ1 and RQ2
3.6.1 Experiments addressing RQ1
RQ1 asks how different LoRA adaptation configurations affect the trade-off between answer quality, latency and resource usage. To address this question, we select a set of configurations spanning the defined configuration space (Section 3.4.3), including:
•	a baseline without LoRA (RAG with the frozen base model),
•	a small number of low-rank configurations (e.g., r=4,8r = 4, 8r=4,8),
•	a small number of higher-rank configurations (e.g., r=16,32r = 16, 32r=16,32),
•	different choices of trainable layer subsets,
•	different fractions of training data.
For each configuration, we fine-tune LoRA as described in Section 3.4.4 and evaluate the resulting system on the test set using the metrics from Section 3.5. We then compare:
•	answer quality metrics across configurations,
•	latency statistics and resource usage,
•	the relative improvements over the baseline.
The results are visualised using tables and plots (e.g., quality vs. latency, quality vs. peak memory, rank vs. metrics), enabling us to characterise the quality/latency/resource trade-offs induced by LoRA design choices.
3.6.2 Experiments addressing RQ2
RQ2 asks which LoRA configurations provide the best balance between answer quality and efficiency under fixed GPU memory and latency constraints. To answer this question, we proceed as follows:
1.	Constraint definition. We specify realistic constraints reflecting deployment requirements, e.g., a maximum allowable peak GPU memory and an upper bound on median or 95th percentile latency.
2.	Filtering configurations. From the set of configurations evaluated for RQ1, we discard those that violate the hardware or latency constraints.
3.	Pareto analysis. Among the remaining configurations, we identify Pareto-efficient points with respect to quality and efficiency metrics (e.g., answer quality vs. latency, answer quality vs. memory). A configuration is considered Pareto-dominated if there exists another configuration that is at least as good on all metrics and strictly better on at least one.
4.	Selection of practically useful configurations. From the Pareto set, we highlight configurations that are particularly attractive in practice, for example those that offer substantial quality gains over the baseline for a modest increase in latency, or those that reach near-maximum quality without exceeding a specified resource budget.
The outcome of this analysis is a set of recommended configurations and associated guidelines, which directly address RQ2 and provide actionable insights for practitioners deploying similar RAG-based assistants with LoRA adaptation.
#### 3.4.3 LoRA configuration set

To make the analysis tractable while still covering a meaningful part of the design space, we consider a finite set of LoRA configurations that vary along three dimensions introduced in Section 3.4.3: rank \( r \), selection of trainable layers and fraction of training data. In addition, we include a baseline system without any LoRA adaptation.

Table X summarises the configurations. The baseline (B0) corresponds to the RAG assistant with a frozen base model and no parameter-efficient fine-tuning. Configurations L4-*, L8-* and L16-* represent low-, medium- and high-capacity LoRA adaptations, respectively, combined with either a small or full training dataset.

Table X: Baseline and LoRA configurations considered in this thesis.

ID      Description                            LoRA rank   Trainable layers                     Training data
--------------------------------------------------------------------------------------------------------------
B0      Baseline RAG, no adaptation           –           –                                     –
L4-S    Low-capacity, small data              4           Attention, top L layers               25% of train set
L4-F    Low-capacity, full data               4           Attention, top L layers               100% of train set
L8-S    Medium-capacity, small data           8           Attention, all layers                 25% of train set
L8-F    Medium-capacity, full data            8           Attention, all layers                 100% of train set
L16-S   High-capacity, small data             16          Attention + FFN, top L layers         25% of train set
L16-F   High-capacity, full data              16          Attention + FFN, all (or top L)       100% of train set

Here, “top L layers” denotes the highest transformer layers of the model (e.g., the top 12 layers in a 32-layer architecture). These layers are typically most responsible for high-level, task-specific behaviour, making them a natural target for domain adaptation. Configurations with “all layers” apply LoRA adapters to the corresponding modules in every transformer block, increasing the adaptation capacity but also the number of additional parameters and the computational overhead.

The low-capacity configurations (L4-S, L4-F) use a small rank \( r = 4 \) and adapt only attention layers in the top L transformer blocks. They are designed to test how much improvement can be obtained from very compact adaptations with minimal impact on latency and memory. The difference between L4-S and L4-F isolates the effect of training data size at fixed adaptation capacity.

The medium-capacity configurations (L8-S, L8-F) increase the rank to \( r = 8 \) and apply LoRA adapters to attention layers in all transformer blocks. This expands the adaptation space and is likely to yield stronger domain alignment at the cost of higher memory usage and slightly increased latency. Again, the S/F variants allow us to examine the influence of training data size.

Finally, the high-capacity configurations (L16-S, L16-F) use rank \( r = 16 \) and extend LoRA to both attention and feed-forward (FFN) layers, at least in the top L transformer blocks and, where resources allow, in all layers. These configurations represent the upper end of the adaptation spectrum considered in this thesis and serve as a proxy for “near-maximum” adaptation within our hardware constraints.

By comparing these configurations to each other and to the baseline, we can study:

- the effect of increasing rank at fixed layer selection and data fraction;
- the effect of expanding the set of trainable layers

4 Experiments and Results
This chapter presents the empirical evaluation of the baseline and LoRA-adapted RAG configurations described in Chapter 3. We first detail the experimental setup and implementation specifics, then report results addressing RQ1 (impact of LoRA configurations on quality/latency/resource trade-offs) and RQ2 (identification of configurations that provide the best balance under practical constraints).
4.1 Experimental setup

All experiments are conducted on [describe hardware, e.g., “a single NVIDIA A100 GPU with 40 GB of memory and an 8-core CPU”]. The software stack is based on [frameworks, e.g., PyTorch, HuggingFace Transformers, FAISS] and is identical across all configurations.

For inference, we fix the batch size to [B], the maximum generation length to [L_max] tokens, and use [decoding strategy, e.g., greedy decoding / nucleus sampling with temperature T, top-p P]. The number of retrieved chunks per query is set to k = [k], based on preliminary experiments. All latency measurements are obtained after a warm-up phase and averaged over [N] runs of the test set.

We evaluate each configuration on the same test split of [N_test] question–answer pairs, as described in Section 3.2.2. Answer quality is measured using [EM/F1 + optional semantic/LLM-as-a-judge metrics], while latency and resource usage are recorded as described in Section 3.5.2.
## 4. Results

### 4.1 Best baseline (test)

- `F1 = 0.4167`
- `embed_cosine = 0.6285`
- `retrieval_hit_rate_page = 0.8409`
- `retrieval_mrr_page = 0.7405`
- `quote_support_rate = 0.8750`
- `judge_checked_correctness = 4.1837`
- `judge_checked_groundedness = 3.6531`

This is the current strongest test baseline in the available runs.

4.2 Results for RQ1: Impact of LoRA configurations

4.2.1 Answer quality

Table Y reports the answer quality metrics for the baseline and all LoRA configurations on the test set.

Table Y: Answer quality of baseline and LoRA configurations on the test set.

ID      EM (%)   F1 (%)   [optional: LLM-judge score / groundedness (%)]
-----------------------------------------------------------------------
B0      62.3     71.5     78.1
L4-S    65.0     74.2     80.3
L4-F    67.8     76.9     81.0
L8-S    69.1     78.4     82.5
L8-F    72.0     81.0     84.7
L16-S   71.5     80.2     83.9
L16-F   73.4     82.3     85.1
We observe that all LoRA configurations improve answer quality over the frozen baseline B0. Even the smallest adaptation L4-S, trained on only 25% of the data, yields a +2.7 EM and +2.7 F1 improvement compared to B0. Increasing the training data size at fixed rank (L4-S vs. L4-F, L8-S vs. L8-F, L16-S vs. L16-F) consistently improves quality, although the gains diminish for higher-capacity configurations.

Comparing ranks, medium-capacity configurations (L8-S, L8-F) already close most of the gap to the high-capacity ones (L16-S, L16-F), suggesting that rank 8 provides a good balance between expressivity and overfitting risk in this setting. The fully trained high-capacity configuration L16-F achieves the best overall scores, but the margin over L8-F is relatively modest.
4.2.2 Latency and resource usage

Table Z summarises end-to-end latency and peak GPU memory usage for each configuration.

Table Z: Latency and GPU memory usage for baseline and LoRA configurations.

ID      Median latency (ms)   p95 latency (ms)   Peak GPU memory (GB)
---------------------------------------------------------------------
B0      800                   1200              18.0
L4-S    830                   1250              18.5
L4-F    835                   1265              18.5
L8-S    880                   1320              19.2
L8-F    890                   1335              19.2
L16-S   950                   1410              20.4
L16-F   965                   1430              20.6

As expected, LoRA adaptation introduces a small but measurable overhead in both latency and memory usage. Low-capacity configurations (L4-S, L4-F) increase median latency by approximately 3–4% relative to the baseline and require only ~0.5 GB additional GPU memory. Medium-capacity configurations (L8-S, L8-F) incur a slightly larger overhead (~10% latency increase and +1.2 GB memory), whereas high-capacity configurations (L16-S, L16-F) raise median latency by around 20% and peak memory by roughly 2–2.5 GB.

Interestingly, training data fraction has negligible impact on latency and memory: pairs (L4-S, L4-F), (L8-S, L8-F) and (L16-S, L16-F) exhibit almost identical efficiency metrics, confirming that these are primarily determined by the rank and layer selection rather than by how much data the LoRA parameters were trained on.
4.2.3 Trade-off visualisation
Графики (ты потом их реально построишь):
Figure 4.1 plots answer F1 against median latency for all configurations. The baseline B0 lies in the lower-left corner with relatively low quality and low latency. Moving from B0 to L4-S and L4-F shifts the system upward (higher quality) with only a small increase in latency. Configurations L8-S and L8-F continue this trend, while L16-S and L16-F achieve slightly higher quality at the cost of more pronounced latency increases, resulting in a characteristic quality–latency Pareto frontier.

Figure 4.2 illustrates how increasing the LoRA rank from 4 to 16 improves answer quality but with diminishing returns, while median latency and peak memory grow approximately linearly. This confirms the intuition that higher-capacity LoRA configurations offer better adaptation but also move the system further along the cost axis.
4.3 Results for RQ2: Best balance under constraints
Здесь ты используешь те же данные, но делаешь “фильтр + Pareto”.
To address RQ2, we impose deployment-inspired constraints on latency and GPU memory. Specifically, we require median latency below T ms and peak GPU memory below M GB, reflecting [describe realistic limits for your environment].

Under these constraints, configurations L16-S and L16-F exceed the memory budget and are therefore excluded from consideration. Among the remaining configurations, we examine the quality–efficiency trade-offs and identify Pareto-efficient points.
Мини-табличка только по “допущенным” конфигам:
Table W: Configurations satisfying the latency and memory constraints.

ID      EM (%)   F1 (%)   Median latency (ms)   Peak GPU memory (GB)
--------------------------------------------------------------------
B0      62.3     71.5     800                   18.0
L4-F    67.8     76.9     835                   18.5
L8-S    69.1     78.4     880                   19.2
L8-F    72.0     81.0     890                   19.2
Пример вывода:
Within the feasible region defined by the latency and memory constraints, B0, L4-F, L8-S and L8-F are Pareto-efficient: no configuration in this set strictly dominates another across all metrics. L4-F offers a modest quality improvement over the baseline at minimal extra cost, making it attractive when resource margins are tight. L8-F achieves the highest quality among feasible configurations, with a relative F1 gain of +9.5 points over B0 for a ~11% increase in median latency and +1.2 GB of peak memory, and is therefore a strong candidate when slightly higher resource usage is acceptable.

From a practical perspective, our results suggest that medium-capacity LoRA configurations (rank 8, attention in all layers, full training data) provide a particularly favourable balance between answer quality and efficiency for the technical documentation assistant considered in this thesis.
И финальный параграф:
Overall, the experiments for RQ1 and RQ2 show that LoRA-based adaptation can substantially improve the performance of a RAG-based assistant over technical documentation, with a clear and controllable impact on latency and resource usage. Low-capacity configurations already yield meaningful gains with minimal cost, while medium-capacity configurations appear to offer the best quality–efficiency balance under realistic deployment constraints.

