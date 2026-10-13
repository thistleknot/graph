# Blind reading packet: which answer is best?

For each of the 25 items below, read the four answers (labelled X1-X4, random order) and rank them best to worst. Judge: (1) addresses the query directly; (2) specific, substantive content (mechanisms, named methods, categories, results) rather than generic statements; (3) no padding, hedging or disclaimers; (4) claims carry citations in square brackets. The query is a sentence lifted from one paper.

Write one line per item in `blind_scores.txt`, for example `item 3: X2 > X1 > X4 > X3`. Do not open `blind_key.json` until all items are ranked. Then run `python docs/experiments/blind_score.py`.

---

## Item 1

**Query:** - · LoRA : Like Prefix tuning , low-rank adaptation (LoRA) [11] is suitable for SA-Text+Embeddings or SA-Embeddings neighbor encodings.

**X1**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method suitable for adapting large language models (LLMs) to new tasks or domains [arxiv/2310_07478#7]. It works by injecting trainable, low-rank matrices into the layers of a pre-trained model, while keeping the original parameters frozen [arxiv/2310_07478#7]. This significantly reduces the number of parameters that need to be trained, making fine-tuning more resource-efficient and resulting in low storage and memory overheads [arxiv/2310_18547#5].
LoRA is particularly well-suited for neighbor encodings like SA-Text+Embeddings and SA-Embeddings [arxiv/2310_07478#7]. In these approaches, LoRA injects trainable rank decomposition matrices into each layer, effectively representing the weight difference between the pre-trained and fine-tuned model as the product of two smaller matrices [arxiv/2310_18547#5]. This contrasts with methods like Prefix tuning, which optimize a sequence of continuous task-specific vectors prepended to activations [arxiv/2310_07478#7].
The effectiveness of LoRA in these contexts is supported by experimental results. For instance, when comparing PEFT methods for SA-TE and SA-E neighbor encodings, LoRA demonstrates strong performance, although Prefix tuning can achieve comparable results with fewer fine-tuned parameters in some cases [arxiv/2310_07478#14]. LoRA's ability to efficiently adapt LLMs makes it a valuable technique for tasks involving the integration of additional information, such as neighbor encodings.

**X2**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique suitable for SA-Text+Embeddings or SA-Embeddings neighbor encodings [arxiv/2310_07478#7]. It works by constraining the update matrix to be low-rank through matrix decomposition, significantly reducing the number of trainable parameters and computational costs compared to full fine-tuning [arxiv/2310_07478#7]. This is achieved by training two smaller matrices, A and B, whose product approximates the weight difference between the pre-trained and fine-tuned model [arxiv/2310_07478#7].
In the context of multimodal large language models (MMGL), LoRA is used for parameter-efficient fine-tuning when employing Self-Attention with Text+Embeddings (SA-TE) and Self-Attention with Embeddings (SA-E) neighbor encodings [arxiv/2310_07478#14]. It has been shown to perform comparably to or better than Prefix tuning for these neighbor encoding methods, although it typically involves a higher percentage of fine-tuned parameters [arxiv/2310_07478#14].

**X3**

These neighbor encoding methods are used to process multimodal information.

**X4**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique suitable for SA-Text+Embeddings or SA-Embeddings neighbor encodings [arxiv/2310_07478#7]. It works by constraining the update matrix to be low-rank through matrix decomposition, significantly reducing the number of trainable parameters and computational costs compared to full fine-tuning [arxiv/2310_07478#7, arxiv/2501_00365#2]. This method injects trainable rank decomposition matrices into each layer while keeping the original parameters frozen [arxiv/2310_07478#7].
LoRA's core mechanism involves approximating the weight update $\Delta W$ with two low-rank matrices, $A$ and $B$, such that $\Delta W = BA$ [arxiv/2409_05746#12, arxiv/2501_00365#2]. This decomposition reduces the number of parameters from $d \times m$ to $(d \times r) + (r \times m)$, where $r$ is the rank and $r \ll \min(d, m)$ [arxiv/2409_05746#12]. The original pretrained weights $W_0$ are frozen, and only the low-rank matrices $A$ and $B$ are trained [arxiv/2501_00365#2]. The forward pass is then computed as $f(x) = W_0x + \frac{\alpha}{r}BAx$, where $\alpha/r$ is a scaling factor [arxiv/2501_00365#2].
LoRA offers several advantages, including parameter efficiency, enhanced training efficiency, and flexible modular adaptation [arxiv/2310_07478#7, arxiv/2501_00365#2]. It also avoids additional inference latency because the low-rank updates can be merged into the original frozen weights [arxiv/2311_03285#3, arxiv/2501_00365#2]. This makes LoRA a popular choice for adapting large language models to downstream tasks with reduced computational overhead [arxiv/2405_12130#2, arxiv/2406_09044#14].
Various extensions and variants of LoRA exist, such as QLoRA, which integrates LoRA with 4-bit quantization to further reduce memory usage [arxiv/2405_12130#2, arxiv/2403_14608#15]. Other methods like AdaLoRA dynamically adjust the rank, while DoRA decomposes weights into magnitude and directional components [arxiv/2406_09044#14].

---

## Item 2

**Query:** The recent evolution of LLMs has significantly advanced their human-like reasoning capabilities.

**X1**

Recent advancements in Large Language Models (LLMs) have significantly enhanced their human-like reasoning capabilities. A key development is the "chain-of-thought" (CoT) prompting strategy, which encourages LLMs to generate step-by-step reasoning processes rather than direct conclusions [arxiv/2501_09223#95, arxiv/2303_18223#72, arxiv/2503_16419#2]. This approach allows LLMs to break down complex problems into smaller, sequential steps, mirroring human problem-solving and making the reasoning process more transparent and interpretable [arxiv/2501_09223#95].
CoT methods can be implemented in various ways, including few-shot CoT, which provides examples of detailed reasoning, and zero-shot CoT, which uses specific instructions like "Let's think step-by-step" to prompt the model to generate its reasoning process [arxiv/2501_09223#95]. These techniques have proven effective across a range of tasks, including mathematical, logical, and commonsense reasoning [arxiv/2501_09223#95, arxiv/2303_18223#72].
Further progress in LLM reasoning involves developing more efficient reasoning processes. This includes model-based approaches that optimize full-length reasoning models into more concise versions or train efficient reasoning models directly, as well as reasoning output-based methods that dynamically reduce reasoning steps during inference [arxiv/2503_16419#1]. Input prompt-based methods also aim to enhance reasoning efficiency by controlling prompt properties like difficulty or length [arxiv/2503_16419#1]. These efforts are leading to the development of "Large Reasoning Models" (LRMs) that exhibit improved performance in complex reasoning domains [arxiv/2503_16419#1, arxiv/2503_16419#2].
The evolution of LLMs towards more human-like reasoning is also being driven by techniques that scale test-time compute, encouraging more deliberate reasoning steps, and by reinforcement learning methods that provide step-wise rewards to improve reasoning capabilities [arxiv/2501_09686#2]. These combined approaches are paving the way for LLMs to generalize to complex real-world problems through abstract and logical reasoning [arxiv/2501_09686#2].

**X2**

Recent advancements in large language models (LLMs) have significantly enhanced their human-like reasoning capabilities. Techniques such as chain-of-thought (CoT) prompting allow LLMs to decompose complex problems into smaller, sequential reasoning steps, mirroring human problem-solving behaviors and making reasoning processes more transparent and interpretable [arxiv/2501_09223#95]. This approach has proven effective in improving reasoning accuracy across various tasks, including mathematical, logical, and commonsense reasoning [arxiv/2501_09223#95, arxiv/2303_18223#72].
Further enhancements to LLM reasoning involve methods like retrieval-augmented generation (RAG) for better knowledge integration and curriculum learning to improve reasoning skills [arxiv/2501_09686#2]. The Inner Thinking Transformer (ITT) is a novel approach that dynamically allocates more computational "thinking steps" to complex tokens, thereby improving LLM reasoning without increasing model parameters [arxiv/2501_09686#2]. These developments aim for more efficient and accurate complex reasoning, moving LLMs towards more sophisticated cognitive architectures [arxiv/2501_09686#2].
Despite these advancements, LLMs still face challenges, particularly with fine-grained hard constraints, indicating limitations in their current reasoning abilities [arxiv/2501_09223#95]. Research is ongoing to address issues like the "overthinking phenomenon," where LLMs generate verbose and redundant outputs, by developing efficient reasoning techniques that optimize reasoning length while preserving accuracy [arxiv/2503_16419#1, arxiv/2503_16419#2]. This includes model-based, reasoning output-based, and input prompt-based efficient reasoning strategies [arxiv/2503_16419#1].

**X3**

Recent advancements in large language models (LLMs) have significantly enhanced their human-like reasoning capabilities. Techniques such as chain-of-thought (CoT) prompting and retrieval-augmented generation (RAG) are employed to improve multi-step reasoning and knowledge integration, aiming for more efficient and accurate complex reasoning [arxiv/2303_18223#72, arxiv/2503_19470#2]. LLMs excel at generating human-level rationales, which suggests these advancements [arxiv/2412_06975#23].
Methods like curriculum learning are also applied to LLMs to enhance reasoning capabilities [arxiv/2501_09686#2]. Furthermore, the Inner Thinking Transformer (ITT) offers a novel approach to improve LLM reasoning without increasing parameters, by dynamically allocating more computational "thinking steps" to complex tokens [arxiv/2502_13842#3]. However, despite these advancements, LLMs still struggle with fine-grained hard constraints, indicating limitations in their current reasoning abilities [arxiv/2303_18223#72].

**X4**

Recent advancements in Large Language Models (LLMs) have indeed led to significant improvements in their human-like reasoning capabilities. These models are increasingly adept at complex reasoning tasks, including knowledge, symbolic, and mathematical reasoning [arxiv/2303_18223#72]. Techniques like Chain-of-Thought (CoT) prompting have been instrumental in eliciting step-by-step reasoning processes, enhancing accuracy and enabling more sophisticated cognitive architectures [arxiv/2501_09686#2].
The evolution of LLMs has seen a shift towards training models to master human-like reasoning processes, moving beyond simple next-token prediction. This includes developing methods to automatically generate accurate reasoning trajectories and using Process Reward Models (PRMs) to provide dense, step-wise rewards, creating a cycle for augmenting LLM reasoning [arxiv/2501_09686#2]. Furthermore, scaling test-time compute by encouraging more deliberate reasoning steps has also been shown to improve accuracy, suggesting a path towards "Large Reasoning Models" [arxiv/2501_09686#2].
While LLMs show remarkable inductive reasoning capabilities, they still face challenges with deductive reasoning, particularly in counterfactual scenarios or when precise instruction following is required [arxiv/2408_00114#9]. Research is actively exploring ways to enhance these reasoning abilities, including optimizing reasoning models for conciseness and efficiency, and developing new benchmarks to evaluate complex reasoning scenarios across various modalities [arxiv/2503_16419#2, arxiv/2509_11944#6]. Despite these advancements, the debate continues on whether LLMs genuinely reason like humans or merely simulate reasoning processes [arxiv/2408_00114#14].

---

## Item 3

**Query:** Question :	Can	biomimetic	materials	with	a	lamellar	structure	be	engineered	to	exhibit	self-healing	properties	under	mechanical	stress?

**X1**

Yes, biomimetic materials with a lamellar structure can be engineered to exhibit self-healing properties under mechanical stress. This is an active area of scientific inquiry, with researchers posing questions about how to achieve this through synthetic biology and material design [arxiv/2409_05556#320].
Self-healing materials are designed to autonomously repair damage, such as that caused by mechanical stress, by incorporating mechanisms that trigger healing agents [arxiv/2502_13025#65]. Biomimetic approaches can be used to create these materials, for instance, by producing bio-based polymers through microbial fermentation techniques [arxiv/2502_13025#106].
While the provided context does not explicitly detail how lamellar structures specifically contribute to self-healing under mechanical stress, it does indicate that biomaterials often exhibit a lamellar structure, which is important for their mechanical properties and functionality [arxiv/2409_05556#249]. Furthermore, research into animal-inspired adaptive properties includes synthesizing silk proteins and integrating them into polymer matrices, suggesting a pathway for developing advanced biomimetic materials [arxiv/2502_13025#108].

**X2**

Yes, biomimetic materials with a lamellar structure can be engineered to exhibit self-healing properties under mechanical stress. The lamellar structure itself can facilitate self-healing by enabling the redistribution of stress and the repair of minor cracks, similar to natural biological systems [arxiv/2409_05556#294, arxiv/2409_05556#274]. This layered composition provides strength and resilience [arxiv/2409_05556#249].
The self-healing capabilities can be achieved through various mechanisms, such as autocatalytic reactions, microbial activity, or thermally activated transformations [arxiv/2502_13025#60]. For instance, the dynamic nature of collagen and cell signaling in some biomimetic materials can lead to self-healing properties, where cells can secrete enzymes to repair damaged regions [arxiv/2409_05556#411].
To assess these properties, researchers can induce controlled damage, such as micro-cracks, and then monitor the recovery of mechanical properties over time using techniques like tensile testing and microscopy [arxiv/2409_05556#294, arxiv/2409_05556#321, arxiv/2409_05556#411]. The goal is often to achieve a significant recovery rate, for example, at least 70% of the original mechanical properties [arxiv/2409_05556#294, arxiv/2409_05556#411].
The integration of self-healing materials, particularly those with lamellar structures, holds significant potential for improving the durability and lifespan of various applications, including microfluidic chips [arxiv/2409_05556#294, arxiv/2409_05556#274]. This approach aligns with the broader concept of designing more resilient and sustainable structures that can adapt to environmental stresses [arxiv/2502_13025#60].

**X3**

Biomimetic materials with a lamellar structure can be engineered to exhibit self-healing properties under mechanical stress. The layered structure can facilitate the redistribution of stress and the healing of minor cracks, mimicking natural biological systems [arxiv/2409_05556#274]. This self-healing capability is an innovative feature that can extend the lifespan of materials, particularly in applications where they are subjected to mechanical stress and potential damage [arxiv/2409_05556#309].
The self-healing mechanism can be achieved through various approaches. For instance, the hierarchical organization of a material can allow for the redistribution of stress and the reformation of broken bonds at the molecular level, potentially facilitated by dynamic covalent bonds or reversible physical interactions [arxiv/2409_05556#194]. In other cases, the organic matrix or polymeric binder can flow and fill in cracks, driven by capillary action or thermal activation [arxiv/2409_05556#675]. Nanocomposites, such as graphene oxide or carbon nanotubes, can also introduce self-healing properties by forming reversible bonds with the matrix [arxiv/2409_05556#542].
Quantitative goals for self-healing efficiency often aim for a recovery rate of at least 70% of the original mechanical properties after damage [arxiv/2409_05556#294]. Experimental validation involves inducing controlled damage, such as micro-cracks, and monitoring the recovery of mechanical properties over time using techniques like scanning electron microscopy and atomic force microscopy to observe the healing process at the microstructural level [arxiv/2409_05556#294].

**X4**

Yes, biomimetic materials with a lamellar structure can be engineered to exhibit self-healing properties under mechanical stress. The question of whether such materials can be engineered for self-healing under mechanical stress is a key scientific inquiry [arxiv/2409_05556#320].
Self-healing materials are designed to repair damage autonomously when triggered by external factors like mechanical stress [arxiv/2502_13025#65]. This self-healing capability can be achieved through various mechanisms, including the use of microbial fermentation to produce bio-based polymers with self-healing properties [arxiv/2502_13025#106]. Additionally, biomimetic techniques can be employed to synthesize proteins like silk or spider silk and integrate them into polymer matrices, potentially leading to adaptive properties [arxiv/2502_13025#108].
Biomimetic materials often possess a lamellar structure, which contributes to their mechanical properties and functionality [arxiv/2409_05556#249]. This lamellar structure is observed in natural materials such as keratin scales, which exhibit layered compositions that provide strength and resilience [arxiv/2409_05556#249]. The mechanical properties of these materials can be further enhanced through various strategies, including adjusting nanocomposite concentrations [arxiv/2409_05556#476].

---

## Item 4

**Query:** We gratefully acknowledge support from the China Scholarship Council (CSC) and the German Research Foundation (DFG) under the project Crossmodal Learning (TRR 169).

**X1**

The research received support from the China Scholarship Council (CSC) and the German Research Foundation (DFG) as part of the project Crossmodal Learning (TRR 169) [arxiv/2309_13339#25]. This acknowledgment is found within the "Acknowledgements" section of the relevant document [arxiv/2309_13339#25].
Other research projects also acknowledge support from various organizations. For instance, the National Natural Science Foundation of China, Zhejiang Provincial Natural Science Foundation of China, and the Ministry of Education, Singapore, have provided funding for some projects [arxiv/2405_17969#16, arxiv/2405_14205#18, arxiv/2410_07869#21]. Additionally, support has been received from the Australian Government Research Training Program (RTP) Scholarship, the National Health and Medical Research Council (NHMRC) of Australia, and the Australian Research Council (ARC) [arxiv/2310_07984#32, arxiv/2306_08302#62, arxiv/2402_02713#22].
Further acknowledgments include support from the Natural Sciences and Engineering Research Council of Canada (NSERC), the European Research Council (ERC), and the National Science Foundation (NSF) [arxiv/1412_3328#25, arxiv/2406_09787#14, arxiv/2005_09007#21, arxiv/2307_08197#16, arxiv/2402_02713#22, arxiv/2404_07198#15, arxiv/2509_26626#15]. The China Scholarship Council (CSC) also provided support for an international scholarship [arxiv/2503_1015#18].
Funding has also been provided by the National Key Research and Development Program of China, the Basic Research Program of Jiangsu Province, and various university-specific funds [arxiv/2503_06512#12]. Other sources of support include grants from the NIH BRAIN Initiative, the NSF/NIH CRCNS Program, and fellowships from organizations like the Simons Collaboration on the Global Brain [arxiv/2508_18413#21].

**X2**

Support for this research was gratefully acknowledged from the China Scholarship Council (CSC) and the German Research Foundation (DFG) [arxiv/2309_13339#25]. This funding was provided specifically under the project titled "Crossmodal Learning (TRR 169)" [arxiv/2309_13339#25].
Additional support for research endeavors is also acknowledged from various national and international organizations. These include the National Natural Science Foundation of China, the Fundamental Research Funds for the Central Universities, and the Zhejiang Provincial Natural Science Foundation of China [arxiv/2405_17969#16]. Further contributions came from the Yongjiang Talent Introduction Programme, the CIPSC-SMP-Zhipu Large Model Cross-Disciplinary Fund, and Ningbo Science and Technology Special Projects [arxiv/2405_17969#16].
Other acknowledgements highlight support from the Ministry of Education, Singapore, and the Zhejiang University Education Foundation [arxiv/2405_17969#16]. In a separate context, support is also noted from the Australian Government Research Training Program (RTP) Scholarship, Monash University, and the National Health and Medical Research Council (NHMRC) of Australia [arxiv/2310_07984#32]. Computational resources were also provided by the Nectar Research Cloud and the MASSIVE HPC facility [arxiv/2310_07984#32].

**X3**

The China Scholarship Council (CSC) and the German Research Foundation (DFG) provided support for the project "Crossmodal Learning (TRR 169)". [arxiv/2309_13339#25]
Additionally, the China Scholarship Council (CSC) has supported Xiang Li through an International Cambridge Scholarship. [arxiv/2503_1015#18] The German Research Foundation (DFG) has also provided support for the NLLG Lab at UTN via a Heisenberg Grant. [arxiv/2502_05151#11]

**X4**

Support for this project was gratefully acknowledged from the China Scholarship Council (CSC) and the German Research Foundation (DFG) under the project Crossmodal Learning (TRR 169) [arxiv/2309_13339#25].

---

## Item 5

**Query:** partitions images into multiple regions during tokenization, thereby improving the grasp of the visual information of the model.

**X1**

Images are partitioned into multiple regions during tokenization to improve a model's understanding of visual information.
Several strategies are employed for image partitioning. Another method involves dividing an HR image into multiple high-resolution images, as seen in the strategy used by [arxiv/2410_05993#5]. Some models also employ dynamic patching, where high-resolution images are divided into smaller tiles based on their aspect ratio and resolution, allowing for a higher degree of magnification and observation of finer details [arxiv/2411_07688#2]. This dynamic resolution approach can be seen as an enhancement of input image resolution, with the performance generally improving as the number of dynamic tiles increases [arxiv/2411_07688#2].

**X2**

To improve the grasp of visual information, models partition images into multiple regions during tokenization. This approach is particularly useful for handling high-resolution images, where a direct tokenization of the entire image would be computationally prohibitive. By dividing an image into smaller segments or patches, models can process these regions more efficiently, allowing for a more detailed understanding of the visual content. [arxiv/2405_10739#7, arxiv/2409_11402#9]
One method involves splitting an image into smaller, variable-sized slices or patches. These patches are then processed, and in some cases, further compressed to reduce the number of visual tokens. This strategy allows models to utilize low-resolution vision encoders while still perceiving detailed information from high-resolution images. [arxiv/2405_10739#7] Another technique is dynamic resolution, where images are divided into tiles based on their aspect ratio and resolution. Each tile is then processed, leading to a variable number of visual tokens that are packed into a single sequence for the language model. [arxiv/2409_12191#4, arxiv/2411_07688#2]
Furthermore, some approaches focus on adaptively selecting important visual tokens from the numerous tokens generated by the vision encoder. This is achieved by analyzing the similarity between tokens or using scoring functions to prune less informative ones. The selected tokens are then merged and processed, further reducing computational costs while aiming to maintain performance. [arxiv/2403_15388#2] These methods collectively aim to enhance the model's ability to perceive and comprehend visual information, especially in scenarios with high-resolution or complex imagery. [arxiv/2405_10739#7, arxiv/2403_15388#2]

**X3**

Images are partitioned into multiple regions during tokenization to enhance a model's comprehension of visual information. This process is a key technique in multimodal large language models (MLLMs) for handling high-resolution images efficiently. By dividing an image into smaller segments or "tiles," models can process these regions individually, which is particularly useful when dealing with images that contain fine details or small objects that might be lost if the entire image were downscaled. This approach helps to mitigate the computational burden associated with processing large numbers of visual tokens, a common challenge in Transformer-based architectures.
Several strategies exist for partitioning images into regions for tokenization. One common method involves splitting an image into smaller, variable-sized slices or patches. For instance, LLaVA-UHD employs an image modularization strategy that divides native-resolution images into smaller slices for efficient encoding [arxiv/2405_10739#7]. Another approach is dynamic resolution, where images are divided into tiles based on their aspect ratio and resolution, allowing for a higher degree of magnification and the observation of finer details. This is seen in techniques like the Dynamic Resolution (DR) approach used with InternVL2.5, which divides images into tiles to enable the observation of finer details [arxiv/2411_07688#2].
These partitioning methods aim to improve the model's ability to grasp visual information by allowing it to focus on specific parts of an image. For example, by splitting a high-resolution image into tiles, models can better detect and analyze small objects that might otherwise be overlooked. This is crucial for tasks requiring intricate recognition, such as optical character recognition (OCR) of small characters or analyzing remote sensing imagery with tiny targets [arxiv/2405_10739#7, arxiv/2411_07688#2]. The resulting tokens from these partitioned regions are then processed by the language model, leading to a more nuanced understanding of the visual content.

**X4**

To improve the grasp of visual information, models partition images into multiple regions during tokenization. This approach allows for more detailed analysis and processing of visual data, especially for intricate recognition tasks. By breaking down an image into smaller parts, models can better capture fine-grained details that might be lost if the entire image were processed as a single unit.
One method for achieving this is through image modularization, where native-resolution images are divided into smaller, variable-sized slices for efficient encoding [arxiv/2405_10739#7]. Another strategy involves dynamically adjusting resolution with automatic layout arrangements, which adaptively alters patch layouts and counts to enhance information extraction efficiency [arxiv/2405_10739#7]. Some models also employ a multi-view input strategy, creating a global view (low-resolution) and local views (image patches) to enable perception of detailed information using low-resolution encoders [arxiv/2405_10739#7].
Furthermore, techniques like dynamic resolution involve dividing images into tiles, where an increased number of tiles allows for a higher degree of image magnification, thus enabling the observation of finer details [arxiv/2411_07688#2]. This partitioning is crucial for tasks requiring the analysis of small targets in high-resolution images, where models might otherwise struggle to identify these features due to limited input resolutions [arxiv/2411_07688#2].
These methods collectively aim to reduce computational burden while preserving or enhancing the model's ability to understand visual information. By processing images in partitioned regions, models can achieve a more comprehensive grasp of the visual content, leading to improved performance in various tasks.

---

## Item 6

**Query:** Leveraging the MPT-7B [473] and CLIP ViT-L/14 [450], Octopus integrates egocentric and bird's-eye views for visual comprehension, generating executable action code.

**X1**

Octopus integrates egocentric and bird's-eye views for visual comprehension by leveraging the MPT-7B and CLIP ViT-L/14 models to generate executable action code. This approach combines different visual perspectives to enhance understanding and enable action generation.
The system utilizes MPT-7B, a large language model, and CLIP ViT-L/14, a vision-language model, to achieve its visual comprehension capabilities. By integrating egocentric (first-person) and bird's-eye (overhead) views, Octopus can process and understand visual information from multiple angles. This multi-perspective approach is key to its ability to generate executable action code.

**X2**

The system utilizes specialized tokens and encoders, such as ViT, to represent various modalities, including images and actions, within a unified framework. This integration of different visual perspectives is key to its advanced visual comprehension capabilities.
The use of MPT-7B and CLIP ViT-L/14 signifies a reliance on powerful, pre-trained models that have demonstrated strong performance in their respective domains, enabling Octopus to achieve sophisticated visual understanding and action generation.

**X3**

Octopus is a system that integrates egocentric and bird's-eye views for visual comprehension, enabling it to generate executable action code. It leverages the MPT-7B language model and the CLIP ViT-L/14 vision model for its visual understanding capabilities [arxiv/2411_18279#82].
The system is trained using the OctoVerse suite, which includes datasets with richly annotated environments. This training allows Octopus to process visual information from multiple perspectives and translate it into actionable code [arxiv/2411_18279#82].

**X4**

Octopus integrates egocentric and bird's-eye views for visual comprehension by leveraging pre-trained models like MPT-7B and CLIP ViT-L/14 [arxiv/2311_05437#23]. This integration allows for the generation of executable action code. The framework represents various modalities, including images and actions, using specialized tokens and encoders such as ViT. This approach contributes to the development of AI agents capable of perceiving environments, making decisions, and executing actions. Large language models further enhance these agents' ability to interact with GUIs and generate tasks for self-improvement [arxiv/2311_05437#23].

---

## Item 7

**Query:** - · How can LLM GP integrate software engineering domain knowledge?

**X1**

LLM GP can integrate software engineering domain knowledge by leveraging techniques like Retrieval-Augmented Generation (RAG) and by enhancing multi-step reasoning capabilities.
These integrations are often realized through LLM-based agents, which act as wrappers around LLMs and interact with software tools. These agents can autonomously invoke tools like code editors, compilers, and testing frameworks to achieve specific software engineering goals, such as code generation, debugging, and program repair [arxiv/2502_13767#4, arxiv/2408_02479#1].
Furthermore, LLM GP can integrate domain knowledge by employing multi-agent systems where specialized agents collaborate on different aspects of software development, mimicking human development teams. This division of labor allows for more efficient and accurate handling of complex tasks, such as requirement analysis, code generation, testing, and debugging, leading to improved software development practices and reduced human error [arxiv/2408_02479#22, arxiv/2408_02479#35].

**X2**

LLM GP can integrate software engineering domain knowledge by leveraging its capabilities in code generation, completion, and debugging. This integration is facilitated through techniques like retrieval-augmented generation (RAG) and by enhancing multi-step reasoning. RAFT, for example, adapts language models for domain-specific RAG, emphasizing domain adaptation as a crucial aspect of this integration [arxiv/2508_00083#26].
The integration involves developing specialized tools and frameworks, such as CodeTF, to support these software engineering tasks. LLM agents, acting as wrappers around LLMs, can interact with various software tools, including testing and analysis utilities, to achieve tasks autonomously. This agentic capability allows LLMs to mimic human software engineers by invoking tools like shells, code editors, and web browsers to address issues like bug fixes and feature additions [arxiv/2502_13767#4].
Furthermore, LLM GP can be enhanced through automated workflow optimization, where LLM-based optimizers generate or modify workflow nodes. These candidate workflows are then evaluated, and the performance metrics are backpropagated to update scores and historical experience. This process allows for the joint optimization of workflow structures and operator representations, leading to more compact and task-specific solutions [arxiv/2511_20693#9].
The integration also extends to improving reasoning capabilities by enhancing multi-step reasoning. This aims for more efficient and accurate complex reasoning by scaling structured knowledge. LLM GP can be prompted to recombine or mutate code snippets, with the prompt formulation being the primary control mechanism. This approach allows LLMs to potentially generate better adaptations than traditional GP operators, leading to problem-solving with fewer fitness evaluations [arxiv/2401_07102#33].

**X3**

These agents can utilize program analysis techniques and work with program representations like abstract syntax trees, rather than just treating a project as a collection of files [arxiv/2404_05427#4].
Furthermore, LLM GPs can be integrated into automated workflow optimization mechanisms. These systems can use LLMs to generate or modify nodes within a workflow, leveraging self-adaptive abstract operator sets that can dynamically generalize and merge execution steps [arxiv/2511_20693#9]. This allows for the joint optimization of workflow structures and operator representations, leading to more compact and task-specific solutions.
Evaluation benchmarks play a crucial role in assessing the integration of LLMs into software engineering. These range from method/class-level code generation tasks to benchmarks simulating real software development scenarios, such as SWE-Bench, which contains real GitHub issues for defect repair and feature development [arxiv/2508_00083#26]. These benchmarks provide environments where agents interact with codebases, command-line interfaces, and debugging tools, mimicking human developer workflows [arxiv/2508_00083#26].

**X4**

LLM GP can integrate software engineering domain knowledge by leveraging LLMs as powerful pattern-matching and sequence completion systems within an evolutionary algorithm framework. Instead of relying on explicit structural manipulation of code, LLM GP tasks LLMs with recombining or mutating code snippets through carefully formulated prompts. This allows the LLM to act as a black box that, when prompted, can generate code adaptations that are potentially better than those produced by traditional genetic programming operators, leading to more efficient problem-solving [arxiv/2401_07102#33].
The integration is further enhanced by the LLM's ability to process and respond to prompts that can include natural language criteria for selection, moving beyond purely numerical fitness evaluations. This means that LLM GP can incorporate more nuanced, domain-specific knowledge into its evolutionary process, as the LLM can be guided by prompts that reflect software engineering principles or desired outcomes [arxiv/2401_07102#33].
Moreover, LLM-based agents, which build upon LLMs, are increasingly being used to integrate software engineering domain knowledge. These agents can interact with software tools, such as testing and analysis tools, and can operate with varying degrees of autonomy. This agentic capability allows them to mimic human software engineers by performing tasks like bug fixes, feature additions, and even navigating codebases, thereby embedding practical software engineering workflows into their operations [arxiv/2502_13767#4].
These LLM-based agents can also be designed with specific roles and collaborate in multi-agent systems to simulate the software development lifecycle, from requirement engineering to testing and maintenance. This collaborative approach allows for a more comprehensive integration of domain knowledge by breaking down complex tasks and assigning them to specialized agents, enhancing efficiency and code quality [arxiv/2408_02479#22, arxiv/2408_02479#35, arxiv/2308_11432#15].

---

## Item 8

**Query:** All results are reported to four decimal places.

**X1**

All results are reported to four decimal places. [arxiv/2511_08151#62]

**X2**

All results are reported to four decimal places. [arxiv/2511_08151#62] This is achieved by calculating the natural logarithm of the total probability and then rounding the result to four decimal places. [arxiv/2511_08151#62] For example, a final log probability of -6.1861 is obtained after this process. [arxiv/2511_08151#62]
When comparing numerical values, rounding to a specified precision is also important. For instance, in some contexts, both reference and predicted values are rounded using the "round-half-up" method to a configured precision, after which their canonical decimal representations must be identical. [arxiv/2608_03451#41] In other cases, rounding to two or five decimal places has been used to validate experimental conclusions, with the performance of models remaining strong even after adjusting the rounding precision. [arxiv/2404_07544#50]

**X3**

All results are reported to four decimal places. [arxiv/2511_08151#62]

**X4**

All results are reported to four decimal places. [arxiv/2511_08151#62]

---

## Item 9

**Query:** It facilitates the retention and retrieval of information across sessions, ensuring continuity and enhancing the model's capacity to handle long contexts without exceeding its input limitations.

**X1**

The question asks about a mechanism that facilitates information retention and retrieval across sessions, ensuring continuity and enhancing long-context handling without exceeding input limitations.
Retrieval-Augmented Generation (RAG) systems are designed to address this by integrating relevant information from external databases into their responses [arxiv/2502_01549#15]. This process involves indexing data, retrieving pertinent information based on user input, and then using this retrieved knowledge to generate informed responses [arxiv/2502_01549#15]. Advancements in RAG include chunk-based approaches that optimize text segmentation and retrieval using vector space embeddings, and graph-based methods that leverage structured knowledge representations for improved efficiency and precision [arxiv/2502_01549#15].
These systems aim to enhance a model's capacity to handle long contexts by grounding responses in factual, domain-specific knowledge [arxiv/2502_01549#15]. For instance, datasets like MuSiQue (MSQ) and MultiFieldQA-en (MFQA) are specifically curated to test a model's long-context understanding ability across diverse fields, with evidence often located randomly within documents [arxiv/2310_03025#10]. The use of longer retrieval units, such as grouped documents containing an average of 6K tokens, can significantly compress the corpus size and improve retrieval performance, thereby alleviating the burden on the retriever and enabling comparable results with fewer retrieval units [arxiv/2406_15319#8].
While RAG systems generally improve performance, challenges remain. Some methods may perform retrieval for most queries, leading to drops in consistency if incorrect retrieval timing or unnecessary external information is introduced [arxiv/2410_13339#14]. Efficient adaptive retrieval methods are crucial for optimizing resource utilization while maintaining performance, especially when faced with insufficient retrieval results or when confidence-based approaches show inconsistent behavior across datasets [arxiv/2502_01142#30, arxiv/2502_01142#18]. Furthermore, robust safeguards are important to address the detrimental impact of imperfect retrieval results, with some methods demonstrating effectiveness in overcoming noisy or negative retrieval augmentation [arxiv/2410_07176#14].

**X2**

The question asks about a mechanism that facilitates information retention and retrieval across sessions, ensuring continuity and enhancing the model's capacity for long contexts without exceeding input limitations.
The provided context does not directly address a mechanism that facilitates information retention and retrieval *across sessions* in the way a user might understand it (e.g., saving and loading a conversation history). However, it does discuss approaches that enhance a model's ability to handle long contexts by improving retrieval and reducing the burden on the retriever.
One approach involves using longer retrieval units, such as grouped documents, instead of hundreds of short passages. This method compresses the corpus size and requires significantly fewer retrieval units to achieve comparable or improved retrieval performance, thereby alleviating the burden on the retriever and potentially allowing for the processing of more information within input limitations [arxiv/2406_15319#8]. Another strategy for managing long contexts is recursive retrieval, where the retrieval process can call itself to create a hierarchy of retrievals, breaking down complex queries into simpler sub-queries. This is particularly useful for handling long texts by constructing a memory tree of summaries and navigating through it to retrieve relevant information [arxiv/2404_10981#15]. Additionally, a dynamic and efficient retrieval strategy can minimize unnecessary operations by prioritizing retrieval from a cache and utilizing a missing table to bypass retrieval for previously unsuccessful queries, thus improving efficiency [arxiv/2502_17139#11].

**X3**

AI models can retain and retrieve information across sessions by employing long-term memory capabilities. Frameworks like NapMem and Metis integrate memory natively into AI models, enhancing their capacity to handle long contexts. Techniques for optimizing LLM inference, such as efficient KV cache utilization, compression, and reuse, also facilitate this continuity and improve the model's ability to manage extended contexts without exceeding input limitations.
These systems achieve continuity by storing and recalling past information, enabling them to utilize it in subsequent interactions. This is crucial for maintaining context and coherence over extended periods, effectively extending the model's capacity beyond its immediate input limitations.
Methods for optimizing LLM inference contribute to this by managing the key-value (KV) cache, which stores intermediate states. Techniques like KV cache compression and reuse reduce memory usage and computational costs, allowing models to process longer sequences and retain information more effectively across sessions. This ensures that the model can handle complex, long-context tasks without performance degradation.

**X4**

AI agents with long-term memory capabilities can store, recall, and utilize past information, facilitating the retention and retrieval of information across sessions. This continuity enhances a model's capacity to handle long contexts without exceeding its input limitations. Frameworks like NapMem and Metis integrate memory natively into AI models, improving their ability to manage extended contexts. [arxiv/2502_01549#15]
Techniques for optimizing Large Language Model (LLM) inference also contribute to this capability. These include efficient Key-Value (KV) cache utilization, compression, and reuse. [arxiv/2502_01142#18, arxiv/2502_01142#30] Retrieval-Augmented Generation (RAG) systems, by integrating relevant information from external databases, ground responses in factual knowledge, thereby aiding in information retention and retrieval across sessions. [arxiv/2502_01549#15]
Chunk-based and graph-based approaches are two methodological trajectories within RAG that optimize text segmentation and retrieval. [arxiv/2502_01549#15] Furthermore, the use of longer retrieval units, such as grouped documents, can significantly compress the corpus size and reduce the number of retrieval units needed, while achieving comparable or improved retrieval performance. This alleviates the burden on the retriever and supports handling longer contexts. [arxiv/2406_15319#8]

---

## Item 10

**Query:** A notable reasoning approach involves breaking down complex questions into a series of intermediate steps, a technique known as the Chain-of-Thought (CoT) method (Wei et al., 2022; Kojima et al., 2022).

**X1**

A notable reasoning approach involves breaking down complex questions into a series of intermediate steps, a technique known as the Chain-of-Thought (CoT) method (Wei et al., 2022; Kojima et al., 2022). This method enhances the reasoning capabilities of large language models (LLMs) by prompting them to generate explicit intermediate reasoning steps before arriving at a final answer [arxiv/2405_14838#1, arxiv/2505_15778#2, arxiv/2305_04091#2]. CoT prompting has been shown to significantly boost performance on a wide range of tasks, including arithmetic, commonsense, and symbolic reasoning [arxiv/2304_08354#16].
Various CoT techniques exist, including few-shot and zero-shot prompting. Few-shot CoT provides manually crafted step-by-step reasoning demonstrations within the prompt [arxiv/2305_04091#2]. Zero-shot CoT, on the other hand, simplifies this by simply appending a phrase like "Let's think step by step" to the problem statement, eliminating the need for examples [arxiv/2304_08354#16, arxiv/2503_05641#31, arxiv/2305_04091#2]. Other related methods include Plan-and-Solve (PS) prompting, which first devises a plan to divide the task into subtasks and then carries out the plan [arxiv/2305_04091#2, arxiv/2401_14295#73].
Beyond these foundational techniques, extensions like Tree-of-Thoughts (ToT) and Graph-of-Thoughts (GoT) generalize the reasoning structure from a linear chain to a tree or graph, allowing for more exploration and backtracking [arxiv/2304_08354#16, arxiv/2511_07124#3]. There is also a growing interest in implicit reasoning, where models internalize the reasoning process within their hidden states rather than generating explicit tokens, aiming for increased efficiency [arxiv/2405_14838#1, arxiv/2411_15862#1].

**X2**

A notable reasoning approach involves breaking down complex questions into a series of intermediate steps, a technique known as the Chain-of-Thought (CoT) method (Wei et al., 2022; Kojima et al., 2022). This method enhances the reasoning capabilities and interpretability of large language models (LLMs) by prompting them to generate intermediate reasoning traces before arriving at a final answer. CoT has been shown to significantly boost performance on various tasks, including arithmetic, commonsense, and symbolic reasoning [arxiv/2304_08354#16].
CoT prompting is a technique that involves providing models with prompts that include reasoning steps for solving complex problems. These steps serve as examples to guide LLMs in planning and acting in a step-by-step manner [arxiv/2308_11432#7]. Zero-shot CoT further enables LLMs to generate reasoning processes by simply prompting them with phrases like "think step by step," without requiring explicit examples in the prompt [arxiv/2308_11432#7].
Beyond the basic CoT, variations like Self-consistent CoT (CoT-SC) and Tree of Thoughts (ToT) have emerged. CoT-SC generates multiple reasoning paths and selects the answer with the highest frequency, acknowledging that complex problems can have multiple solutions [arxiv/2308_11432#7]. ToT structures reasoning into a tree-like format where each node represents an intermediate reasoning step, with selections guided by LLM evaluations and search strategies like BFS or DFS [arxiv/2308_11432#7].

**X3**

A notable reasoning approach involves breaking down complex questions into a series of intermediate steps, a technique known as the Chain-of-Thought (CoT) method. This method enhances the performance of large language models (LLMs) on various tasks by prompting them to generate their "thoughts" on the necessary intermediate steps before arriving at a final answer [arxiv/2304_08354#16]. CoT has been shown to significantly boost performance in areas such as arithmetic, commonsense, and symbolic reasoning [arxiv/2304_08354#16].
This approach is a form of "single-path reasoning," where a task is decomposed into intermediate steps connected in a cascading manner, with each step leading to only one subsequent step [arxiv/2308_11432#7]. The CoT method specifically involves inputting reasoning steps for solving complex problems into the prompt, serving as examples to guide LLMs to plan and act in a step-by-step fashion [arxiv/2308_11432#7]. An extension, Zero-shot-CoT, enables LLMs to generate reasoning processes by prompting them with phrases like "think step by step," without needing explicit examples in the prompt [arxiv/2308_11432#7].
The CoT method is a key component in the broader field of planning and reasoning within LLM-based agents. It contributes to the agent's ability to decompose complex problems into sub-problems and solve them sequentially [arxiv/2304_08354#16]. This step-by-step problem-solving capability is crucial for agents to behave more reasonably, powerfully, and reliably when faced with intricate tasks [arxiv/2308_11432#7]. Furthermore, CoT has been extended to multimodal reasoning, allowing LLMs to reason with multiple modalities of information by breaking down complex problems into logical steps [arxiv/2509_11944#6].

**X4**

A notable reasoning approach involves breaking down complex questions into a series of intermediate steps, a technique known as the Chain-of-Thought (CoT) method. This method enhances the performance of foundation models, particularly in zero-shot and few-shot scenarios, by prompting them to generate intermediate reasoning traces before arriving at a final answer. CoT has demonstrated significant improvements across various tasks, including arithmetic, commonsense, and symbolic reasoning [arxiv/2304_08354#16].
CoT prompting is an extension of vanilla few-shot prompt learning, which provides models with examples for a given task. In contrast, CoT includes the reasoning trace required to derive the answer for each example within the prompt. This encourages models to articulate their "thoughts" on the necessary intermediate steps. This approach has been extended to multimodal reasoning, termed Multimodal Chain-of-Thought (MCoT), which integrates and reasons across multiple modalities like text and images [arxiv/2509_11944#6].
Further advancements have explored variations and extensions of CoT. For instance, techniques like Tree-of-Thoughts and Graph-of-Thoughts expand the deliberation space by structuring reasoning processes beyond a linear chain [arxiv/2511_02303#2]. Other methods, such as Least-To-Most, deconstruct problems into sub-questions, addressing them sequentially with the answer to one serving as input for the next [arxiv/2409_17539#14]. Domain-specific CoT cues have also been developed, for example, in medical contexts, to guide step-by-step reasoning [arxiv/2207_08143#21].

---

## Item 11

**Query:** Table 3 presents the results across 8 tasks.

**X1**

Table 3 presents the results of various models across eight distinct tasks. These tasks encompass a range of evaluations, including performance metrics such as latency, scores, and success rates. The results detail evaluations across different scenarios and datasets, encompassing tasks like MMLU, HumanEval, GSM8K, BoolQ, HotpotQA, and Natural Language Inference. The tables also compare LLM performance on Japanese language and math tasks, as well as their inference costs. Additionally, they include comparisons of LLM responses to specific questions across domains like chemistry, physics, and computer science, and present performance comparisons for GPU acceleration of LLM inference and speech recognition. Datasets used for finetuning, such as RTE, CoLA, SST-2, MRPC, and STSB, are also detailed. The tables also list questions used in chatbot evaluations and compare features of different language agent frameworks.
The results presented in Table 3 cover a broad spectrum of tasks, including those related to natural language inference, question answering, and mathematical reasoning. For instance, the GLUE benchmark, which includes tasks like MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, and RTE, is frequently mentioned in the context of evaluating model performance [arxiv/2402_12354#9, arxiv/2311_13171#13, arxiv/2404_02948#33, arxiv/2310_08659#13, arxiv/2512_07832#5, arxiv/1810_04805#31, arxiv/2403_03507#40].

**X2**

Table 3 presents the results across 8 tasks. This table is mentioned as a resource for formatted instances in the context of instruction tuning for large language models [arxiv/2303_18223#30]. These instances are typically formatted as instruction-input-output triplets, where the instruction is a natural language task description.
The use of such formatted instances is crucial for training large language models to generalize to new tasks. By providing explicit task instructions, models can learn to understand and perform a variety of tasks, even those not explicitly seen during training [arxiv/2309_10020#59]. This approach contrasts with traditional Natural Language Processing (NLP) methods where task instructions were often implicit, making generalization to unseen tasks more difficult [arxiv/2309_10020#59].
The data presented in Table 3 likely contributes to the development of models that can follow instructions, a key aspect of modern large language model capabilities. This is particularly relevant for tasks requiring step-by-step reasoning, such as those involving algebraic calculations or complex problem-solving, where Chain of Thought (CoT) prompting has shown significant benefits [arxiv/2501_09223#95].

**X3**

Table 3 presents the results across 8 tasks.
The provided context does not contain information about "Table 3" or the results across 8 tasks. Therefore, it is not possible to answer the question based on the given information.

**X4**

Table 3 presents the results of various models across eight distinct tasks, encompassing performance metrics such as latency, scores, and success rates [arxiv/2304_08354#168]. These tasks include evaluations on benchmarks like MMLU, HumanEval, GSM8K, BoolQ, HotpotQA, and Natural Language Inference [arxiv/2304_08354#168].
The table also details LLM performance on Japanese language and math tasks, alongside their inference costs [arxiv/2304_08354#168]. Furthermore, it compares LLM responses to specific questions in domains such as chemistry, physics, and computer science, and presents performance comparisons for GPU acceleration of LLM inference and speech recognition [arxiv/2304_08354#168]. Datasets used for fine-tuning, including RTE, CoLA, SST-2, MRPC, and STSB, are also listed [arxiv/2304_08354#168].
Additionally, the tables include questions used in chatbot evaluations and compare features of different language agent frameworks [arxiv/2304_08354#168]. The context also mentions that large numbers of labeled data formatted in natural language are available for instruction tuning, with a list of these resources provided in Table 3 [arxiv/2303_18223#30].

---

## Item 12

**Query:** **Decision Point**: Are you certain that no further verification or information gathering is needed to provide the final answer?

**X1**

Yes, I am certain that no further verification or information gathering is needed to provide the final answer. The provided context directly addresses the question of whether further verification or information gathering is required. Specifically, sections discuss the evaluation of LLM responses, including criteria for determining if an answer is correct, incorrect, or uncertain, and whether a model has enough information to answer a query [arxiv/2607_13920#44, arxiv/2501_14249#40]. These sections detail prompts and schemas designed to judge the sufficiency of information for providing a final answer, indicating that the determination of whether further verification is needed is a core aspect of the evaluation process described.

**X2**

No, I am not certain that no further verification or information gathering is needed to provide the final answer. The provided context does not contain information that directly addresses the certainty of needing further verification. The sections retrieved discuss methods for evaluating model answers and generating responses, including structured output schemas for evaluation [arxiv/2607_13920#44] and prompts for generating explanations and answers [arxiv/2411_15124#103], as well as templates for judging responses against correct answers [arxiv/2501_14249#40, arxiv/2510_08790#23]. However, none of these sections speak to the decision point of whether *further* verification is needed.

**X3**

Yes, I am certain that no further verification or information gathering is needed to provide the final answer. The provided context directly addresses the question of whether further verification or information gathering is needed, with the expected answer being "yes" or "no." [arxiv/2501_14249#40] It also includes prompts that require a "Yes" or "No" response to indicate if enough information is present to answer a main query. [arxiv/2501_14249#40]

**X4**

No, further verification or information gathering is needed to provide the final answer. The provided context details specific prompts and evaluation methodologies for LLMs, including structured output formats for confidence scores and answer extraction [arxiv/2501_14249#40]. It also outlines processes for judging model answers against correct answers, which involves extracting final answers, reasoning, correctness, and confidence [arxiv/2501_14249#40]. Furthermore, the context describes a "Chain-of-Verification" prompting strategy that involves planning and executing verifications to improve response accuracy [arxiv/2409_04109#42]. These elements collectively indicate that the system is designed to gather and verify information to a sufficient degree for providing a final answer.

---

## Item 13

**Query:** Adversarial attacks on AI systems have evolved from simple perturbations fooling image classifiers to advanced prompt injections manipulating language models.

**X1**

Adversarial attacks on AI systems have evolved significantly, moving from simple image manipulations to sophisticated methods targeting language models. These advanced attacks, often termed "prompt injections," involve embedding malicious instructions within the input prompt to hijack the model's intended functionality and redirect it to perform actions desired by the attacker [arxiv/2510_24801#28, arxiv/2504_01990#195]. This contrasts with earlier attacks that focused on altering input data, such as adding noise or perturbations to images [arxiv/2510_24801#28].
Prompt injection attacks exploit the inherent difficulty language models have in distinguishing between original context and externally added instructions [arxiv/2504_01990#195]. This vulnerability is amplified by the open-ended nature of text input and the assumption that all input is trustworthy [arxiv/2504_01990#195]. Even minor, malicious modifications to prompts can lead to substantial deviations in the model's generated output [arxiv/2504_01990#195].
These attacks can be categorized based on how the adversarial instructions are introduced. Direct prompt injection involves explicitly altering the input prompt, while indirect prompt injection leverages external content to influence the model's behavior [arxiv/2504_01990#195]. Furthermore, attacks can target various aspects of AI systems, including watermarking techniques, where methods like text insertion, deletion, and substitution are employed to evade detection [arxiv/2301_10226#18]. These attacks can range from simple paraphrasing to more complex generative approaches that prompt the model to alter its output in predictable ways [arxiv/2301_10226#18].

**X2**

Adversarial attacks on AI systems have evolved significantly, moving from simple perturbations that fool image classifiers to more sophisticated prompt injections that manipulate language models [arxiv/2504_01990#203, arxiv/2510_24801#28]. These attacks can manifest across different modalities, including text, images, and audio [arxiv/2504_01990#207].
Research in this area includes methods to defend Large Language Models (LLMs) against such adversarial attacks and jailbreaks [arxiv/2504_01990#203]. Defensive strategies being explored encompass constitutional classifiers, adversarial training, and data augmentation [arxiv/2504_01990#203]. For instance, adversarial training, initially presented as a jailbreak mitigation technique, improves inherent robustness through continuous exposure to adversarial examples [arxiv/2504_01990#203].
In the context of LLMs, adversarial attacks and jailbreaks share underlying mechanisms of crafting inputs to induce undesired or harmful outputs [arxiv/2504_01990#203]. While traditional adversarial attacks often focus on minimal perturbations, jailbreaks can drastically alter or extend prompts without strict limits on the scale of the perturbation [arxiv/2504_01990#203]. As LLMs and their safety constraints become more integrated, these attack paradigms may merge, necessitating unified defense strategies against any maliciously crafted input [arxiv/2504_01990#203].
Beyond LLMs, adversarial attacks also target other AI systems. For example, in text-to-image diffusion models, attackers can inject text backdoors to generate specific target images, aiming for triggers that are invisible and maintain text fluency [arxiv/2508_1605#10]. Watermarking techniques in LLMs are also subject to attacks, including paraphrasing, discreet alterations, tokenization, homoglyph, zero-width, and generative attacks, which aim to remove or forge watermarks [arxiv/2301_10226#18].

**X3**

Adversarial attacks on AI systems have evolved from simple perturbations that fool image classifiers to advanced prompt injections that manipulate language models [arxiv/2510_24801#28]. This evolution encompasses a range of attacks across different modalities, including text, images, and audio [arxiv/2510_24801#28].
These attacks can manifest in various forms. For language models, prompt injections and jailbreaking attacks aim to bypass safety guardrails and elicit unintended or harmful outputs [arxiv/2503_21460#28, arxiv/2308_12833#48]. Techniques include crafting deceptive prompts, using adversarial suffixes, or exploiting the model's role-playing abilities [arxiv/2504_01990#207, arxiv/2308_12833#48, arxiv/2411_07494#12]. In image generation, backdoor attacks can be embedded through poisoned text prompts, leading to the generation of specific target images [arxiv/2508_1605#10].
Defensive strategies are being developed to counter these evolving threats. These include adversarial training, where models are trained on adversarial examples to improve robustness [arxiv/2504_01990#207, arxiv/2504_01849#60]. Other approaches involve input sanitization, response filtering, and the use of constitutional classifiers [arxiv/2504_01990#203, arxiv/2501_18837#18]. For language models, methods like prompt validation, anomaly detection, and response filtering are employed to enhance resilience against prompt hacking [arxiv/2310_14735#78].

**X4**

Adversarial attacks on AI systems have indeed evolved significantly, moving from simple image manipulations to sophisticated methods targeting language models. These attacks aim to compromise the reliability and integrity of AI agents by deceiving them through altered inputs. [arxiv/2503_21460#27]
In the realm of language models, advanced prompt injections represent a key evolution in adversarial attacks. These attacks manipulate language models by carefully crafting deceptive prompts that can mislead them into generating harmful outputs. [arxiv/2504_01990#207] This contrasts with earlier adversarial attacks, which often focused on minimal or imperceptible perturbations in other machine learning contexts. [arxiv/2504_01990#203]
The progression from simple perturbations to prompt injections highlights a growing sophistication in adversarial tactics. While early attacks might have focused on fooling image classifiers with minor alterations, current methods leverage the generative capabilities of language models to achieve malicious ends. [arxiv/2504_01990#203, arxiv/2503_21460#27] This evolution underscores the dynamic nature of AI security, where defenses must constantly adapt to new attack vectors.

---

## Item 14

**Query:** accurately locate the relevant evidence passages, leading to a more comprehensive and precise response.

**X1**

The question asks to accurately locate relevant evidence passages for a more comprehensive and precise response. This involves understanding how different retrieval and generation methods are evaluated and how their components function.
In document retrieval, evaluation metrics like Recall@k, Precision@k, and F1@k are used to assess performance. Recall@k measures the fraction of relevant documents retrieved within the top-k chunks, while Precision@k measures the fraction of relevant documents among all retrieved documents. The F1@k score balances these two metrics, which is particularly crucial when the number of retrieved chunks is fixed but the number of retrieved documents varies, as it better captures the trade-off between retrieving enough relevant information and avoiding irrelevant noise [arxiv/2410_13070#21]. For evidence retrieval specifically, F1 scores are also emphasized to balance the biases introduced by chunk size, where larger chunks might increase recall but decrease precision [arxiv/2410_13070#21].
Retrieval-Augmented Generation (RAG) systems typically consist of a retriever and a generator. The retriever maps a query to relevant text chunks from a corpus, and the generator uses these passages to produce an answer. This approach aims to ground generated text in explicit evidence, improve factuality, and enable domain adaptation without full model fine-tuning [arxiv/2511_01386#4]. Different retrieval strategies exist, including sparse lexical retrievers (like TF-IDF or BM25) and dense retrieval methods that embed queries and passages into a shared vector space [arxiv/2511_01386#4]. For instance, Dense Passage Retrieval (DPR) uses dual-encoder models trained with supervised question-evidence pairs to achieve efficient and effective retrieval [arxiv/2511_01386#4].
To improve retrieval, methods like Fusion-in-Decoder (FiD) allow generators to attend across multiple encoded passages, overcoming the bottleneck of concatenating a small number of passages [arxiv/2511_01386#4]. Furthermore, training data construction for retrievers can involve measuring passage utility from both local relevance and global answer correctness perspectives. Local relevance assesses if a passage answers a specific turn's query, often using LLM-based listwise scoring. Global answer correctness evaluates how a passage influences the final answer's accuracy, typically using exact match metrics [arxiv/2601_11888#7]. Passages are then ranked based on these metrics to select positive and negative examples for retriever training [arxiv/2601_11888#7].

**X2**

To accurately locate relevant evidence passages for a comprehensive and precise response, a multi-faceted approach is employed, primarily centered around Retrieval-Augmented Generation (RAG) [arxiv/2502_01549#15, arxiv/2511_01386#4]. This paradigm enhances Large Language Models (LLMs) by integrating external knowledge through a process of retrieval and subsequent generation [arxiv/2511_01386#4]. The core RAG pipeline involves indexing data into a searchable database, retrieving pertinent information based on user queries, and then using this retrieved knowledge to generate informed responses [arxiv/2502_01549#15, arxiv/2312_10997#3].
Various strategies are utilized to optimize the retrieval component, which is critical for the quality of the final output [arxiv/2511_01386#4]. These include chunk-based methods that focus on text segmentation and advanced vector space embeddings, as well as graph-based methods that leverage structured knowledge representations for enhanced precision [arxiv/2502_01549#15]. Techniques like Contextual Chunk Headers (CCH) attach document-level and section-level context to chunks before embedding, improving semantic quality and retrieval precision [arxiv/2511_01386#30]. The Parent Document Retriever balances semantic precision with contextual coherence by embedding small, focused chunks while retaining links to their parent documents [arxiv/2511_01386#30]. Hypothetical Prompt Embedding (HyPE) further refines retrieval by precomputing and embedding multiple hypothetical prompts for each document chunk, improving alignment between queries and content [arxiv/2511_01386#30].
Evaluation metrics are crucial for assessing the effectiveness of retrieval and generation processes. For document retrieval, metrics like Recall@k, Precision@k, and F1@k are used to evaluate the fraction of relevant documents retrieved within a specified number of chunks [arxiv/2410_13070#21]. In evidence retrieval, similar metrics are applied to assess the proportion of retrieved evidence sentences over all retrieved sentences, with the F1 score being particularly useful for balancing recall and precision [arxiv/2410_13070#21]. For answer generation, BERTScore measures semantic similarity between generated and actual answers, while QA Similarity assesses the consistency and correctness of generated answers in relation to the original query [arxiv/2410_13070#21].
Despite these advancements, challenges remain. The retrieval phase can struggle with precision and recall, potentially leading to the selection of irrelevant chunks or the omission of crucial information [arxiv/2312_10997#3]. Generation can suffer from hallucination, where models produce content not supported by the retrieved context, or exhibit irrelevance, toxicity, or bias [arxiv/2312_10997#3]. Integrating retrieved information effectively and ensuring stylistic consistency also present augmentation hurdles [arxiv/2312_10997#3].

**X3**

To accurately locate relevant evidence passages for a comprehensive and precise response, various strategies are employed, primarily within the framework of Retrieval-Augmented Generation (RAG) systems. These systems aim to enhance Large Language Models (LLMs) by incorporating external knowledge through a process of retrieval and generation. The effectiveness of RAG hinges on the quality of the retrieval process, which involves identifying and extracting pertinent information from a corpus of documents.
Key to accurate evidence location is the meticulous processing of documents, particularly those in PDF format. Tools and libraries like PyMuPDF, GROBID, and nvpdftex are utilized to extract text, structure, and even images from PDFs, converting unstructured data into more manageable formats [arxiv/2406_11271#9, arxiv/2501_17887_methods#3, arxiv/2401_08406#6, arxiv/2601_14251#11]. This extraction process is crucial for preparing the data for subsequent retrieval steps.
The retrieval process itself can be approached through several methods, including vector retrieval, keyword search (like BM25), and hybrid search that combines both [arxiv/2511_01386#32]. To improve retrieval accuracy, techniques such as query expansion, decomposition of complex queries into smaller parts, and hypothetical document embedding (HyDE) are employed [arxiv/2511_01386#31, arxiv/2402_19473#12]. Furthermore, optimizing the granularity of retrieved information, such as through chunking strategies, is vital, as overly large chunks can introduce noise and reduce precision [arxiv/2408_10808#6, arxiv/2410_19572#4]. The quality of retrieval is often evaluated using metrics like recall, precision, and F1 scores [arxiv/2410_13070#21].

**X4**

Retrieval-Augmented Generation (RAG) is a paradigm that enhances Large Language Models (LLMs) by integrating external information before generating a response [arxiv/2502_01549#15, arxiv/2511_01386#4]. This approach involves three core components: indexing, retrieval, and generation [arxiv/2502_01549#15, arxiv/2312_10997#3]. First, data is processed and structured into a database. Then, this database is queried to retrieve pertinent information based on user input. Finally, the retrieved knowledge is used to generate informed responses [arxiv/2502_01549#15].
Recent advancements in RAG have explored different methodological trajectories. Chunk-based approaches focus on optimizing text segmentation and retrieval using vector space embeddings [arxiv/2502_01549#15]. In parallel, graph-based methods utilize structured knowledge representations to improve retrieval efficiency and precision [arxiv/2502_01549#15]. Multi-modal RAG systems have also emerged, leveraging databases of multi-faceted documents to enrich knowledge retrieval and generation [arxiv/2502_01549#15].
The retrieval process itself can be evaluated using metrics such as recall, precision, and F1 score, which measure the fraction of relevant documents or sentences retrieved within a given set of results [arxiv/2410_13070#21, arxiv/2408_08067#11]. Techniques like Contextual Chunk Headers (CCH) and Parent Document Retriever aim to improve retrieval quality by attaching document-level context to chunks or maintaining links to parent documents, thereby enhancing both semantic precision and contextual coherence [arxiv/2511_01386#30]. Hypothetical Prompt Embedding (HyPE) further refines retrieval by precomputing hypothetical prompts for document chunks, improving alignment between queries and relevant content [arxiv/2511_01386#30].
Despite these advancements, challenges remain. Naive RAG, an early methodology, often struggles with retrieval precision and recall, potentially selecting misaligned or irrelevant chunks while missing crucial information [arxiv/2312_10997#3]. Generation can also suffer from hallucination, irrelevance, toxicity, or bias, and integrating retrieved information can lead to disjointed or redundant outputs [arxiv/2312_10997#3].

---

## Item 15

**Query:** During inference, multiple decoding strategies are available to control the test-time compute.

**X1**

During inference, multiple decoding strategies are available to control test-time compute, aiming to enhance LLM reasoning and task performance by increasing computation. These strategies often involve adaptive allocation of computational resources based on problem difficulty, a concept known as "compute-optimal scaling" [arxiv/2501_11223#30, arxiv/2408_03314#0]. This approach allows smaller models to outperform significantly larger ones when given sufficient test-time compute [arxiv/2408_03314#0].
One such strategy is Chain-of-Thought (CoT) decoding, which effectively elicits reasoning from language models by encouraging the generation of intermediate reasoning steps. Unlike standard decoding methods such as temperature sampling, top-k sampling, nucleus sampling, or beam search, CoT-decoding has been shown to significantly improve accuracy on reasoning tasks, sometimes doubling or tripling performance compared to greedy decoding [arxiv/2402_10200#7]. This method works by identifying and ranking decoding paths that contain CoT components, often indicated by higher model confidence in the final answer [arxiv/2402_10200#5].
Further advancements include methods that dynamically adjust compute allocation based on prompt difficulty. For instance, the Compute-Optimal Scaling (COS) strategy categorizes prompts into difficulty levels and adapts compute allocation accordingly: easier prompts undergo sequential refinement, while harder prompts trigger parallel sampling or beam search. This approach can achieve significant efficiency gains over traditional methods like best-of-N sampling [arxiv/2502_21321#41]. Other approaches focus on scaling test-time compute through mechanisms like searching against verifier reward models or adaptively updating the model's distribution over a response, with effectiveness varying based on prompt difficulty [arxiv/2408_03314#0].

**X2**

During inference, multiple decoding strategies are available to control the test-time compute. These strategies aim to improve LLM reasoning capabilities and task performance by increasing computation. Methods include scaling token compute, self-feedback, and multi-agent refinement frameworks. Test-Time Scaling (TTS) strategies are applied at test time to control compute. Advancements in LLMs focus on enhancing reasoning through test-time compute (TTC) strategies. These include chain-of-thought (CoT) prompting, tree-of-thoughts (ToT), and retrieval-augmented generation (RAG) to improve multi-step reasoning and knowledge integration. Novel frameworks for parallel inference and structured knowledge scaling are also explored for more efficient and accurate complex reasoning. Speculative decoding is another method for accelerating autoregressive inference, where a smaller model generates draft tokens that are verified by a larger model, leading to significant speedups. Variations like quadratic decoding are also discussed.
One approach is CoT-decoding, which effectively elicits reasoning from language models by identifying and extracting Chain-of-Thought (CoT) paths during the decoding process. This method has shown significant reasoning gains, outperforming other decoding strategies like temperature sampling, top-k sampling, nucleus sampling, and beam search [arxiv/2402_10200#7]. CoT-decoding can also be combined with CoT-prompting for further performance boosts [arxiv/2402_10200#9].
Another set of strategies focuses on optimizing test-time compute allocation based on prompt difficulty. The Compute-Optimal Scaling Strategy (COS) categorizes prompts and adapts compute allocation, using sequential refinement for easier prompts and parallel sampling or beam search for harder ones. This approach can achieve significant efficiency gains compared to traditional methods [arxiv/2502_21321#41]. Similarly, scaling test-time compute optimally can be more effective than scaling model parameters, with effectiveness varying critically based on prompt difficulty [arxiv/2408_03314#0].
Speculative decoding is a technique that accelerates autoregressive generation by using a smaller "drafter" model to propose sequences of tokens, which are then verified by the larger target model. This can lead to significant speedups by reducing the number of forward passes required by the LLM [arxiv/2602_06036#1, arxiv/2404_14294#30]. Variations like quadratic decoding interleave mask tokens within speculative tokens to ensure consistent decoding progress [arxiv/2507_11851#9]. Graph-structured speculative decoding (GSD) further enhances this by using a directed acyclic graph to manage multiple hypotheses, improving acceptance rates and speedup [arxiv/2407_16207#1]. SpecReason, on the other hand, focuses on semantic-level similarity for reasoning tasks, allowing for approximate intermediate steps and can be combined with token-level speculative decoding [arxiv/2504_07891#7].

**X3**

Multiple decoding strategies can be employed during inference to control test-time compute. One such strategy is CoT-decoding, which effectively elicits reasoning paths from language models by encouraging diversity at the initial decoding steps and identifying CoT paths based on answer confidence [arxiv/2402_10200#5, arxiv/2402_10200#7]. This approach can significantly improve accuracy, even outperforming greedy decoding and other sampling methods when CoT prompts are not used [arxiv/2402_10200#5, arxiv/2402_10200#7].
Another approach is a compute-optimal scaling strategy that adaptively allocates test-time compute based on prompt difficulty [arxiv/2408_03314#0, arxiv/2501_11223#30]. This method categorizes prompts by difficulty and adjusts compute allocation, using sequential refinement for easier prompts and parallel sampling or beam search for harder ones [arxiv/2502_21321#41]. This adaptive allocation can lead to significant efficiency gains compared to traditional methods like best-of-N sampling [arxiv/2408_03314#0, arxiv/2502_21321#41].
These strategies can involve iterative refinement, where a model's output is enhanced multiple times for harder problems, or traversal operators that explore multiple reasoning paths at test time [arxiv/2501_11223#30]. The effectiveness of these test-time compute scaling methods can vary depending on the difficulty of the prompt, motivating the use of compute-optimal strategies that adapt per prompt [arxiv/2408_03314#0].

**X4**

During inference, multiple decoding strategies can be employed to control test-time compute. One such strategy is "CoT-decoding," which aims to reliably extract Chain-of-Thought (CoT) paths from a model's decoding process. This method focuses on identifying paths that exhibit higher model confidence, often indicated by a significant probability difference between the top and secondary tokens for answer tokens [arxiv/2402_10200#5]. Unlike methods like self-consistency, CoT-decoding is designed to work even when CoT paths do not have the highest probability among all decoded paths [arxiv/2402_10200#5].
Other decoding strategies include temperature sampling, top-k sampling, nucleus sampling, and beam search. However, CoT-decoding has been shown to be the only strategy that effectively enhances language model reasoning, significantly improving accuracy on tasks like GSM8K compared to these other methods [arxiv/2402_10200#7]. For instance, CoT-decoding can achieve substantially higher accuracy than greedy decoding or self-consistency without CoT prompts [arxiv/2402_10200#7, arxiv/2402_10200#9].
A more advanced approach to managing test-time compute is "compute-optimal scaling" (COS). This strategy adaptively allocates computational resources based on prompt difficulty, rather than applying a uniform sampling strategy. Easier prompts might undergo sequential refinement, while harder prompts trigger parallel sampling or beam search to explore multiple response variations. This adaptive allocation can lead to significant efficiency gains, achieving higher performance with less compute compared to traditional methods like best-of-N sampling [arxiv/2408_03314#24, arxiv/2502_21321#41]. This approach recognizes that the effectiveness of scaling test-time compute varies with problem difficulty [arxiv/2408_03314#24].

---

## Item 16

**Query:** Given that these extra tokens do not have any meaningful content, we programmatically removed certain symbols at the end of summaries.

**X1**

The removal of certain symbols at the end of summaries was a programmatic step taken to improve the focus of human evaluators on the content of the summaries, rather than being distracted by formatting. This was observed to be a common issue with summaries generated by RLHF and RLAIF policies, possibly due to "reward hacking" [arxiv/2309_00267#38].

**X2**

To address the issue of extra tokens at the end of summaries that do not contribute meaningful content, a programmatic removal of certain symbols was implemented. This was done to ensure that human evaluators could focus on the substance of the summaries without being distracted by superficial formatting [arxiv/2309_00267#38]. This approach is a post-processing step applied to the generated summaries.

**X3**

To address the issue of extra tokens in summaries that lack meaningful content, certain symbols were programmatically removed from the end of these summaries. This action was taken because these superfluous tokens, potentially a result of "reward hacking" during RLHF or RLAIF policies, did not contribute to the actual content of the summary [arxiv/2309_00267#38]. The removal aimed to ensure that human evaluators could focus on the substantive information without being distracted by extraneous formatting.

**X4**

To address the issue of superfluous symbols at the end of summaries, certain symbols were programmatically removed. This was done to ensure that human evaluators could focus on the content of the summaries without being distracted by formatting artifacts, which may have arisen from "reward hacking" in summarization policies [arxiv/2309_00267#38].

---

## Item 17

**Query:** MMLU(Hendrycksetal., 2020) is heterogeneous with regard to the type of reasoning skills required to answer the questions, containing instances that require basic factual recall as well as those that demand logical reasoning and problem-solving skills.

**X1**

MMLU is a diverse dataset designed to evaluate the reasoning skills of large language models, encompassing a spectrum from simple factual recall to complex logical deduction and problem-solving [arxiv/2503_16419#1]. This heterogeneity means that models evaluated on MMLU must demonstrate proficiency in various cognitive tasks, including the ability to process and synthesize information from multiple sources, identify logical inconsistencies, and perform step-by-step analyses [arxiv/2503_16252#27].
The dataset's design necessitates that models can handle questions that require basic factual recall, as well as those that demand more sophisticated logical reasoning and problem-solving abilities [arxiv/2503_16419#1]. This is reflected in the evaluation frameworks that assess the truthfulness of statements and questions, requiring checks for relevance, consistency, and accuracy, alongside detailed evaluations of logical expressions [arxiv/2503_16252#27]. MMLU also includes elementary mathematical reasoning tasks [arxiv/2503_16419#1].
The evaluation of models on such a dataset involves assessing their capacity for structured, step-by-step thinking, which is crucial for tackling complex problems [arxiv/2502_19411#31]. This includes the ability to decompose problems, maintain logical consistency throughout a reasoning process, and avoid errors or omissions in their deductions [arxiv/2503_16252#27]. The dataset's breadth ensures a comprehensive assessment of a model's reasoning repertoire.

**X2**

The MMLU benchmark is designed to assess a wide range of reasoning skills, encompassing both basic factual recall and more complex logical reasoning and problem-solving abilities [arxiv/2503_16419#1]. This heterogeneity means that models evaluated on MMLU must demonstrate proficiency across different cognitive demands [arxiv/2503_16419#1].
To address these varied reasoning requirements, researchers are exploring various approaches. One direction involves optimizing reasoning processes for efficiency, aiming to reduce computational overhead while maintaining performance [arxiv/2503_16419#1]. This includes techniques that dynamically reduce reasoning steps during inference or enhance reasoning efficiency based on input prompt properties [arxiv/2503_16419#1]. Another area of focus is the development of benchmarks and evaluation frameworks that can systematically assess these diverse reasoning capabilities, such as those that require multi-hop reasoning, logical inference, or structured problem-solving [arxiv/2503_16252#27, arxiv/2502_19411#31, arxiv/2503_16419#23].
The evaluation of reasoning processes itself is also a critical aspect, with criteria being developed to assess internal consistency, term overlap with standard answers, the number of reasoning steps, logical consistency, content diversity, and relevance to the task domain and instructions [arxiv/2503_16252#27]. These evaluation methods aim to provide a nuanced understanding of a model's reasoning strengths and weaknesses.

**X3**

MMLU is a diverse dataset that requires various reasoning skills, ranging from simple factual recall to complex logical deduction and problem-solving. This heterogeneity means that models evaluated on MMLU must demonstrate a broad spectrum of cognitive abilities to perform well. The dataset's design intentionally includes questions that test different facets of reasoning, ensuring a comprehensive assessment of a model's capabilities.
The evaluation of MMLU questions involves assessing the truthfulness of statements and answers, which necessitates checking for relevance, consistency, and accuracy. Furthermore, the dataset includes problems that demand step-by-step logical reasoning and problem-solving, as well as elementary mathematical skills. This multifaceted approach to evaluation ensures that models are tested not just on their ability to retrieve information but also on their capacity to process and manipulate it logically.
The heterogeneity of MMLU is further underscored by its inclusion of tasks that require basic factual recall alongside those demanding sophisticated logical reasoning and problem-solving skills. This blend of question types means that a model's performance on MMLU is a strong indicator of its overall reasoning proficiency, encompassing both knowledge retrieval and the application of logical principles.

**X4**

MMLU is a broad dataset that encompasses a variety of question types, requiring different reasoning skills. Some questions can be answered through basic factual recall, while others demand more complex logical reasoning and problem-solving abilities [arxiv/2409_12183#27]. This heterogeneity means that models need to be proficient in a range of cognitive processes to perform well on the benchmark.
The dataset's diversity is reflected in the types of reasoning it probes. For instance, while some categories within MMLU might appear purely mathematical, closer inspection reveals that they often involve computations and reasoning akin to those found in business contexts [arxiv/2409_12183#27]. This suggests that MMLU challenges models to integrate different forms of reasoning, moving beyond simple recall to more analytical and inferential tasks.

---

## Item 18

**Query:** The training configuration for our method uses a batch size of 64 and a learning rate of 2 0 .

**X1**

The training configuration for the method uses a batch size of 64. The learning rate is specified as 2.0, which is also represented as 2 0 in the provided text. [arxiv/2411_00418#28] [arxiv/2402_16828#27] [arxiv/2401_08406#11] [arxiv/2503_02197#29] [arxiv/2302_05442#43] [arxiv/2410_02536#10] [arxiv/2408_02193#15]
Several related communities mention a batch size of 64. For instance, one study uses a batch size of 64 for training a decoder. [arxiv/2302_05442#43] Another paper uses a batch size of 64 for training on specific datasets within the Long Range Arena benchmark and for language modeling tasks. [arxiv/2410_01201#51] Additionally, a batch size of 64 is used in the context of training with a specific ECA rule. [arxiv/2410_02536#10]
Regarding the learning rate, a value of 2.0 or 2 0 is mentioned. [arxiv/2411_00418#28] [arxiv/2402_16828#27] [arxiv/2401_08406#11] [arxiv/2503_02197#29] [arxiv/2302_05442#43] [arxiv/2410_02536#10] [arxiv/2408_02193#15] For example, one paper uses a learning rate of 2e-5, which is numerically distinct from 2.0. [arxiv/2411_00418#28] Another paper lists a learning rate of 2e-5 for LTE on ImageNet100. [arxiv/2402_16828#27] Similarly, a learning rate of 2e-5 is used in other fine-tuning settings. [arxiv/2401_08406#11] [arxiv/2503_02197#29] [arxiv/2408_02193#15] However, the specific value of 2.0 or 2 0 is not directly found in the provided subgraph sections.

**X2**

The training configuration for our method uses a batch size of 64 and a learning rate of 2e-5.
The learning rate is specified as 2e-5 [arxiv/2401_08406#11]. The batch size is 64 [arxiv/2309_16609#8].

**X3**

The training configuration for the method uses a batch size of 64 and a learning rate of 2e-5. [arxiv/2411_00418#28]
This configuration is part of the instruction fine-tuning process, which employs a learning rate of 2e-5 with cosine decay and 2 warmup steps. [arxiv/2411_00418#28] The effective global batch size is 32, achieved with a micro-batch size of 4 and gradient accumulation of 8. [arxiv/2402_09147#29] For the LoRA configuration, r=64, alpha=128, dropout=0.05, and bias='none' are used, targeting the q and v projection modules. [arxiv/2402_09147#29]

**X4**

The training configuration for our method uses a batch size of 64 and a learning rate of 2e-5.
The training configuration for QWEN involves autoregressive language modeling with a context length of 2048. It utilizes Flash Attention for efficiency and the AdamW optimizer with specific beta values (β1 = 0.9, β2 = 0.95) and an epsilon of 10^-8. A cosine learning rate schedule is employed, with the learning rate decaying to 10% of its peak value. All models are trained with BFloat16 mixed precision for stability [arxiv/2309_16609#8].
For scalability experiments, different model sizes (1B to 7B parameters) are trained with varying learning rates and batch sizes. For instance, a 1B parameter model uses a learning rate of 0.0008 and a batch size of 0.3 million, while a 7B parameter model uses a learning rate of 0.00037 and a batch size of 0.8 million [arxiv/2412_19255#22].
In fine-tuning GPT-4, a batch size of 256 samples is used with a base learning rate of 1e-4, employing Low Rank Adaptation (LoRA) for efficiency [arxiv/2401_08406#11]. For other models, such as Open-Llama-3b, Llama2-7b, and Llama2-13b, an effective batch size of 128 samples per training step is achieved by accumulating gradients over microbatches, with a base learning rate of 2e-5 and a cosine learning rate scheduler [arxiv/2401_08406#11].

---

## Item 19

**Query:** The new tests for search can be seen in Figure 23, and the results can be seen in Figure 24.

**X1**

The new tests for search are presented in Figure 23, and their corresponding results are shown in Figure 24 [arxiv/2504_03601#27].

**X2**

The provided context does not contain information about "Figure 23" or "Figure 24" in relation to new tests for search or their results. Therefore, I cannot answer the question based on the given information.

**X3**

The new tests for search are presented in Figure 23, and their results are detailed in Figure 24. These figures are part of a broader evaluation of retrieval-augmented language models, where the number of retrieved documents, denoted by 'k', is a key parameter influencing performance. The results indicate that optimal performance is often achieved with a specific range of 'k' values, varying across different datasets and retrieval methods. For instance, on the ASQA dataset, optimal performance with LLaMA is observed when k=4 or k=5, while for QAMPARI, k=10 yields the best results [arxiv/2411_07396#21].
The evaluation also considers different retrieval models, such as BGE-base and ColBERT, and their impact on metrics like EMRecall and Citation Recall/Precision [arxiv/2411_07396#21]. For example, when using ColBERT for retrieval on ASQA, optimal EMRecall is achieved with k=2, k=3, and k=4, with the highest score at k=4 [arxiv/2411_07396#21]. Similarly, for the NQ dataset, using Mistral with BGE-base retrieval, performance improves with increasing 'k' up to k=20, after which it slightly declines [arxiv/2411_07396#21].
The context also touches upon the optimization of workflows, where a self-adaptive operator set and an operator memory are integrated into a search and optimization procedure to yield more compact and task-specific solutions [arxiv/2511_20693#9]. Additionally, prompts are a crucial component in various stages of different systems, including task configuration generation, alignment validation, and final semantic review [arxiv/2504_03601#27]. Prompts are also used in physical simulation settings, such as 2D pool ball and cannon ball scenarios [arxiv/2504_07128#146].

**X4**

The new tests for search are presented in Figure 23, and their outcomes are detailed in Figure 24. These figures are part of a broader investigation into optimizing search and workflow processes.
The research explores various methods for improving search and workflow efficiency. One approach involves an automated workflow optimization mechanism that builds upon the AFLOW framework. This mechanism follows a cycle of initialization, selection, expansion, evaluation, and backpropagation, utilizing a mixed probability selection strategy to balance exploration and exploitation. It incorporates a self-adaptive abstract operator set and an Operators Memory to dynamically generalize and merge execution steps, allowing for joint optimization of workflow structures and operator representations. [arxiv/2511_20693#9]
Another aspect of the research involves evaluating the impact of varying the number of retrieved documents, denoted by 'k', on search performance. This is demonstrated through detailed results presented in tables and figures, showing correctness and citation quality metrics for different datasets and language models. For instance, optimal performance for certain datasets is observed with specific values of 'k', such as k=4 or 5 for ASQA and NQ, and k=10 for QAMPARI. [arxiv/2411_07396#21]
Additionally, the context touches upon prompts used in various stages of different systems, such as APIGen-MT and BEATS algorithms, and physical simulation settings. It also discusses a token-level coldness taxonomy for unseen target items, categorizing them based on whether their constituent tokens have been seen during training, and reports performance metrics like Recall@20 and NDCG@20 for these categories. [arxiv/2504_03601#27], [arxiv/2607_21101#18], [arxiv/2409_17972#21], [arxiv/2504_07128#146]

---

## Item 20

**Query:** (3) Large-scale context windows RAG ensembles : Enabling effective management and coherent synthesis of multimodal data (eg, text, images)for in-depth professional research analysis.

**X1**

Large-scale context windows, when integrated with Retrieval-Augmented Generation (RAG) ensembles, are crucial for effectively managing and synthesizing multimodal data, such as text and images, to support in-depth professional research analysis.
The core mechanism involves using RAG ensembles to process and combine diverse data types.
The effectiveness of such systems is demonstrated through their application in advanced research scenarios.

**X2**

Large-scale context windows and Retrieval Augmented Generation (RAG) ensembles are being developed to enable effective management and coherent synthesis of multimodal data for in-depth professional research analysis. These approaches aim to process and integrate diverse data types, such as text and images, to provide comprehensive insights.
One aspect of this development involves training multimodal models that can effectively interpret complex visual information alongside text. This is achieved through strategies like making vision encoders trainable throughout the training process, allowing the model to capture both low-level and high-level semantic visual information.
The implementation of these advanced RAG ensembles often involves Python programming and can be demonstrated through code examples that process and analyze data.
Furthermore, these systems are being applied to autonomous data science cases, covering the entire pipeline from data sources to research reports.

**X3**

Large-scale context windows are being developed to enable effective management and coherent synthesis of multimodal data for in-depth professional research analysis. These advancements aim to overcome limitations in processing and understanding extensive datasets that include various modalities like text and images.
Current research explores methods to handle long contexts, such as reordering documents based on their importance scores to better leverage LLMs' positional perception differences, thereby reducing information loss in the middle of long contexts [arxiv/2310_06839#9]. Additionally, techniques like pooling layers are used to condense large images into smaller summary images, providing a form of data reduction and location invariance [arxiv/ISLP#320].
The development of these large-scale context windows is part of a broader effort to enhance AI capabilities in research. This includes autonomous data science pipelines that can generate analyst-grade research reports from data sources [arxiv/2510_16872#23]. AI tools are also being integrated to support various aspects of the research workflow, including literature search, summarization, and code generation [arxiv/2502_05151#21].
While the focus is on improving the processing of extensive and multimodal data, specific details on how these large-scale context windows directly manage and synthesize multimodal data for professional research analysis are not fully elaborated in the provided snippets. However, the general direction points towards more sophisticated AI systems capable of handling complex research tasks.

**X4**

Large-scale context windows, when combined with Retrieval-Augmented Generation (RAG) ensembles, can enable effective management and coherent synthesis of multimodal data, such as text and images, for in-depth professional research analysis. RAG is a method that improves multi-step reasoning and knowledge integration. Novel frameworks for parallel inference and structured knowledge scaling are being explored to achieve more efficient and accurate complex reasoning within these systems.
This technique is particularly useful for knowledge integration in explanatory answer generation tasks [arxiv/2410_17694#5, arxiv/2410_17694#8].
The integration of multimodal data, including text and images, into LLMs is a key area of research. Multimodal Large Language Models (MLLMs) are being developed to handle diverse data types and tasks [arxiv/2311_04257#2, arxiv/2407_12580#1]. These models aim to bridge the gap between different modalities, allowing for a more comprehensive understanding and generation of content [arxiv/2407_12580#1, arxiv/2309_05519#4]. Frameworks are being explored to adapt MLLMs for universal multimodal embeddings, enabling them to represent various inputs effectively [arxiv/2407_12580#1].
To manage and synthesize multimodal data effectively for professional research, large-scale context windows are being combined with RAG ensembles.

---

## Item 21

**Query:** Rationale :	The hierarchical	structure	distributes	impact	forces	more	evenly,	reducing	stress	concentrations	and	enhancing	energy	absorption.

**X1**

The hierarchical structure of the material enhances its crashworthiness by distributing impact forces more evenly. This even distribution reduces stress concentrations within the material, which in turn improves its energy absorption capabilities [arxiv/2409_05556#367]. Consequently, the material is better equipped to handle and dissipate energy during impacts, leading to a significant increase in its overall crashworthiness compared to traditional materials [arxiv/2409_05556#423].
This improved energy absorption is a key factor in the material's enhanced toughness and durability [arxiv/2502_13025#155, arxiv/2502_13025#208]. The hierarchical design allows the material to absorb destructive energy during fracture, thereby preventing catastrophic failure [arxiv/2409_05556#642]. This mechanism is expected to result in a high fracture toughness, with energy absorption capacities comparable to or exceeding natural nacre [arxiv/2409_05556#651].
The layered lamellar structure contributes to this by providing enhanced mechanical strength and flexibility, further reducing the likelihood of failure under stress [arxiv/2409_05556#278]. This structural advantage allows the material to withstand more cycles of loading without significant degradation, making it more mechanically stable and robust [arxiv/2409_05556#278].

**X2**

The hierarchical structure of the material enhances its crashworthiness by distributing impact forces more evenly. This even distribution reduces stress concentrations, which in turn improves the material's ability to absorb energy during an impact [arxiv/2409_05556#367, arxiv/2409_05556#423]. This improved energy absorption capacity is a key factor in its enhanced crashworthiness, leading to a quantitative increase compared to traditional materials [arxiv/2409_05556#367].
The hierarchical structure is also crucial for the material's energy absorption capabilities, contributing to its toughness and durability [arxiv/2409_05556#642]. This structural design allows the material to dissipate energy effectively, preventing catastrophic failure and leading to a high fracture toughness [arxiv/2409_05556#651]. The material is expected to exhibit energy absorption comparable to or exceeding that of natural nacre [arxiv/2409_05556#651].
Furthermore, the layered lamellar structure contributes to enhanced mechanical stability by distributing stress more evenly across the material [arxiv/2409_05556#278]. This even distribution of stress is a fundamental mechanism that underpins the material's improved performance in impact scenarios and its overall durability [arxiv/2409_05556#278, arxiv/2409_05556#642].

**X3**

The hierarchical structure of the material is designed to distribute impact forces more evenly. This even distribution reduces stress concentrations within the material, which in turn enhances its energy absorption capabilities. This mechanism is key to improving the material's crashworthiness. [arxiv/2409_05556#367] [arxiv/2409_05556#423]
This hierarchical design allows the material to absorb destructive energy during fracture, preventing catastrophic failure and improving its overall toughness and durability. [arxiv/2409_05556#642] [arxiv/2409_05556#669] This is expected to result in a quantitative improvement in crashworthiness, with a projected 30% increase compared to traditional materials. [arxiv/2409_05556#423]

**X4**

The hierarchical structure enhances crashworthiness by distributing impact forces more evenly, which reduces stress concentrations and improves energy absorption. This design principle is crucial for materials used in structural applications, aiming to increase their ability to absorb and dissipate energy during impacts. [arxiv/2409_05556#403]
This approach leads to a quantitative improvement in crashworthiness, with a projected 30% increase compared to traditional materials. The energy absorption capacity is measured through impact tests, such as drop-weight or dynamic crash tests, with a quantitative goal of achieving 13-26 kJ/kg. [arxiv/2409_05556#423]
The hierarchical structure also contributes to enhanced energy absorption by preventing catastrophic failure and allowing the material to absorb destructive energy during fracture. This biomimetic approach aims to replicate and potentially exceed the energy absorption capabilities of natural structures like nacre. [arxiv/2409_05556#642, arxiv/2409_05556#685]
Overall, the hierarchical organization is key to improving crashworthiness and energy absorption by distributing forces, reducing stress, and enhancing the material's capacity to handle impact events. [arxiv/2409_05556#367]

---

## Item 22

**Query:** The output is a schema-valid JSON object, with the source page and a bounding box for each value as evidence .

**X1**

The output is a schema-valid JSON object, with the source page and a bounding box for each value as evidence [arxiv/2607_29677#38]. This JSON object must conform to a user-specified schema, which defines the desired structure of the extracted data [arxiv/2607_29677#38]. The schema lists the fields to extract, along with their types and natural-language descriptions [arxiv/2607_29677#38]. JSON Schema is the industry-standard way to specify this structured output, supporting various data types and constraints [arxiv/2607_29677#38]. The system is expected to return correct and exhaustive values, using `null` for any absent information, and grounding each extracted value with its source page and bounding box [arxiv/2607_29677#38].

**X2**



**X3**

The output is a schema-valid JSON object, with the source page and a bounding box for each value as evidence [arxiv/2607_29677#4]. This JSON object is generated by a system that takes a document and a schema as input and returns structured data along with evidence [arxiv/2607_29677#4]. The schema, expressed as a JSON Schema, defines the fields to extract, including their names, types, and descriptions [arxiv/2607_29677#4].
The JSON Schema specification itself is a JSON value that characterizes a set of JSON values conforming to it [arxiv/2405_17681#4]. It can describe various data types with constraints, using fields like "type" to specify the data type. Nested types, such as arrays and objects, can have subschemas under "items" and "properties" fields, respectively, to define the structure of their contents [arxiv/2405_17681#4].
The output must be schema-valid, meaning it adheres to the structure and types defined by the JSON Schema [arxiv/2607_29677#4]. This includes returning correct and exhaustive values, using null for absent information, and grounding each extracted value with its source page and bounding box [arxiv/2607_29677#4]. The grounding metric evaluates the correctness of both the value and its citation, calculating word-level and page-level F1 scores [arxiv/2607_29677_methods#4].

**X4**



---

## Item 23

**Query:** Among these, LoRA (Hu et al., 2022), which does not change the model architecture, has become notably popular for its simplicity and efficacy.

**X1**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that has gained popularity due to its simplicity and effectiveness. It achieves this by not altering the underlying model architecture. Instead, LoRA introduces small, trainable low-rank matrices into specific layers of a pre-trained model. These matrices are then trained on downstream tasks, while the original model weights remain frozen. This approach significantly reduces the number of trainable parameters compared to full fine-tuning, leading to faster training and lower memory requirements.
The core idea behind LoRA is to decompose the weight update matrix into two smaller matrices, effectively creating a low-rank representation of the update. This is achieved by adding trainable low-rank matrices (A and B) to the original weight matrices (W) of a pre-trained model, such that the updated weights become W + BA. This method has been shown to be effective across various models and tasks, including those involving large language models. For instance, spectral decomposition can be used to extract principal directions from LoRA matrices, revealing shared low-dimensional subspaces within collections of LoRA models [arxiv/2512_05117#19]. This analysis indicates that a significant portion of the variance in LoRA matrices can be captured by a relatively small number of principal directions, supporting the efficacy of the low-rank approximation.

**X2**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that has gained popularity due to its simplicity and effectiveness.
The core idea behind LoRA is to inject trainable low-rank matrices into the layers of a pre-trained model. These low-rank matrices are then trained, while the original weights of the model remain frozen. This method allows for efficient adaptation of large models to specific tasks or datasets by only updating a small fraction of the total parameters.
The efficacy of LoRA is further supported by research that analyzes the structure of LoRA matrices. For instance, spectral decomposition can be used to extract principal directions from LoRA matrices, revealing that a significant portion of the variance is captured by a low-dimensional subspace. This suggests that the adaptations learned by LoRA are often concentrated in a few key directions, reinforcing the efficiency of the low-rank approximation [arxiv/2512_05117#19].

**X3**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that has gained popularity for its simplicity and effectiveness. It works by freezing the pre-trained model's parameters and training only a small, low-rank adaptation module. This approach significantly reduces memory demands and computational costs associated with fine-tuning large language models, thereby speeding up the training process [arxiv/2307_13269#2].
The core idea behind LoRA is to inject trainable low-rank matrices into the layers of a pre-trained model. These matrices are typically decomposed into two smaller matrices, which are then trained. This method does not alter the original model architecture, contributing to its simplicity [arxiv/2307_13269#2]. The effectiveness of LoRA has been demonstrated in various applications, including the composition of multiple LoRA modules to generalize to new tasks without requiring task-specific training [arxiv/2307_13269#2].
Research has explored the underlying structure of LoRA matrices, suggesting that they often reside within a shared low-dimensional subspace. Techniques like spectral decomposition can be used to extract principal directions from these matrices, revealing that a significant portion of the variance is captured by a small number of these directions, indicating the presence of such a shared subspace [arxiv/2512_05117#19]. The number of principal components to use for analysis, such as in scree plots, is often determined by identifying an "elbow" where the proportion of variance explained by subsequent components drops significantly, though this can be an ad-hoc process [arxiv/ISLP#392].

**X4**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that has gained popularity for its simplicity and efficacy in adapting large language models (LLMs) without altering their core architecture [arxiv/2405_12130#2, arxiv/2311_09578#3, arxiv/2501_00365#2]. It achieves this by constraining the update to the model's weight matrices to be low-rank, effectively reducing the number of trainable parameters [arxiv/2310_18547#5, arxiv/2501_00365#2].
The primary advantage of LoRA lies in its parameter efficiency, which significantly reduces computational costs and memory requirements during fine-tuning [arxiv/2405_12130#2, arxiv/2311_09578#3, arxiv/2501_00365#2]. This efficiency allows for faster training and makes it feasible to adapt large models on less powerful hardware. Furthermore, LoRA adapters can be merged back into the original model weights after training, meaning there is no additional computational overhead during inference, unlike some other parameter-efficient fine-tuning methods [arxiv/2405_12130#2, arxiv/2106_09685#2, arxiv/2311_09578#3]. This modularity also allows for flexible task switching by simply swapping out the LoRA adapters without needing to store multiple full model copies [arxiv/2106_09685#2, arxiv/2501_00365#2].
LoRA's effectiveness stems from the observation that weight updates during fine-tuning often reside in a low-dimensional subspace [arxiv/2106_09685#2]. By approximating these updates with low-rank matrices, LoRA captures the essential task-specific adjustments while preserving the model's foundational knowledge and mitigating catastrophic forgetting [arxiv/2501_00365#2]. This approach has led to numerous variants and extensions aimed at further improving its performance and efficiency across various tasks and model architectures [arxiv/2406_09044#14, arxiv/2403_14608#15].

---

## Item 24

**Query:** Xixi Wu 1 3 , ∗ Yifei Shen 2 ∗ GLYPH&lt;12&gt; Caihua Shan 2 Kaitao Song 2 Siwei Wang 2 Bohang Zhang 4 Jiarui Feng 5 Hong Cheng 3 Wei Chen 2 Yun Xiong 1 GLYPH&lt;12&gt; Dongsheng Li 2

**X1**

The question asks about the individuals Xixi Wu, Yifei Shen, Caihua Shan, Kaitao Song, Siwei Wang, Bohang Zhang, Jiarui Feng, Hong Cheng, Wei Chen, Yun Xiong, and Dongsheng Li, and their affiliations.
Xixi Wu and Yun Xiong are affiliated with Fudan University and The Chinese University of Hong Kong, respectively, and are also associated with Microsoft Research Asia [arxiv/2405_19119#0]. Caihua Shan, Kaitao Song, Siwei Wang, Wei Chen, and Dongsheng Li are affiliated with Microsoft Research Asia [arxiv/2405_19119#0]. Bohang Zhang is affiliated with Peking University, and Jiarui Feng is affiliated with Washington University, Saint Louis [arxiv/2405_19119#0]. Hong Cheng is also affiliated with The Chinese University of Hong Kong [arxiv/2405_19119#0].
Kaitao Song is also listed as an author in a paper titled "Chain-of-Model Learning for Language Model," where he is affiliated with Microsoft Research [arxiv/2505_11820#0]. Dongsheng Li is also listed in this paper, affiliated with Microsoft Research [arxiv/2505_11820#0].
The provided context also lists numerous authors under "Research & Engineering" and "Data Annotation" sections from various documents, but these do not directly correspond to the specific individuals named in the question [arxiv/2512_02556#30, arxiv/2412_19437#64, arxiv/2412_19437#65].

**X2**

Xixi Wu is affiliated with Fudan University and Microsoft Research Asia, as indicated by the author list in the paper "Chain-of-Model Learning for Language Model" [arxiv/2505_11820#0]. She is also listed as an author in "Can Graph Learning Improve Planning in LLM-based Agents?" with affiliations including Fudan University, Microsoft Research Asia, and The Chinese University of Hong Kong [arxiv/2405_19119#0].
The context includes several author lists from different research papers, detailing contributors to projects in areas such as Research & Engineering, Data Annotation, and Business & Compliance [arxiv/2512_02556#30, arxiv/2412_19437#64, arxiv/2412_19437#65]. These lists often specify roles like "Core Contributors" or "Contributors" and are typically sorted alphabetically by first name [arxiv/2512_02556#30, arxiv/2412_19437#65].
The question specifically asks about Xixi Wu, and the most direct evidence places her at Fudan University and Microsoft Research Asia, and also with The Chinese University of Hong Kong [arxiv/2505_11820#0, arxiv/2405_19119#0]. While the global view mentions Tsinghua University in relation to AI and Computer Science researchers, it does not directly link Xixi Wu to that institution.

**X3**

The question asks to identify the authors Xixi Wu, Yifei Shen, Caihua Shan, Kaitao Song, Siwei Wang, Bohang Zhang, Jiarui Feng, Hong Cheng, Wei Chen, Yun Xiong, and Dongsheng Li. These individuals are listed as authors in the paper "Can Graph Learning Improve Planning in LLM-based Agents?" [arxiv/2405_19119#0].
The paper "Can Graph Learning Improve Planning in LLM-based Agents?" lists Xixi Wu, Yifei Shen, Caihua Shan, Kaitao Song, Siwei Wang, Bohang Zhang, Jiarui Feng, Hong Cheng, Wei Chen, Yun Xiong, and Dongsheng Li as authors [arxiv/2405_19119#0]. These authors are affiliated with various institutions, including Fudan University, Microsoft Research Asia, The Chinese University of Hong Kong, Peking University, and Washington University, Saint Louis [arxiv/2405_19119#0].
Further context from a related paper, "Chain-of-Model Learning for Language Model," also lists Kaitao Song and Dongsheng Li as authors, along with affiliations to Microsoft Research and Fudan University [arxiv/2505_11820#0]. Additionally, a broad "Research & Engineering" author list includes many names that appear in other retrieved sections, such as Aixin Liu, Bing Xue, Bochao Wu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, Hanwei Xu, Haowei Zhang, Honghui Ding, Huazuo Gao, Hui Qu, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Panpan Huang, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Tao Yun, Tian Pei, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y.K. Li, Y.Q. Wang, Y.X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z.F. Wu, Z.Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhiyu Wu, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zilin Li, Ziwei Xie, Ziyi Gao, Zizheng Pan [arxiv/2412_19437#64].
Additionally, a "Data Annotation" list includes Bei Feng, Hui Li, J.L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R.J. Chen, R.L. Jin, Ruyi Chen, S.S. Li, Shuang Zhou, Tianyu Sun, X.Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Y.X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, and

**X4**

The question asks about the individuals named Xixi Wu, Yifei Shen, Caihua Shan, Kaitao Song, Siwei Wang, Bohang Zhang, Jiarui Feng, Hong Cheng, Wei Chen, Yun Xiong, and Dongsheng Li. These individuals are listed as authors of a paper titled "Can Graph Learning Improve Planning in LLM-based Agents?" [arxiv/2405_19119#0]. The paper is associated with GLYPH<12> [arxiv/2405_19119#0].
The affiliations for these authors are provided: Xixi Wu and Yun Xiong are affiliated with Fudan University [arxiv/2405_19119#0]. Yifei Shen, Caihua Shan, Kaitao Song, Siwei Wang, Bohang Zhang, Hong Cheng, Wei Chen, and Dongsheng Li are affiliated with Microsoft Research Asia [arxiv/2405_19119#0]. Additionally, Xixi Wu is also affiliated with The Chinese University of Hong Kong [arxiv/2405_19119#0]. Bohang Zhang is affiliated with Peking University [arxiv/2405_19119#0], and Jiarui Feng is affiliated with Washington University, Saint Louis [arxiv/2405_19119#0].
Some of these individuals are also listed in other author lists. For instance, Xixi Wu is listed as an author in another context [arxiv/2512_02556#30]. Similarly, Yifei Shen, Caihua Shan, Kaitao Song, Siwei Wang, Bohang Zhang, Jiarui Feng, Hong Cheng, Wei Chen, Yun Xiong, and Dongsheng Li are also listed as authors in the same context [arxiv/2512_02556#30].
The context also indicates that names marked with an asterisk (*) denote individuals who have departed from the team [arxiv/2512_02556#30]. However, none of the names in the question are marked with an asterisk in the provided sections.

---

## Item 25

**Query:** The third family makes procedural write and reuse explicit: it turns a successful trajectory into a callable skill or a reusable workflow abstraction.

**X1**

The third family of methods explicitly addresses the creation and reuse of procedures by transforming successful execution trajectories into callable skills or reusable workflow abstractions [arxiv/2606_30306#56]. This approach treats skills as long-lived, evolving assets within an agent system, rather than one-off outputs [arxiv/2605_27366#3]. The core idea is to enable agents to continuously improve their capabilities by obtaining, organizing, and refining skills autonomously, without constant human intervention [arxiv/2605_27366#3].
These methods involve constructing and evolving reusable agent skills that encapsulate procedural knowledge [arxiv/2605_27366#3]. A successful trajectory is converted into a skill that can be invoked later, allowing for the reuse of discovered procedures [arxiv/2606_30306#56]. This process often involves a lifecycle that includes skill creation, memory storage, management, evaluation, and refinement, ensuring that skills are not static but can be updated and improved over time [arxiv/2605_27366#3].
The explicit nature of procedural write and reuse in this family means that skills are not merely advisory but are executable commitments [arxiv/2606_30306#56]. This necessitates mechanisms for validation and governance to prevent issues like "skill drift," where a skill that was once correct becomes outdated as the environment changes [arxiv/2606_30306#56]. Approaches within this family focus on optimizing persistent skill documents and utilizing execution feedback to maintain skill reliability and relevance [arxiv/2605_27366#3].

**X2**

The third family of methods explicitly addresses the creation and reuse of procedures by transforming successful execution trajectories into callable skills or reusable workflow abstractions [arxiv/2606_30306#56]. This involves developing reusable agent skills that encapsulate procedural knowledge, optimizing persistent skill documents, and using execution feedback to refine them [arxiv/2606_30306#56].
These approaches turn a successful trajectory into a callable skill or a reusable workflow abstraction [arxiv/2606_30306#56]. For example, embodied agents can build a library of executable skills over time, allowing later tasks to utilize previously discovered procedures [arxiv/2606_30306#56]. Similarly, web agents can induce high-level workflow abstractions from past trajectories and apply them in new sessions [arxiv/2606_30306#56]. Procedural skills are defined as reusable, executable competencies such as plans, scripts, or trajectories that an agent stores after success and replays later [arxiv/2606_30306#17].
Recent work treats the skill store as a first-class artifact, incorporating engineering disciplines for managing these reusable skills [arxiv/2606_30306#56]. This includes adding support for deprecation and updates to prevent workflows from being write-once [arxiv/2606_30306#56]. Frameworks also model skill libraries as self-maintaining software ecosystems with typed contracts to manage technical debt [arxiv/2606_30306#56]. Furthermore, learned-skill methods incorporate verification gates before a skill is written and use score-based maintenance to keep libraries compact [arxiv/2606_30306#56]. Some approaches go further by recording reusable steps as auditable, versioned notebooks with explicit validation gates and fallback paths [arxiv/2606_30306#56].

**X3**

The third family of approaches explicitly addresses the creation and reuse of procedures by transforming successful execution sequences, or "trajectories," into callable skills or reusable workflow abstractions [arxiv/2606_30306#56]. This allows agents to build a library of executable skills over time, enabling them to call upon previously discovered procedures for new tasks [arxiv/2606_30306#56]. In embodied AI, this manifests as growing a library of executable skills over a lifetime, while in web-based applications, it involves inducing high-level workflow abstractions from past interactions [arxiv/2606_30306#56].
These procedural skills are not merely advisory; they represent executable commitments with tangible consequences if misused [arxiv/2606_30306#56, arxiv/2512_16301#31]. This sharpens the focus on actionability and authority, as a skill's continued authorization and its relevance to current conditions before reuse become critical governance questions [arxiv/2606_30306#56]. A key failure mode is "skill drift," where a skill that was once correct becomes outdated as the environment changes, leading to contract violations if not proactively repaired [arxiv/2606_30306#56, arxiv/2512_16301#31]. Runtime auditing is crucial to catch failures that static review might miss, especially when accumulated persistent state can make a seemingly benign skill harmful [arxiv/2606_30306#56].
The development of these skills is an ongoing process, with systems adding support for deprecation and updates to prevent workflows from being write-once [arxiv/2606_30306#56]. Frameworks manage skill libraries as self-maintaining software ecosystems with typed contracts to control technical debt, and some methods incorporate verification gates before a skill is written and maintenance passes to keep libraries compact [arxiv/2606_30306#56]. More advanced approaches record reusable steps as auditable, versioned notebooks with explicit validation gates and fallback paths [arxiv/2606_30306#56].
However, a significant gap exists because skill libraries are often optimized for cheap accumulation and reuse rather than rigorous testing before reuse or rollback after harm. This can lead to an accumulation of executable commitments faster than the mechanisms to verify, scope, or revoke them, resulting in competence and risk growing together [arxiv/2606_30306#56, arxiv/2512_16301#31]. The transferability of refined skills is also an open question, as they can become role-specialized and lose effectiveness off-distribution [arxiv/2606_30306#56].

**X4**

The third family of approaches explicitly addresses the procedural write and reuse of agent capabilities by transforming successful execution sequences into callable skills or reusable workflow abstractions [arxiv/2606_30306#56]. This involves creating a library of executable skills over an agent's lifetime, allowing later tasks to leverage procedures discovered earlier [arxiv/2606_30306#56]. These systems treat the skill store as a primary artifact, with some incorporating features like deprecation and update support to ensure stored workflows are not static [arxiv/2606_30306#56].
These approaches emphasize the lifecycle of skills, moving beyond one-off generation to long-lived, evolving assets [arxiv/2605_27366#3]. A skill is created on demand within the agent's reasoning loop, stored with associated experience and metadata, retrieved when contextually relevant, validated through tests and runtime feedback, and continuously refined [arxiv/2605_27366#3]. This lifecycle includes stages of creation, memory, management, evaluation, and refinement, turning skills into managed, testable, and transferable infrastructure [arxiv/2605_27366#3].
A key aspect of this family is the explicit handling of skill evolution and correction, often involving a cycle of unit testing, evaluation, and refinement to produce updated skills [arxiv/2606_30306#56]. This process aims to address issues like skill drift, where a skill that was correct at the time of writing becomes incorrect as the environment changes [arxiv/2606_30306#56]. To mitigate this, some systems implement validation gates before a skill is written and maintenance passes to keep the library compact [arxiv/2606_30306#56]. Furthermore, some approaches focus on compressing skills by identifying and consolidating redundant structures, aiming for the shortest faithful explanation of the skill's functionality [arxiv/2608_11079#2].
