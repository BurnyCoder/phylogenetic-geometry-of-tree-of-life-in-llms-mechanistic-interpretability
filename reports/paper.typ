#set document(title: "Phylogenetic geometry of animal names in language models", author: "Libor Burian")
#set page(paper: "a4", margin: (x: 1.8cm, y: 1.7cm), numbering: "1")
#set text(size: 10pt, lang: "en")
#set par(leading: 0.6em, justify: false)
#set heading(numbering: none)
#show link: set text(fill: rgb("#244f77"))
#show table: set text(size: 8.5pt)
#set table(inset: (x: 3pt, y: 3pt), stroke: 0.45pt + rgb("#b4c3ce"))
#show figure.caption: set text(size: 8.8pt)
#align(center)[#text(size: 19pt, weight: "bold")[Phylogenetic geometry of animal names in language models]
#linebreak() Libor “Burny” Burian · 26 November 2026]
== Abstract
<abstract>
The 400 English animal names in this panel show within-group agreement with the frozen Open Tree topology. Family-cross-fitted triplet accuracy was 0.473, against one-third chance. At layer 6, accuracy was 0.475 with a family-block 95% interval \[0.427, 0.535\]. The primary result met its family-block decision rule. The intervention archive establishes 11 donor-class answers after editing and 4 before editing in 200 trials. Its registered comparison of genuine answer switches remains unevaluable. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/geometry.json")[Geometry] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/dating.json")[Dating] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/outcomes.json")[Outcomes]
== Question and prior work
<question-and-prior-work>
Does the distance between two English animal names in a text model reflect their evolutionary relationship within familiar animal groups? Goodfire studied bacterial genomes in Evo 2, using angular nearest-neighbour graphs and clade holdouts. Cutinha tested embeddings of mammal descriptions. Park and colleagues studied hierarchical concepts using lexical categories. The contribution here is a within-group rooted-triplet evaluation with named controls and a frozen animal topology. #link("https://www.goodfire.com/research/phylogeny-manifold")[Goodfire] #link("https://github.com/jadencutinha/phylotext-mammals")[Cutinha] #link("https://arxiv.org/abs/2406.01506")[Park et al.]
== Registration and chronology
<registration-and-chronology>
The initial registration preceded the first result commit. The final species freeze, nuisance controls and several analysis details followed viewing exploratory Qwen plots on the earlier panel. The habitat addendum followed the discovery plots and preceded the held-out test. The causal random-subspace count increased after the first favorable rank ordering was known. Git establishes commit order, not the earliest moment a human saw data. The November 26 calculations are retrospective. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/preregistration.md")[Registration and dated deviations] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-analysis-protocol.md")[Retrospective analysis protocol] #link("https://doi.org/10.1073/pnas.1708274114")[Preregistration and prediction]

#figure(image("figures/architecture.png", height: 6.4cm, width: 100%, fit: "contain"), caption: [Data, model readouts, reference annotations and evaluation stages in the original experiment. The reanalysis starts with archived readouts.])

#pagebreak()
== Methods
<methods>
The frozen panel contains 400 species and 258 families in 118 orders. For each template, we subtracted the mean last-name-token vector across species, averaged over templates, and measured cosine distance. A resolved triple has one sister pair whose common ancestor is strictly below the common ancestor of all three. We sampled up to 200 resolved triples per decision node, restricted to one eligible stratum and three distinct final name words, then averaged node means. A unique correct minimum earns one point, a unique wrong minimum zero and a tied minimum one-third. The run originally divided credit among tied winning pairs. The max-T test permutes leaf assignments within strata identically across layers. Family cross-fitting selects a layer in one half and evaluates the other half, then reverses the halves. The primary analysis averages over 20 family splits. The point estimate and its fixed-layer bootstrap target therefore differ. The primary p-value uses the observed maximum and 1,000 maxima under permutation with the plus-one Monte Carlo correction. The per-template mean used the entire name panel, so cross-fitting holds out labels for layer selection while preprocessing has access to all species vectors. Paired percentile intervals use 1,000 draws and reuse the same blocks for all controls. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/preregistration.md")[Primary registration] #link("https://doi.org/10.1186/s13015-020-00175-0")[Rooted triples] #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC6405018/")[Permutation invariance]
== Hypothesis decisions
<hypothesis-decisions>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,auto,auto,auto,auto,),
    table.header([ID], [Hypothesis], [Estimate], [Test p], [Assessment],),
    table.hline(),
    [H1], [Within-group rooted triples], [0.473 cross-fitted], [0.000999], [Criterion met with family blocks],
    [S1], [Global edge-count association], [0.362 at layer 6], [0.0050], [Criterion met],
    [S2], [Family/genus resolution within order], [0.544 at layer 6], [0.0050], [Criterion met],
    [S3], [Unique wrong spelling choice], [0.382 at layer 6], [0.2438], [Criterion unmet],
    [S4], [Unique wrong GloVe choice], [0.372 at layer 6], [0.2438], [Criterion unmet],
    [S5], [Age given rank and MRCA depth], [0.042 at layer 6], [0.1409], [Criterion unmet],
    [C-abl], [Candidate log-probability drop], [0.083 nats], [0.2400], [Criterion unmet (Holm upper bound)],
    [C-swap], [New donor-class answers], [7--11 / 200], [---], [Paired comparison unevaluable],
    [M1], [Marine versus ground mammals], [6 marine / 61 ground], [4.81e-07], [Criterion met],
    [M2], [Aerial versus ground mammals], [0 aerial], [---], [Unevaluable],
    [B1], [Bird water foraging], [ρ = -0.385], [0.000635], [Criterion met],
    [B2], [Bird aerial foraging], [ρ = -0.157], [1.0000], [Criterion unmet],
  )]
  , kind: table
  )

H1 uses the layer maximum. Secondary and habitat rows report Holm-adjusted p-values. Secondary effects are displayed at the H1 layer; their tests cover all layers. C-swap is bounded by the observed marginals. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-scientific-audit.md")[Audit, evidence and interpretation]

#pagebreak()
== Primary geometry and control comparisons
<primary-geometry-and-control-comparisons>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,auto,auto,auto,auto,),
    table.header([Representation], [TA], [Family 95% CI], [Paired Δ, families], [Paired Δ, orders],),
    table.hline(),
    [Qwen3.5-2B], [0.475], [\[0.427, 0.535\]], [---], [---],
    [Random Qwen], [0.382], [\[0.345, 0.445\]], [\[0.014, 0.163\]], [\[-0.014, 0.122\]],
    [Pythia-410m step 0], [0.402], [\[0.354, 0.460\]], [\[0.008, 0.146\]], [\[-0.020, 0.124\]],
    [Character trigrams], [0.332], [\[0.296, 0.368\]], [\[0.085, 0.216\]], [\[0.059, 0.192\]],
    [Input embeddings], [0.354], [\[0.293, 0.396\]], [\[0.074, 0.203\]], [\[0.042, 0.189\]],
    [Name token count], [0.378], [\[0.320, 0.405\]], [\[0.054, 0.180\]], [\[0.021, 0.151\]],
    [Log sitelinks], [0.365], [\[0.300, 0.398\]], [\[0.066, 0.198\]], [\[0.038, 0.146\]],
  )]
  , kind: table
  )

The smallest primary point margin was 0.073 against Pythia-410m step 0. Its family interval is \[0.008, 0.146\], and its order interval is \[-0.020, 0.124\]. A post-hoc check counting each distinct family once per triple, instead of multiplying the three tip multiplicities when a family repeats, gives \[0.003, 0.141\] against the same control. These percentile intervals condition on the modal layers. Both intervals against untrained networks include zero with order blocks. Both the cross-fitted and modal-layer scores exceed the registration\'s 0.40 smallest effect of interest. Model-seed variability and uncertainty from selecting the layer are outside these intervals. The sample contains 11,094 eligible draws at 174 nodes. Exhaustive enumeration of 200,192 qualifying triples gives 0.474. Transporting name-head labels with vectors and recomputing eligibility in every permutation gives p 0.000999. All comparisons concern these checkpoints and this curated name panel. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/geometry.json")[Frozen-data calculation]

#figure(image("figures/viz/h1.png", height: 10cm, width: 100%, fit: "contain"), caption: [Primary accuracy and paired family-block intervals. Each dot is a fixed-layer estimate.])

#pagebreak()
== Secondary hypotheses
<secondary-hypotheses>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,auto,auto,auto,auto,),
    table.header([ID], [Triplets or pairs], [H1-layer effect], [Layer-max p], [Holm p],),
    table.hline(),
    [S1], [All pairs], [0.362], [0.000999], [0.0050],
    [S2], [612], [0.544], [0.000999], [0.0050],
    [S3], [3029], [0.382], [0.1219], [0.2438],
    [S4], [6072], [0.372], [0.1528], [0.2438],
    [S5], [75318], [0.042], [0.0470], [0.1409],
  )]
  , kind: table
  )

The archived within-order analysis included 3,173 triples, TA 0.445. Requiring a recorded family or genus that distinguishes the sister pair retains 612, TA 0.544. Permutations confined to the same order give exploratory layer-max p 0.000999. The paired interval against Pythia step 0 is \[-0.039, 0.277\]. Testing against random assignment and testing for an advantage over an untrained network answer different questions. The archived wrong-choice spelling and GloVe subsets were checked against unique wrong choices. GloVe covered 391 names. A word-vector baseline measures one published co-occurrence representation; it cannot exhaust lexical or ecological cues. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/geometry.json")[Frozen-data calculation]

The partial Spearman value at layer 6 is 0.042. Its fixed-layer p is 0.1199. The global two-sided maximum occurs at layer 5 (p 0.0470, Holm 0.1409). For a claim specifically about layer 6, the layer-adjusted p is 0.8102. The original unpartialled pooled rank statistic at this layer is 0.049 (global max p 0.1319). The amended statistic adjusts for depth of the common ancestor within rank bins using 75,318 eligible pairs. The permutation null assumes exchangeability of leaf assignments within each stratum. Other aspects of topology can still covary with estimated age. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/dating.json")[Frozen-data calculation]

#figure(image("figures/viz/secondary.png", height: 9cm, width: 100%, fit: "contain"), caption: [Trained and untrained estimates on the specified subsets. The spelling-neutral subset is exploratory.])

#pagebreak()
== Interventions and behavior
<interventions-and-behavior>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,auto,auto,auto,auto,),
    table.header([Run], [Random spaces (ablation / swap)], [Donor before / after], [New donor bounds], [Donor prevalence rank p],),
    table.hline(),
    [Original], [20 / 10], [4 / 13], [9--13], [0.0909],
    [Expanded], [99 / 99], [4 / 11], [7--11], [0.0100],
  )]
  , kind: table
  )

The expanded run compares donor prevalence with 99 random subspaces. Its historical Holm p 0.0200 belongs to that substitute endpoint. Net donor prevalence rose by 7/200. Aligned pre/post predictions for the probe and random swaps were not retained in the original result file or intervention transcript. Aggregate comparisons cannot recover the paired event or its random-subspace p-value. C-abl uses candidate-normalized sequence log-probabilities for class words. Its raw rank p is 0.1200; treating the missing registered p as one gives an upper bound of 0.2400 on its Holm p. The rank comparison depends on the random-subspace generator. The original transcript gives a mean captured variance of 0.126 for random spaces, against 0.123 for the probe, in a span of 64 PCs. Only the mean was matched. A class-only control also affected this coarse-class readout. No experiment here isolates a mechanism for within-group ancestry. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/outcomes.json")[Frozen-data calculation] #link("https://arxiv.org/abs/2106.02997")[Interchange interventions]

The model selected the tree-closer answer on 0.567 of 300 within-group two-choice questions, using 170 distinct anchors. The transcript has 9 ties at its printed precision. The ordinary binomial p-value assumes independent trials, so we report the observed fraction descriptively. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/stage3/results.json")[Behavioral observations]
== Conflict stimuli
<conflict-stimuli>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,auto,auto,auto,auto,),
    table.header([Representation], [Tree closer], [Tied], [Other closer], [Missing],),
    table.hline(),
    [Qwen3.5-2B], [11], [0], [25], [0],
    [Random Qwen], [13], [0], [23], [0],
    [Pythia-410m step 0], [13], [0], [23], [0],
    [Trained Pythia-410m], [10], [0], [26], [0],
    [Character trigrams], [9], [13], [14], [0],
    [GloVe], [14], [0], [22], [0],
  )]
  , kind: table
  )

The stimulus set has 79 names and 78 distinct tree IDs. It shares 45 IDs with the primary panel. Each tree choice was recalculated using ancestor depth. Spelling ties previously appeared as choices of the other partner. This hand-picked set reuses anchors and alternative species. Counts describe the items tested; they do not establish population prevalence or equal performance of the networks. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/outcomes.json")[Frozen-data calculation]

#figure(image("figures/viz/causal.png", height: 7cm, width: 100%, fit: "contain"), caption: [The right panel measures post-edit donor prevalence. The missing paired transition outcomes prevent the registered comparison.])

#pagebreak()
== Habitat on held-out species
<habitat-on-held-out-species>
#figure(
  align(center)[#table(
    columns: 4,
    align: (auto,auto,auto,auto,),
    table.header([Prediction], [Species-level p], [Holm (four slots)], [Within-order sensitivity, Holm],),
    table.hline(),
    [M1], [1.2e-07], [4.81e-07], [0.0004],
    [M2], [---], [---], [---],
    [B1], [0.000212], [0.000635], [0.0222],
    [B2], [0.9173], [1.0000], [1.0000],
  )]
  , kind: table
  )

All 159 held-out species have distinct IDs from the discovery set. The panels share 32 families and 18 orders. We recovered oriented PC1 from the discovery vectors and independently reproduced the archived test scores within 2.8e-05 axis units. Marine mammals span 2 orders. The sensitivity test permutes scores within taxonomic orders and retains four Holm slots. Evolutionary dependence can extend within an order. EltonTraits measures foraging stratum, and aquatic words in common names can also contribute. An absent aerial-mammal group supplies no test. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/outcomes.json")[Frozen-data calculation] #link("https://esapubs.org/archive/ecol/E095/178/")[EltonTraits] #link("https://doi.org/10.1086/284325")[Comparative methods]

#figure(image("figures/viz/habitat.png", height: 8cm, width: 100%, fit: "contain"), caption: [Fixed discovery axis projected onto held-out species. The marine comparison is concentrated in two orders.])
== Exploratory geometry, probes and power
<exploratory-geometry-probes-and-power>
The published probe\'s mean held-out all-pair Pearson r was 0.749. A class-centroid-plus-noise input reached 0.833; a shuffled target reached -0.015. Within held-out strata, the archived probe Spearman mean was 0.326 and the independently reconstructed raw-distance comparator was 0.401. The class-blob baseline used empirical stratum centroids constructed on the full panel. The training folds determined the PCA transform. The historical target scale used the maximum tree distance over all species, and the layer was selected using the full panel. Fold predictions and fitted fold weights were not archived. The full-panel released probe has a different training set. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/stage2/results.json")[Original probe outcomes] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/geometry.json")[Frozen-data calculation]

On 23 of 25 layers the three numeric H1a checks pass. The contemporaneous visual description did not identify clear nested arms within groups. The full visual criterion for a nested tree therefore remains unmet. A reference overlay is fitted to the points. The sampled-base-point relative hyperbolicity was 0.322 for the model and 0.188 for class blobs. The unrooted normalized RF was 0.812. Counting splits on the frozen tree gives 388 resolved reference splits. The archived symmetric difference implies 70 shared splits with the binary neighbor-joining tree. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/graph.json")[Frozen-data calculation] The single-layer simulation estimated power from 40 draws per setting and tested one permutation criterion. Estimated detection probability 0.800 at simulated mean TA 0.386 has exact 95% interval \[0.644, 0.909\]. Power for the joint H1 criteria and S5 remains unknown. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/outcomes.json")[Frozen-data calculation] #link("https://aclanthology.org/D19-1275/")[Hierarchical probes]

#pagebreak()
== Ablations, sensitivity and model scale
<ablations-sensitivity-and-model-scale>
At the common-name H1 layer, replacing common names with scientific binomials on the same 10,950 eligible triples changes accuracy from 0.476 to 0.320. Uncentered cosine gives 0.424 and the name-token mean 0.399 on the main subset. These are fixed-layer, descriptive contrasts. Perturbation draws varied triplet samples, prompt subsets and species. The sets of controls examined varied by perturbation. These draws do not estimate between-training-seed variation. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/stage4/results.json")[Ablations] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/stage6/results.json")[Sensitivity]

All 21 archived scale rows were recalculated from ordered name vectors. Identical float32 distance-array digests occur for pythia14b / pythia14b\_step143000; pythia14b\_step0 / pythia14b\_step1. These rows duplicate measurements. Pythia checkpoint points share a training trajectory, and final weights in several configs are referenced by release aliases. The study cannot isolate a causal effect of model size. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/scale.json")[Frozen-data calculation] #link("https://arxiv.org/abs/2304.01373")[Pythia training suite]

#figure(image("figures/viz/scale.png", height: 7cm, width: 100%, fit: "contain"), caption: [Archived checkpoint trajectories. Final models differ in several characteristics. The audit identifies duplicate geometry in the scale series.])
== Limitations and conclusion
<limitations-and-conclusion>
Within-group agreement is an association with a sampled synthetic topology. Edge counts depend on taxon sampling. A rank or a depth in this tree is not elapsed evolutionary time. Selection on English common names and Wikipedia sitelinks, repeated clades, and shared ecological and lexical traits limit generalization. Untrained comparisons refer to individual checkpoints. A confidence interval containing chance or zero leaves a range of plausible effects; equivalence would require an explicit margin. Neither the distance correlations nor the intervention archive identify a phylogeny-specific causal mechanism. #link("https://github.com/OpenTreeOfLife/germinator/wiki/Synthetic-tree-API-v3")[Open Tree] #link("https://doi.org/10.1093/molbev/msac174")[TimeTree] #link("https://doi.org/10.1177/1948550617697177")[Equivalence tests]

#pagebreak()
== Species selection, model readout and related work
<species-selection-model-readout-and-related-work>
Wikidata and English Wikipedia supplied candidates, common names and sitelinks. Selection used broad taxonomic groups and a family cap. Open Tree supplied the induced topology. NCBI-first ranks, with Open Tree fallback, annotate the tips. A deep sister pair can have a longer edge-count path than a shallow outgroup, so the shared ancestor determines the correct pair on each resolved triple. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/data/species.csv")[Frozen panel] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/preregistration.md")[Selection and amendments]

The archived Qwen runs used its text decoder in chat format. Token offsets located each name in several sentence frames. The last name token supplied the primary readout. The embedding layer can already encode a shared final token. Untrained networks, spelling, the input embedding, token count and sitelinks are the primary controls. Static GloVe embeddings and a rank oracle provide additional comparisons. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/src/phylogeo/prompts.py")[Prompt construction] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/SUMMARY.md#stage-1b")[Baseline tables]

Goodfire's bacterial genome comparison motivated graph shortest paths and an angular distance probe with clade holdouts. Engels et al. found calendar circles in language-model features. Modell et al. describe conditions under which distances along a manifold recover conceptual distances. Work on lexical hierarchies and co-occurrence gives a competing explanation for a tree-like arrangement of names. #link("https://www.goodfire.com/research/phylogeny-manifold")[Goodfire] · #link("https://arxiv.org/abs/2405.14860")[Engels et al.] · #link("https://arxiv.org/abs/2505.18235")[Modell et al.] · #link("https://arxiv.org/abs/2406.01506")[Park et al.] · #link("https://arxiv.org/abs/2605.23821")[Nava and Wyart] · #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/literature-review.md")[Literature review]
== Visual checks
<visual-checks>
The early stage compared uncolored and labeled projections with the embedding layer, random weights, calendar words and a simulated Brownian tree. Broad class clusters can produce a short tree overlay. A calendar loop illustrates what an ordered low-dimensional feature looks like in this pipeline. The habitat hypothesis arose after inspection of the discovery plots. The subsequent fixed-axis calculation used different species. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/experiments/01-pca-shape/experiment.md")[Discovery record] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/experiments/06-habitat-addendum/experiment.md")[Held-out habitat]

#figure(image("figures/viz/pca_expected.png", height: 7cm, width: 100%, fit: "contain"), caption: [Reference simulation and animal-name projections. Each condition has its own principal components.])

#pagebreak()
== Probe design, ablations and sensitivity
<probe-design-ablations-and-sensitivity>
The angular probe applies a learned linear map to PCA coordinates. A learned scale converts angular separation into predicted tree distance. The loss combines squared distance error with a reconstruction term. Historical cross-validation assigned whole families to folds. Shuffled targets, random weights and a class-centroid-plus-noise input check different explanations for a good fit. Within a stratum, probe and raw distances can be compared on the same held-out pairs. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/src/phylogeo/probes/flat.py")[Probe implementation] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/experiments/03-geometry-probes-stage2/experiment.md")[Fold design and limitations]

Raw text, per-dimension standardization and within-stratum centering yield estimates close to the main score on this panel. The token readout and common versus scientific name choice give larger changes. Species subsampling produces more variation than triplet resampling in the archived draws. The template runs record model-only scores. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/stage4/results.json")[Ablations] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/stage6/results.json")[Per-draw scores]

#figure(image("figures/viz/probe.png", height: 7.3cm, width: 100%, fit: "contain"), caption: [Held-out probe and class-only comparisons. The raw within-stratum comparator uses the same held-out pairs.])

#figure(image("figures/viz/sensitivity.png", height: 7.3cm, width: 100%, fit: "contain"), caption: [Variation across analysis choices in the archived sensitivity study.])

#pagebreak()
== Appendix and reproducibility
<appendix-and-reproducibility>
At the embedding layer, 221 distinct name pairs have exactly zero angular distance. Preserving those sparse edges and excluding self by row index gives minimum connected k=9 (archived 11) and graph-distance Spearman 0.039 (archived 0.053). At the H1 layer k=3, graph Spearman is 0.251 and angular Spearman is 0.362. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/scientific-verification/graph.json")[Frozen-data calculation]

The public archive includes the frozen names, tree topology, activations, unaltered historical summaries, the independent numerical replay, and the post-hoc protocol. The TimeTree-derived individual node ages are local and must be refetched for a raw replay. The printed behavioral log has lower precision than the model scores. Historical fold-level probe predictions and the aligned intervention trials are missing. Dataset hashes authenticate file identity. They cannot establish that the models had never encountered these species, their names or their relationships in pretraining. The study has no independent held-out language, unseen clade replication of H1, or population sample of training seeds.

The command `uv run --frozen python -m phylogeo.verification --render-only` rebuilds the public output from saved statistics. Omitting the flag replays statistics from archived inputs. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/docs/pipeline.md")[Reproduction instructions] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/reports/reviews/2026-11-26-claim-sources.md")[Claim and source register] #link("https://doi.org/10.1038/nn.2303")[Selection bias] #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/")[Exchangeability]

#figure(image("figures/viz/view_birds.png", height: 8cm, width: 100%, fit: "contain"), caption: [Bird examples from the archived PCA display. The camera angle was optimized for this view. The reference overlay uses the known topology.])

#figure(image("figures/months.png", height: 5cm, width: 100%, fit: "contain"), caption: [Calendar calibration. Months in calendar order, shown in a projection from the original run.])

#pagebreak()
== Output examples
<output-examples>
The query is the species with the most Wikidata sitelinks in each broad group, chosen by the same rule throughout. Each column lists its first archived neighbor. The neural representations use their own selected layers. The tree ranks by deepest shared ancestor, then by edge count. Remaining distance ties use frozen row order. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/src/phylogeo/examples.py")[Selection rule]

#figure(
  align(center)[#table(
    columns: 6,
    align: (auto,auto,auto,auto,auto,auto,),
    table.header([Query], [Reference tree], [Qwen3.5-2B], [Random Qwen], [Character trigrams], [GloVe],),
    table.hline(),
    [axolotl], [fire salamander], [budgerigar], [capybara], [hylotrupes], [aardwolf],
    [diving bell spider], [Redback spider], [great raft spider], [Mediterranean recluse spider], [great diving beetle], [great diving beetle],
    [common ostrich], [greylag goose], [northern giraffe], [North American beaver], [common roach], [common quail],
    [great white shark], [Shortfin mako shark], [tiger shark], [tiger shark], [great white pelican], [great white pelican],
    [Portuguese man o\' war], [Nomura\'s jellyfish], [lion\'s mane jellyfish], [Mediterranean mussel], [European mantis], [Spanish slug],
    [coconut crab], [red king crab], [Japanese spider crab], [velvet crab], [velvet crab], [pea crab],
    [crown-of-thorns starfish], [orange knobby star], [common starfish], [western capercaillie], [common starfish], [common starfish],
    [housefly], [stable fly], [stable fly], [Atlantic cod], [house mouse], [silverfish],
    [lion], [snow leopard], [wolf], [bonobo], [lion\'s mane jellyfish], [lion\'s mane jellyfish],
    [common octopus], [giant Pacific octopus], [giant Pacific octopus], [mimic octopus], [mimic octopus], [common cuttlefish],
    [sea lamprey], [pouched lamprey], [European river lamprey], [Mediterranean mussel], [brook lamprey], [loggerhead sea turtle],
    [goldfish], [Crucian carp], [zebrafish], [garfish], [garfish], [Crucian carp],
    [Komodo dragon], [Asian water monitor], [Gila monster], [greylag goose], [emperor dragonfly], [lion],
  )]
  , kind: table
  )

The full output lists three neighbors per query under every representation, including trained Pythia and step-zero Pythia. #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/RESULTS.md#nearest-neighbor-examples")[Full comparison table] #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability/blob/main/results/qwen35_2b/examples/neighbors.json")[Original examples]

#v(6pt)
*AI statement.* Codex wrote the independent verification code and this revision of the paper to Libor Burian’s specification. The original implementation and manuscript attributed their drafting to Claude Code. Human cleanup and rewriting are in progress.
