#set document(title: "Phylogenetic geometry of the tree of life in an LLM", author: "Libor Burian")
#set page(margin: 2cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#align(center)[
  #text(16pt, weight: "bold")[Do LLMs Represent a Tree of Life? \ Phylogenetic Geometry of Animal Species in Qwen3.5-2B]
  #v(4pt)
  Libor "Burny" Burian #h(1em) · #h(1em) 2026-11-20 \
  #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability")[code, data, preregistration]
]

*Abstract.* Goodfire showed that the DNA model Evo 2 embeds bacterial species on a phylogenetic manifold. We ask whether a
*text* LLM does the same for animals. We extract residual-stream representations of 400 animal species names from
Qwen3.5-2B (8 prompts each) and compare their geometry with the Open Tree of Life under a preregistered protocol with
untrained-model, lexical, co-occurrence, token-count and popularity controls. Within coarse groups and with head nouns
controlled, rooted-triplet accuracy is 0.475 (95% CI 0.427–0.535; chance 0.333;
max-T p < 0.001), above every control; a 10-dimensional linear probe predicts tree distances between held-out families
with r = 0.749 (shuffled-tree control -0.015).
However, the signal is not demonstrably beyond spelling or GloVe co-occurrence on triplets where those baselines fail,
does not track divergence time beyond topology, and where folk taxonomy and phylogeny conflict the representation follows
folk taxonomy (11/36 conflict triplets follow the tree). The top principal axes form class-level arms and
an ecology-like water→land→air gradient rather than a nested tree. LLMs encode a *folk-taxonomic tree*, not an evolution-faithful phylogeny.

= Introduction
Representation geometry is a window into what models learn: days of the week form circles
#footnote[Engels et al. 2024, arXiv:2405.14860.] and DNA models place species on a phylogenetic manifold
#footnote[Goodfire, "Finding the Tree of Life in Evo 2", 2025.]. Text contains phylogeny only indirectly, entangled with folk
categories, shared words, popularity and co-occurrence; whether an LLM's species geometry reflects the tree of life is open
(no peer-reviewed study found; see reports/literature-review.md). Contributions: (i) a preregistered, controlled test with a
topology-only primary metric (node-balanced rooted-triplet accuracy within groups, head nouns disjoint); (ii) Goodfire-style
probe and manifold analyses with the controls their Evo 2 post lacked; (iii) causal and behavioural tests; (iv) evidence that the
learned geometry is folk-taxonomic.

#figure(image("figures/architecture.png", width: 100%), caption: [Pipeline.])

= Methods
*Species.* 400 animals in 13 strata chosen by a fixed rule from Wikidata (English Wikipedia common-name titles, ranked by
sitelinks, ≤ 3 per family, polysemous names excluded), frozen with sha256 before extraction. *Tree.* Open Tree of Life induced
topology (CC0); NCBI-first Linnaean ranks; TimeTree 5 divergence times (secondary). *Representations.* Qwen3.5-2B (post-trained,
bf16, batch size 1), residual stream at the last name token for 8 chat-formatted templates, all 25 layers; per-template mean
centering, template average, cosine distance. *Primary metric.* For triplets (a, b | c) resolved by the tree, correct if
d(a,b) < min(d(a,c), d(b,c)); node-balanced mean over decision nodes; restricted to triplets inside one stratum with three
distinct head nouns (TA#sub[within]). Significance by max-T permutation over layers (within-stratum label shuffles), point
estimate by family-blocked cross-fitting, CI by clade-block (family) bootstrap; H1 additionally requires beating random-init
Qwen, Pythia step 0, character 3-grams, the input embedding, token count and popularity. A power simulation on the frozen tree
gave a minimum detectable TA#sub[within] of 0.386 at 80% power.

= Results
== What the top principal components show (stage 1a)
#figure(image("figures/spot.png", width: 100%), caption: [Layer 21, PC1–PC2; one stratum highlighted per panel. The main axis
runs from aquatic groups to flyers (insects, birds), cutting across phylogeny.])
#grid(columns: 2, gutter: 6pt, figure(image("figures/months.png", width: 90%), caption: [Positive control: months form a circle.]),
  figure(image("figures/brown.png", width: 100%), caption: [What a true tree looks like (Brownian motion on the reference tree).]))
Days and months form circles in calendar order (pipeline check). Species form class-level arms (layers 4–5) and a curved,
ecology-like band in late layers; within-class order clustering exceeds the lexical (layer 0) and random-init references, but no
nested within-class branching is visible in the top 3 PCs.

== Triplet accuracy and controls (stage 1b, confirmatory)
#figure(table(columns: 3, align: (left, center, center),
    [*representation*], [*TA#sub[within]*], [*95% CI*],
    [*Qwen3.5-2B (main)*], [0.475], [0.427–0.535],
    [random-init Qwen3.5-2B], [0.382], [0.345–0.445],
    [Pythia-410m step 0 (untrained)], [0.402], [0.354–0.460],
    [Pythia-410m (trained)], [0.423], [0.378–0.500],
    [character 3-grams], [0.332], [0.296–0.368],
    [input embedding (layer 0)], [0.354], [0.293–0.396],
    [GloVe 300d (co-occurrence)], [0.395], [0.356–0.454],
    [name token count], [0.383], [0.326–0.413],
    [log Wikipedia sitelinks], [0.365], [0.300–0.398],
    [class oracle], [0.333], [0.333–0.333],
    [Linnaean rank oracle], [0.644], [0.617–0.687],
    [tree oracle (upper bound)], [1.000], [1.000–1.000],
  ), caption: [Within-stratum, head-noun-disjoint rooted-triplet accuracy (chance 0.333); models at their cross-fitted layer.])
H1 is supported: cross-fitted TA#sub[within] 0.473 (layer 6),
max-T p < 0.001, and every paired-bootstrap difference to the required controls excludes zero.
#figure(table(columns: 5, [*id*], [*statistic*], [*value*], [*p*], [*Holm p*], [S1], [global Mantel ρ (cosine vs path length)], [0.362], [< 0.001], [= 0.005],
    [S2], [TA within orders (head-noun-disjoint)], [0.445], [< 0.001], [= 0.005],
    [S3], [TA on lexical-discordant triplets], [0.382], [= 0.120], [= 0.360],
    [S4], [TA on co-occurrence-discordant triplets], [0.372], [= 0.153], [= 0.360],
    [S5], [partial ρ(distance, age | topology)], [0.042], [= 0.120], [= 0.360],), caption: [Secondary hypotheses.])
#figure(image("figures/ta.png", width: 85%), caption: [TA#sub[within] by layer for the model and controls.])
*Conflict triplets.* The representation picks the tree-closer partner in 11/36 triplets
(binomial p = 0.029), e.g. blue whale is closer to whale shark than to hippopotamus.

== A linear phylogeny subspace (stage 2)
#figure(table(columns: 3, [*probe input / target*], [*held-out-family r*], [*train r*], [model], [0.749 ± 0.010], [0.959],
    [shuffled\_tree], [-0.015 ± 0.013], [0.562],
    [random\_init], [0.016 ± 0.013], [0.519],
    [glove], [0.405 ± 0.005], [0.845],),
  caption: [Flat-subspace probe (k = 10), 5-fold family holdout, 5 seeds.])
Geodesic distance on the kNN graph did not improve on angular distance; δ-hyperbolicity of the model (0.322)
lies between a spectrum-matched Gaussian (0.460) and pure class clusters
(0.188), so it does not discriminate trees from clusters; a neighbour-joining tree from the
representations has normalized RF 0.812 to the reference (permutation null 0.988).

= Discussion
The model's species geometry agrees with the tree of life within groups beyond untrained, lexical, token-count and popularity
controls, and a linear subspace generalizes to unseen families. But on triplets where spelling or co-occurrence statistics
are wrong the model is not reliably right, the geometry tracks rank structure rather than divergence time, and where intuition
and phylogeny conflict it follows intuition. The dominant axes mix taxonomy with ecology. This is the geometry one expects from
text: a folk-taxonomic tree. *Limitations:* one 2B model (scale in stage 5), English common names, a single tree source, a
correlational primary metric, and S3/S4 subsets that are harder by construction. *Related work* is reviewed in
reports/literature-review.md (Goodfire Evo 2 and neural geometry; Engels et al.; Park et al.; Gurnee & Tegmark; Modell et al.;
Nava & Wyart; Hewitt & Manning/Liang; phylotext-mammals).

= Conclusion
Qwen3.5-2B encodes a tree of animals, but it is the tree people talk about, not the one evolution built.

#v(8pt)
#line(length: 100%)
*AI statement.* This codebase and paper were written by Claude Code (Anthropic) executing a specification provided by a human
(Libor Burian); human cleanup and rewriting are in progress. All numbers are generated from the repository's results files.
