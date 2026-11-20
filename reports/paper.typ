#set document(title: "Phylogenetic geometry of the tree of life in an LLM", author: "Libor Burian")
#set page(margin: 2cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show figure: set block(breakable: true)
#show link: set text(fill: rgb("#2a78d6"))
#align(center)[
  #text(16pt, weight: "bold")[Do LLMs Represent a Tree of Life? \ Phylogenetic Geometry of Animal Species in Qwen3.5-2B]
  #v(4pt)
  Libor "Burny" Burian #h(1em) · #h(1em) 2026-11-20 \
  #link("https://github.com/BurnyCoder/phylogenetic-geometry-of-tree-of-life-in-llms-mechanistic-interpretability")[code, data, preregistration, interactive results]
]

*Abstract.* Goodfire reported that the DNA model Evo 2 places bacterial species on a curved manifold whose geometry tracks
phylogeny [#link(<ref-goodfire_evo2>)[1]]. We test whether a *text* LLM does the same for animals. We extract residual-stream
representations of 400 animal species names from Qwen3.5-2B (8 prompts each) and compare their geometry with the Open Tree of
Life under a preregistered protocol with untrained-network, lexical, token-count, popularity and co-occurrence controls. Inside
coarse groups and with shared head nouns excluded, rooted-triplet accuracy is 0.475 (95% CI
0.427–0.535; chance 0.333; max-T p \= 0.001), above every required control (primary
hypothesis supported), though by a modest margin and clearly only in mammals and birds. The signal is not
significant on triplets where spelling or GloVe co-occurrence points the wrong way, where the model equals untrained networks;
it is not detected in divergence times beyond topology; and on phylogeny-versus-folk conflict triplets the geometry rarely
follows the tree, as for untrained networks. A Goodfire-style 10-dimensional probe predicts tree distances between held-out
families with r = 0.749, but a class-only input does at least as well
(0.833), so this correlation mainly reflects coarse
groups. The top late-layer component orders animals from aquatic to terrestrial, confirmed on held-out species. Qwen3.5-2B thus
encodes modest within-group tree order entangled with folk categories and ecology, rather than a phylogenetic manifold.

= Introduction
*Context.* Representation geometry is a window into what models learn. Days of the week and months form circles in LLMs
[#link(<ref-engels>)[3]], concept manifolds can be explained by the metric structure of the underlying features [#link(<ref-modell>)[4]], and
Goodfire found that Evo 2 embeds the bacterial tree of life so that geodesic distance on a kNN graph tracks phylogenetic distance
[#link(<ref-goodfire_evo2>)[1]][#link(<ref-goodfire_geom>)[2]]. *Motivation.* Text carries phylogeny only indirectly, entangled with folk
categories ("fish", "bugs"), shared words ("tiger shark"), popularity and co-occurrence; co-occurrence alone can produce
hierarchical splits [#link(<ref-nava>)[8]]. Whether a text LLM's species geometry reflects the tree of life is therefore not obvious,
and global correlations can be inflated by class-level clusters. We found no peer-reviewed study of this question; the closest
work found sentence embeddings uncorrelated with the mammal tree [#link(<ref-phylotext>)[12]].

*Contributions.* (i) A preregistered, controlled test with a topology-only primary metric: node-balanced rooted-triplet accuracy
inside coarse groups with head nouns disjoint, against a fixed set of untrained-network and nuisance controls. (ii) Goodfire-style
probe and manifold analyses with family-held-out evaluation, a shuffled-tree control and a class-only control; the last shows that
the probe's held-out correlation mostly reflects coarse groups. (iii) Behavioural judgments and causal ablation and interchange
of the probe subspace against variance-matched random subspaces and a class-only subspace.
(iv) An out-of-sample test of a post-hoc observation (a habitat axis), registered before new species were selected.
(v) Scale, training-dynamics and sensitivity analyses on the same metric.

#figure(image("figures/architecture.png", width: 100%), caption: [Pipeline.])

= Background and related work
*Linear and hierarchical concept geometry.* Park et al. [#link(<ref-park>)[6]] show that hierarchically related concepts are orthogonal
in a causal inner product and that categorical concepts form simplices; Gurnee and Tegmark [#link(<ref-gurnee>)[7]] recover linear maps of
space and time. *Manifolds.* Engels et al. [#link(<ref-engels>)[3]] find circular features; Modell et al. [#link(<ref-modell>)[4]] predict that
geodesic distance on representation manifolds tracks feature distance, explicitly including tree metrics; Goodfire's manifold
steering moves along such manifolds [#link(<ref-wurgaft>)[5]]. *Trees and hyperbolic geometry.* Structural probes recover parse-tree
distances [#link(<ref-hewitt>)[9]]; hyperbolic embeddings represent hierarchies compactly [#link(<ref-nickel>)[10]], and Gromov δ-hyperbolicity
measures tree-likeness of a metric [#link(<ref-khrulkov>)[11]]. *Phylogeny in models.* Evo 2 encodes the bacterial tree
[#link(<ref-goodfire_evo2>)[1]]; for text models, evidence is sparse and mostly from small projects [#link(<ref-phylotext>)[12]].

= Methods
*Species and trees.* 400 animal species in 13 strata, chosen by a fixed rule from Wikidata (English Wikipedia common-name titles
ranked by sitelinks, at most 3 per family, polysemous names excluded), frozen with sha256 before extraction. The reference
topology is the Open Tree of Life induced subtree [#link(<ref-opentree>)[13]], with NCBI-first Linnaean ranks; divergence times come from
TimeTree 5 [#link(<ref-timetree>)[14]] (secondary analysis only). *Representations.* Qwen3.5-2B [#link(<ref-qwen>)[16]] (post-trained, bf16, batch
size 1), residual stream at the last token of the name for 8 chat-formatted templates, all 25 layers; each template is
mean-centred over species and the templates are averaged; distance is cosine. *Primary metric.* A triplet (a, b | c) is resolved
when the most recent common ancestor of a and b lies strictly below that of all three; it is correct if d(a, b) < min(d(a, c),
d(b, c)). Accuracy is averaged per decision node (at most 200 triplets per node) and then over nodes, restricted to triplets inside
one stratum whose three names have distinct head nouns (TA#sub[within]). *Inference.* Max-T permutation over layers (species
shuffled within strata, 1000 permutations), family-blocked 2-fold cross-fitting of the layer (20 splits), family-block bootstrap
CIs (1000 draws); exploratory tests are BH-corrected over the registered family. H1 additionally requires paired-bootstrap differences above zero against random-init Qwen3.5-2B, Pythia-410m
step 0 [#link(<ref-pythia>)[17]], character 3-grams, the input embedding, name token count and log Wikipedia sitelinks. GloVe
[#link(<ref-glove>)[18]] defines the co-occurrence-discordant subset (S4). A power simulation on the frozen tree gave a minimum detectable
TA#sub[within] of 0.386 at 80% power (40 simulations per cell, so this is a rough estimate). *Probe.* Following Goodfire, z = W(x − b) with
d̂ = β·arccos(cos(z#sub[1], z#sub[2])), PCA-100 fitted inside each fold, whole families held out. *Causal tests.* Forward hooks
replace the residual stream after the chosen decoder layer. *Rigor.* All hypotheses, metrics and tests were preregistered before
extraction; every deviation is timestamped in the repository. Practices follow Lones [#link(<ref-lones>)[19]].

= Results
== What the top principal components show (stage 1a, exploratory)
#figure(image("figures/spot.png", width: 100%), caption: [Layer 21, PC1–PC2; one stratum highlighted per panel. The main axis runs from aquatic groups to terrestrial ones, cutting across phylogeny.])

#grid(columns: 2, gutter: 6pt, figure(image("figures/months.png", width: 90%), caption: [Positive control: months form a circle.]),
  figure(image("figures/brown.png", width: 100%), caption: [What a true tree looks like (Brownian motion on the reference tree).]))
Days and months form circles in calendar order, so the pipeline recovers known geometry. Class purity of 5-nearest neighbours in PC1–3 peaks at layer 4 (0.738 vs permutation null 0.122). Species form class-level arms
in early layers and a curved band in late layers; no nested within-class branching is visible in the top three components, which
hold only a small share of the variance.

== Triplet accuracy and controls (stage 1b, confirmatory)
#figure(table(columns: 4, align: left, stroke: 0.4pt + luma(180),
    [*representation*], [*role*], [*TA\_within*], [*95% CI*],
    [*Qwen3.5-2B (main method)*], [method], [0.475], [0.427–0.535],
    [random-init Qwen3.5-2B], [control], [0.382], [0.345–0.445],
    [Pythia-410m step 0 (untrained)], [control], [0.402], [0.354–0.460],
    [Pythia-410m (trained)], [baseline], [0.423], [0.378–0.500],
    [character 3-grams], [control], [0.332], [0.296–0.368],
    [input embedding (layer 0)], [control], [0.354], [0.293–0.396],
    [name token count], [control], [0.383], [0.326–0.413],
    [log Wikipedia sitelinks], [control], [0.365], [0.300–0.398],
    [GloVe 300d (co-occurrence)], [baseline], [0.395], [0.356–0.454],
    [class oracle], [oracle], [0.333], [0.333–0.333],
    [Linnaean-rank oracle], [oracle], [0.644], [0.617–0.687],
    [tree oracle (upper bound)], [oracle], [1.000], [1.000–1.000],
    [ablation: raw cosine (no centering)], [ablation], [0.424], [–],
    [ablation: z-scored dimensions], [ablation], [0.480], [–],
    [ablation: stratum-centered], [ablation], [0.483], [–],
    [ablation: mean over name tokens], [ablation], [0.399], [–],
    [ablation: raw text (no chat template)], [ablation], [0.482], [–],
    [ablation: scientific binomials (genus-split triplets)], [ablation], [0.332], [–],
  ), caption: [Within-stratum, head-noun-disjoint rooted-triplet accuracy (chance 0.333). Models at their cross-fitted layer; ablations at layer 6. CIs from a family-block bootstrap.])

H1 is supported: cross-fitted TA#sub[within] 0.473 (modal layer 6), max-T p
\= 0.001 (the permutation floor), and every paired-bootstrap difference to the required controls excludes zero.
The margin over the strongest required control is modest, and the signal is clear only in mammals and birds, which hold about
half of the within-stratum decision nodes (Table below). The untrained networks are themselves above chance; one hypothesis is
that name form and length cluster by family, but they stay above chance even on triplets with no shared token.
#figure(table(columns: 5, align: left, stroke: 0.4pt + luma(180),
    [*id*], [*statistic*], [*value*], [*max-T p*], [*Holm p*],
    [S1], [global Mantel ρ (cosine vs path length)], [0.362], [0.001], [0.005],
    [S2], [TA within orders (head-noun-disjoint)], [0.445], [0.001], [0.005],
    [S3], [TA on lexical-discordant triplets], [0.382], [0.120], [0.360],
    [S4], [TA on co-occurrence-discordant triplets], [0.372], [0.153], [0.360],
    [S5], [partial ρ(distance, age | topology)], [0.042], [0.120], [0.360],
  ), caption: [Confirmatory secondary hypotheses (Holm family of 5).])

 S1 is not specific to phylogeny: the class oracle alone reaches a Mantel r of 0.508 and log popularity alone 0.296. S2 passes its registered test against chance, but the untrained random-init Qwen reaches the same value; on the
spelling- and GloVe-discordant subsets (S3, S4) the model also equals the untrained networks. On triplets where spelling has no
preference, the model exceeds them (post hoc):
#figure(table(columns: 5, align: left, stroke: 0.4pt + luma(180),
    [*subset*], [*triplets*], [*Qwen3.5-2B*], [*qwen35\_2b\_random (diff. CI)*], [*pythia410m\_step0 (diff. CI)*],
    [S2: within orders], [3173], [0.445], [0.451 (-0.104, +0.132)], [0.424 (-0.074, +0.162)],
    [S3: spelling-discordant], [3034], [0.382], [0.379 (-0.072, +0.100)], [0.375 (-0.069, +0.122)],
    [S4: GloVe-discordant], [6072], [0.372], [0.367 (-0.076, +0.090)], [0.390 (-0.083, +0.078)],
    [spelling has no preference (lexically neutral)], [6306], [0.484], [0.350 (+0.047, +0.209)], [0.371 (+0.036, +0.184)],
  ), caption: [Post hoc: the model vs the untrained controls on the same triplet subsets (paired family-block bootstrap CI of the difference).])

#figure(table(columns: 4, align: left, stroke: 0.4pt + luma(180),
    [*stratum*], [*decision nodes*], [*TA\_within*], [*95% CI*],
    [birds], [45], [0.504], [0.373–0.623],
    [mammals], [43], [0.533], [0.435–0.645],
    [ray-finned fish], [29], [0.467], [0.311–0.585],
    [insects], [20], [0.343], [0.249–0.542],
    [reptiles], [10], [0.514], [0.316–0.779],
    [molluscs], [8], [0.368], [0.125–0.912],
    [amphibians], [4], [0.278], [0.000–1.000],
    [cartilaginous fish], [4], [0.600], [0.000–1.000],
    [echinoderms], [4], [0.571], [0.000–1.000],
    [arachnids], [3], [0.124], [0.000–0.261],
    [crustaceans], [3], [0.640], [0.068–1.000],
    [cnidarians], [1], [0.440], [0.000–1.000],
  ), caption: [Post hoc: TA#sub[within] per stratum at layer 6 (family-block bootstrap CIs).])

#figure(image("figures/ta.png", width: 85%), caption: [TA#sub[within] by layer for the model and controls.])

*Conflict triplets.* The representation picks the tree-closer partner in 11/36 triplets (binomial p
\= 0.029); for example, the blue whale is closer to the whale shark than to the hippopotamus. The untrained
controls behave the same (qwen35\_2b\_random 13/36 (McNemar vs the model p \= 0.804); pythia410m\_step0 13/36 (McNemar vs the model p \= 0.774); pythia410m 10/36 (McNemar vs the model p \= 1.000); char3 9/36 (McNemar vs the model p \= 0.804); glove 14/36 (McNemar vs the model p \= 0.549)), and the triplets share anchors and partners, so this does not show a learned folk
or ecological taxonomy; it shows only that the geometry does not follow the tree in these hand-picked cases. Exploratory p-values are corrected with BH-FDR over the 176 registered exploratory tests; q-values are listed in the repository.

== Flat-subspace probe and manifold tests (stage 2, exploratory)
#figure(table(columns: 3, align: left, stroke: 0.4pt + luma(180),
    [*probe input / target*], [*held-out-family r*], [*train r*],
    [model], [0.749 ± 0.010], [0.959],
    [shuffled tree], [-0.015 ± 0.013], [0.562],
    [random init], [0.016 ± 0.013], [0.519],
    [glove], [0.405 ± 0.005], [0.845],
    [class blob], [0.833 ± 0.006], [0.958],
  ), caption: [Flat-subspace probe (k = 10), 5-fold family holdout, 5 seeds (mean ± SD).])

#figure(image("figures/probe.png", width: 100%), caption: [Held-out-family performance of the flat probe and its controls: all pairs (left) and within strata (right). Error bars are SDs over probe seeds.])

The headline held-out correlation mainly measures coarse-group separation. A predictor that knows only the 13 strata reaches r = 0.871 with path length over all pairs, and the same probe trained on a class-blob input (each species at its stratum centroid plus matched noise, no within-stratum tree) reaches held-out r = 0.833. Within strata (mean of per-stratum Spearman on held-out pairs) the model's probe reaches 0.326, the class-blob input 0.008, and the raw angular distances without any probe 0.290: the representation carries some within-stratum order, but the probe adds little to it. kNN-geodesic distance does not order triplets better than angular distance. Gromov δ#sub[rel] of the model
(0.322) lies between a spectrum-matched Gaussian (0.460) and pure
class clusters (0.188), so it does not separate trees from clusters. A neighbour-joining tree
from the representations has normalized Robinson–Foulds distance 0.812 to the reference (permutation null
0.988; with 20 permutations the smallest attainable p is 0.048). Against NJ trees built from class-blob inputs (0.954, 20 draws) the model's tree is closer to the reference.

== Behaviour and causal tests (stage 3)
*Behaviour (exploratory).* Asked which of two animals is more closely related to a third (answer letters A/B scored by log-probability, both orders averaged), the model chose the tree-closer animal in 56.7% of 300 within-stratum triplets (binomial p \= 0.024); on the same triplets its layer-6 representation picks the tree-closer animal in 61.3%, and the two agree in 63.3%. On the conflict triplets it chose the tree-closer animal in 11/36 (binomial p \= 0.029). Class-word accuracy was 90.9% over 385 species.

== Ablations (stage 4, exploratory)
Single-change ablations are listed in the evaluation table. Normalization (z-scoring, stratum-centering), prompt format (no chat template) and including polysemous names leave TA#sub[within] essentially unchanged; raw cosine and the mean over name tokens are lower. With scientific binomials on genus-split, head-noun-disjoint triplets, TA is 0.320 at layer 6 (best layer 0.380) versus 0.476 for common names on the same triplets: much weaker, and only in late layers. Within strata (mean of per-stratum Spearman on held-out pairs), the k = 10 probe reaches 0.353, the shuffled-tree probe 0.000 and the class-blob input 0.003. An earlier version pooled all within-stratum pairs into one Spearman; that metric is inflated by strata differing in typical distance and is not used.
#figure(table(columns: 4, align: left, stroke: 0.4pt + luma(180),
    [*probe dimension k*], [*angular*], [*Euclidean*], [*Poincaré*],
    [2], [0.631], [0.699], [0.675],
    [3], [0.716], [–], [–],
    [5], [0.737], [0.730], [0.721],
    [10], [0.750], [0.736], [0.728],
    [20], [0.753], [–], [–],
  ), caption: [Held-out-family Pearson r of the flat probe by dimension and metric (3 seeds).])

== Habitat axis on held-out species (Addendum A, confirmatory)
The stage-1a observation that the top late-layer axis orders strata from aquatic to terrestrial groups was registered as a directional hypothesis before any held-out species was selected. On 79 held-out mammals and 80 held-out birds (EltonTraits foraging strata [#link(<ref-elton>)[15]]), projected onto the fixed layer-21 PC1 axis of the primary set: M1 (marine mammals below ground-dwelling ones): median -0.581 vs ground 1.676 (n = 6), one-sided p \= 1.2e-07, Holm p \= 3.6e-07; M2 (bats above ground-dwelling mammals): not evaluable (n = 0 < 5); B1 (more water foraging, lower score; birds): ρ = -0.385, one-sided p \= 2.1e-04, Holm p \= 4.2e-04; B2 (more aerial foraging, higher score; birds): ρ = -0.157, one-sided p \= 0.917, Holm p \= 0.917. The aquatic end is confirmed and the aerial end is not. Caveats: the marine mammals are few and their names carry aquatic cues (whale, seal, porpoise, sea otter), water foraging in birds is phylogenetically clustered, and the axis holds 7.5% of the variance at that layer. The registered reading is that the geometry mixes ecology with taxonomy.

== Other models: scale and training dynamics (stage 5, descriptive)
#figure(table(columns: 7, align: left, stroke: 0.4pt + luma(180),
    [*model*], [*params*], [*layers*], [*max TA\_within*], [*cross-fitted TA\_within*], [*max-T p*], [*Mantel r*],
    [pythia70m], [0.07B], [6], [0.395], [0.379], [0.010], [0.062],
    [pythia160m], [0.16B], [12], [0.394], [0.373], [0.030], [0.048],
    [pythia410m], [0.41B], [24], [0.423], [0.416], [0.005], [0.312],
    [pythia1b], [1.00B], [16], [0.412], [0.419], [0.010], [0.368],
    [pythia14b], [1.40B], [24], [0.461], [0.446], [0.005], [0.376],
    [pythia28b], [2.80B], [32], [0.474], [0.470], [0.005], [0.420],
    [qwen35\_08b], [0.85B], [24], [0.449], [0.432], [0.005], [0.347],
    [qwen3\_17b], [1.70B], [28], [0.475], [0.465], [0.005], [0.349],
    [qwen35\_2b], [2.27B], [24], [0.475], [0.480], [0.005], [0.362],
  ), caption: [Same species, triplets and metric for every model (199 permutations; p floor 0.005).])
#figure(image("figures/scale.png", width: 75%), caption: [Cross-fitted TA#sub[within] vs parameter count.])
#figure(image("figures/dynamics.png", width: 75%), caption: [Cross-fitted TA#sub[within] over Pythia-1.4B training.])

== Sensitivity (stage 6)
#figure(table(columns: 4, align: left, stroke: 0.4pt + luma(180),
    [*perturbation*], [*repetitions*], [*TA\_within*], [*margin over best control (min)*],
    [triplet sampling seed], [20], [0.474 ± 0.002 (range 0.467–0.476)], [0.070],
    [80% species subsample], [20], [0.475 ± 0.019 (range 0.437–0.512)], [0.044],
    [template halves (4 of 8)], [20], [0.475 ± 0.004 (range 0.469–0.482)], [–],
    [single template], [8], [0.467 ± 0.006 (range 0.456–0.475)], [–],
    [cross-fit family-split seed], [20], [0.472 ± 0.004 (range 0.466–0.478)], [–],
  ), caption: [Spread of TA#sub[within] at layer 6; margin = model minus the best untrained or nuisance control on the same draw.])
Flat probe over 20 fresh seeds: held-out r 0.750 ± 0.012 (range 0.728–0.771).
#figure(image("figures/sens.png", width: 85%), caption: [TA#sub[within] under each perturbation.])

= Discussion
*Claims and evidence.* (1) *Systematic, correlational:* Qwen3.5-2B's species geometry agrees with the tree of life within coarse
groups beyond untrained-network, lexical, token-count and popularity controls; across all stage-6 perturbations TA#sub[within] stays at or above 0.437 and the smallest margin over the best control is 0.044. The margin is modest and the
signal is concentrated in mammals and birds. Because ecology and body size are phylogenetically conserved and were not
controlled, the precise claim is agreement with the tree, not phylogenetic knowledge per se. (2) *Negative for the probe:* the
Goodfire-style flat probe generalizes to held-out families, but a class-only input does at least as well, and within strata it
adds little over the raw distances; we therefore do not claim that a linear subspace "carries the tree". (3) *Negative:* the
order is not significant on triplets where spelling or GloVe co-occurrence disagree with the tree, where the model equals
untrained networks, so we cannot claim it goes beyond such statistics; S2 is matched by an untrained network; no divergence-time
signal is detected once topology is controlled; on conflict triplets the geometry does not follow the tree, as for untrained
networks. (4) *Out-of-sample confirmation of a post-hoc pattern:* the top late-layer component separates aquatic from terrestrial
animals. Causal claims are limited to what the stage-3 tests show above.

*Implications.* For interpretability, global distance correlations with a taxonomy are weak evidence of a phylogenetic
representation: class clusters, ecology and naming conventions all contribute, and untrained networks already reach above-chance
values on several metrics. Topology-only, within-group metrics with untrained and nuisance controls separate these. Distance
probes evaluated on all held-out pairs should be compared with a class-only input, because coarse groups alone can produce high
correlations. For the analogy with Evo 2, a text model's species geometry is better described as a mixture of folk taxonomy,
ecology and modest phylogenetic order than as a tree-shaped manifold.

= Limitations
One main 2B-parameter model (stage 5 adds smaller and older models, not larger ones); English common names only (binomials carry
much weaker, late-layer structure); one tree source (Open Tree) and NCBI ranks, both imperfect for deep splits, and edge-count
path length depends on how densely each clade is sampled; GloVe is a proxy for co-occurrence in the model's own training data;
S3 and S4 select triplets where simple statistics fail and are therefore adversarial by construction; the within-group signal
rests mostly on mammals and birds; ecological and body-size similarity is not controlled; probes and interventions use a single
layer chosen by cross-fitting; the conflict triplets are few and share anchors; the habitat addendum has only 6 marine mammals,
whose names carry aquatic cues, and no bats.

= Conclusion
Within this preregistered design, Qwen3.5-2B's species geometry agrees with the tree of life inside coarse groups by a modest,
robust margin over untrained networks and name statistics, clearly so for mammals and birds. The geometry is otherwise dominated
by coarse folk categories and ecology, the Goodfire-style probe mostly recovers the coarse groups, and the evidence does not show
phylogenetic information beyond what lexical and co-occurrence statistics could supply.

= Output examples
Nearest neighbours (3) of the most-linked species of every stratum (query rule fixed in advance), under the reference tree, the model, the random-init control, character 3-grams and GloVe.
#figure(table(columns: 6, align: left, stroke: 0.4pt + luma(180),
    [*query*], [*tree*], [*qwen35\\\_2b*], [*qwen35\\\_2b\\\_random*], [*char3*], [*glove*],
    [axolotl], [fire salamander, smooth newt, alpine newt], [budgerigar, fire salamander, Indonesian coelacanth], [capybara, Christmas Island red crab, western capercaillie], [hylotrupes, colossal squid, Colorado potato beetle], [aardwolf, cockchafer, olm],
    [diving bell spider], [Redback spider, Darwin's bark spider, giant house spider], [great raft spider, raft spider, Darwin's bark spider], [Mediterranean recluse spider, great raft spider, Redback spider], [great diving beetle, raft spider, zebra spider], [great diving beetle, Sydney funnel-web spider, great raft spider],
    [common ostrich], [greylag goose, western capercaillie, greater flamingo], [northern giraffe, common moorhen, greater flamingo], [North American beaver, goliath birdeater, European hare], [common roach, common kestrel, common octopus], [common quail, common kestrel, common raven],
    [great white shark], [Shortfin mako shark, porbeagle, basking shark], [tiger shark, blue shark, frilled shark], [tiger shark, goblin shark, blue shark], [great white pelican, white stork, whale shark], [great white pelican, blue shark, bull shark],
    [Portuguese man o' war], [Nomura's jellyfish, lion's mane jellyfish, blue jellyfish], [lion's mane jellyfish, blue jellyfish, Nomura's jellyfish], [Mediterranean mussel, osprey, deathstalker], [European mantis, Hermann's tortoise, Chinese mitten crab], [Spanish slug, African bush elephant, house cricket],
    [coconut crab], [red king crab, Japanese spider crab, heikegani], [Japanese spider crab, red king crab, Christmas Island red crab], [velvet crab, pea crab, Chinese alligator], [velvet crab, pea crab, crab louse], [pea crab, mangrove horseshoe crab, crab louse],
    [crown-of-thorns starfish], [orange knobby star, common sunstar, leather star], [common starfish, sunflower sea star, lion's mane jellyfish], [western capercaillie, sand lizard, great diving beetle], [common starfish, garfish, silverfish], [common starfish, orange knobby star, red king crab],
    [housefly], [stable fly, six-spot burnet, codling moth], [stable fly, oriental cockroach, German cockroach], [Atlantic cod, desert locust, mugger crocodile], [house mouse, house sparrow, house cricket], [silverfish, ruffe, cockchafer],
    [lion], [snow leopard, cheetah, meerkat], [wolf, cheetah, polar bear], [bonobo, Eurasian woodcock, American cockroach], [lion's mane jellyfish, emperor scorpion, grey heron], [lion's mane jellyfish, leopard seal, golden eagle],
    [common octopus], [giant Pacific octopus, mimic octopus, vampire squid], [giant Pacific octopus, mimic octopus, European squid], [mimic octopus, loggerhead sea turtle, great crested grebe], [mimic octopus, common ostrich, common toad], [common cuttlefish, mimic octopus, common starfish],
    [sea lamprey], [pouched lamprey, brook lamprey, European river lamprey], [European river lamprey, brook lamprey, pouched lamprey], [Mediterranean mussel, European river lamprey, northern krill], [brook lamprey, pouched lamprey, European river lamprey], [loggerhead sea turtle, brook lamprey, hawksbill sea turtle],
    [goldfish], [Crucian carp, common carp, grass carp], [zebrafish, siamese fighting fish, garfish], [garfish, milkfish, ocean sunfish], [garfish, European goldfinch, milkfish], [Crucian carp, milkfish, chimpanzee],
    [Komodo dragon], [Asian water monitor, common slow worm, Gila monster], [Gila monster, frilled lizard, marine iguana], [greylag goose, Antarctic krill, chimpanzee], [emperor dragonfly, lion, koala], [lion, viviparous lizard, frilled lizard],
  ), caption: [Nearest neighbours per representation.])
Fraction of each species' 5 nearest neighbours in the same taxonomic order: qwen35\_2b 0.442, qwen35\_2b\_random 0.123, pythia410m\_step0 0.225, pythia410m 0.339, char3 0.194, glove 0.274, tree 0.670.

= References
+ #link("https://www.goodfire.com/research/phylogeny-manifold")[Pearce, Simon, Byun, Balsam (Goodfire) 2025. Finding the Tree of Life in Evo 2.] <ref-goodfire_evo2>
+ #link("https://www.goodfire.com/research/neural-geometry")[Goodfire 2025-26. Neural geometry research series.] <ref-goodfire_geom>
+ #link("https://arxiv.org/abs/2405.14860")[Engels, Michaud, Liao, Gurnee, Tegmark 2024. Not All Language Model Features Are One-Dimensionally Linear.] <ref-engels>
+ #link("https://arxiv.org/abs/2505.18235")[Modell, Rubin-Delanchy, Whiteley 2025. The Origins of Representation Manifolds in Large Language Models.] <ref-modell>
+ #link("https://arxiv.org/abs/2605.05115")[Wurgaft et al. (Goodfire) 2026. Steering Along Manifolds. arXiv 2605.05115.] <ref-wurgaft>
+ #link("https://arxiv.org/abs/2406.01506")[Park, Choe, Jiang, Veitch 2024. The Geometry of Categorical and Hierarchical Concepts in Large Language Models.] <ref-park>
+ #link("https://arxiv.org/abs/2310.02207")[Gurnee, Tegmark 2023. Language Models Represent Space and Time.] <ref-gurnee>
+ #link("https://arxiv.org/abs/2605.23821")[Nava, Wyart 2026. Co-occurrence statistics and hierarchical representation geometry (arXiv 2605.23821).] <ref-nava>
+ #link("https://aclanthology.org/N19-1419/")[Hewitt, Manning 2019. A Structural Probe for Finding Syntax in Word Representations. NAACL.] <ref-hewitt>
+ #link("https://arxiv.org/abs/1705.08039")[Nickel, Kiela 2017. Poincaré Embeddings for Learning Hierarchical Representations.] <ref-nickel>
+ #link("https://arxiv.org/abs/1904.02239")[Khrulkov, Mirvakhabova, Ustinova, Oseledets, Lempitsky 2020. Hyperbolic Image Embeddings.] <ref-khrulkov>
+ #link("https://github.com/jadencutinha/phylotext-mammals")[phylotext-mammals (GitHub repository, user jadencutinha).] <ref-phylotext>
+ #link("https://tree.opentreeoflife.org/")[Open Tree of Life (synthetic tree and taxonomy; CC0).] <ref-opentree>
+ #link("https://timetree.org/")[Kumar et al. 2022. TimeTree 5. Molecular Biology and Evolution.] <ref-timetree>
+ #link("https://esapubs.org/archive/ecol/E095/178/")[Wilman et al. 2014. EltonTraits 1.0. Ecology 95:2027.] <ref-elton>
+ #link("https://huggingface.co/Qwen/Qwen3.5-2B")[Qwen team 2026. Qwen3.5-2B model card.] <ref-qwen>
+ #link("https://arxiv.org/abs/2304.01373")[Biderman et al. 2023. Pythia: A Suite for Analyzing Large Language Models.] <ref-pythia>
+ #link("https://nlp.stanford.edu/projects/glove/")[Pennington, Socher, Manning 2014. GloVe.] <ref-glove>
+ #link("https://arxiv.org/abs/2108.02497")[Lones 2021. How to avoid machine learning pitfalls.] <ref-lones>

#v(8pt)
#line(length: 100%)
*AI statement.* This codebase and paper were written by Claude Code (Anthropic), executing a specification provided by a human
(Libor Burian); human cleanup and rewriting are in progress. All numbers are generated from the repository's results files.
