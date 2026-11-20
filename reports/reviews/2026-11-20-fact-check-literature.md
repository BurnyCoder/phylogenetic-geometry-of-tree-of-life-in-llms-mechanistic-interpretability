# Fact-check of the literature reviews (2026-11-20)

**Scope.** An independent pass checked every citation and every number in [`../literature-review.md`](../literature-review.md) and [`../state-of-the-art-review.md`](../state-of-the-art-review.md) against its primary source:
- arXiv abs pages, resolved DOIs, ACL PDFs, GitHub, Hugging Face, PyPI;
- figure images, for numbers that appear only in figures (the Goodfire Evo 2 post).

**Overall result.** Every DOI, arXiv id, ACL id and GitHub link resolves to the stated work. No fabricated sources were found. About 15 claims needed correction, mostly framing or attribution. All of them have been fixed in the reviews.

## Corrections applied

| Location | Original claim | Correction |
|---|---|---|
| LR §1.1 | Goodfire Spearman −0.80 is for *angular distance* | It is for **cosine similarity** vs phylogenetic distance, which is why the sign is negative |
| LR §1.1 | "~70% of variance captured" summarizes the 0.98/0.89 distance correlations | The ~70% is the embedding variance that the 10-d subspace **reconstructs** (29% / 37% unexplained). Train–test distance r = 0.92 added |
| LR §1.1 | k-mer XGBoost result is a confound "the authors flag" | Goodfire frames it as species encoded as genomic "styles"; reading it as a confound is ours. PC0 vs TTT/AAT/GTG r = 0.74/0.70/−0.69 added |
| LR §2.1 | Neural-geometry index "has no methods" | The index has short summaries of six posts |
| LR §2.1 | Manifold steering reads "the answer position" | It reads the **last-token position** |
| LR §2.1 | "2.8× closer to the behavior manifold" | A 2.8× average reduction of a cumulative Bhattacharyya-energy *proxy* for distance (p < 0.001) |
| LR §2.1 | SAE paper title | "Do Sparse Autoencoders Capture Concept Manifolds?" |
| LR §2.2 | Engels title in parentheses | "Not All Language Model Features Are One-Dimensionally Linear" |
| LR §2.7 | Kissane et al. SAE transfer | Tested on Mistral-7B and Qwen1.5-0.5B. Transfer failed for Gemma v1 2B. Applying it to Qwen3.5 is an extrapolation |
| LR §2.8 | Chen et al.: better "at low dimension" | The qualifier is not supported and was removed |
| LR §3 | Nickel & Kiela embed "the WordNet mammal closure" | They embed the full noun hierarchy; mammals appear only in a 2-D visualization |
| LR §3 | Hyperbolicity Atlas: "only one random-init control" | There is also a prompt-length-quartile control |
| LR §4 | arXiv 2510.02830: "class-to-family", abstract only | Kingdom-to-family; 94.9% / 91.8% / 98.0% (full text read) |
| SOTA §1 | Template averaging credited to Gurnee & Tegmark | They test sensitivity to prepended prompts and do not average |
| SOTA §1 | Mean-centering is "standard practice" (uncited) | Cited: Ethayarajh 2019; Mu & Viswanath 2018; Timkey & van Schijndel 2021 |
| SOTA §2 | TimeTree no-redistribution cited to the paper | The rule is in the timetree.org footer |
| SOTA §4 | infini-gram Pile index "pairs with Pythia" | Same corpus, but counts are in Llama-2 tokens |
| SOTA §6 | sm_120 needs CUDA ≥ 12.8, cited to the uv guide | Now cited to the PyTorch 2.7 release notes, plus a local `get_arch_list()` check |

## Still unverifiable
- The local ete4 build failure (observed on this machine; no external source).
- The claim that Goodfire released no paper or code (a negative cannot be proven).
