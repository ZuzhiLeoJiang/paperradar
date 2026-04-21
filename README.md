# paperradar
Customize your personal academic literature radar. paperradar pulls recent papers from the journals you select, filters by keywords, ranks them by semantic relevance to seed papers you define, and delivers the results as a clean HTML digest.

## Credits
Originally created as `lit_feed` by [Yujie York Zhang](https://github.com/zyj1729). This fork is maintained by Zuzhi Leo Jiang with customizations for single-cell multi-omics and perturbation biology research.


## Installation

```bash
git clone https://github.com/ZuzhiLeoJiang/paperradar.git
pip install feedparser requests sentence-transformers torch

cd paperradar
python paperradar.py
```

## Journals default

**High-impact broad**  
_Nature_ · _Cell_ · _Science_ · _eLife_

**Developmental biology**  
_Development_ · _Developmental Cell_ · _Cell Stem Cell_ · _Genes & Development_

**Physiology & gut motility**  
_Journal of Physiology_ · _American Journal of Physiology – GI_

**Vascular & lymphatic**  
_Circulation Research_

**Preprints**  
_bioRxiv Developmental Biology_ · _bioRxiv Cell Biology_ · _arXiv q-bio_

## Customization

You can customize the journals to include in **Feeds** section, keywords to include or exclude in **Keywords filters** section, seed paper of relevance in **Canonical papers** section.
