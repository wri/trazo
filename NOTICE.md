# Attribution

Trazo is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).
You may share and adapt this work, including commercially, provided you give
appropriate credit, link to the license, and indicate whether changes were made.

## How to attribute

> Trazo (2026), World Resources Institute and the Kerner Lab at Arizona State
> University. Licensed under CC BY 4.0. https://github.com/wri/trazo

BibTeX:

```bibtex
@misc{trazo2026,
  title        = {Trazo: creating training data and models for field boundary detection},
  author       = {{World Resources Institute} and {Kerner Lab, Arizona State University}},
  year         = {2026},
  howpublished = {\url{https://github.com/wri/trazo}},
  note         = {Licensed under CC BY 4.0}
}
```

## Third-party components

- Builds on [Fields of the World (FTW)](https://github.com/fieldsoftheworld) and
  the `ftw-tools` package.
- The LoRA implementation in `src/trazo/pt3_finetune/lora.py` is adapted from
  [Microsoft LoRA](https://github.com/microsoft/LoRA), MIT licensed. The original
  copyright notice is retained in that file.
- SOS/EOS season rasters under `seasontifs/` originate from
  [ucg-uv/research_products](https://github.com/ucg-uv/research_products); see
  `seasontifs/readme.txt`.

## Funding

Funded by a Walmart Foundation grant with support from Land and Carbon Lab,
Taylor Geospatial Engine, and WRI's Data Lab.
