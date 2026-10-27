# GENEA 2023 derived A3 gesture assets

Source: [GENEA Challenge 2023 dataset, Zenodo record 8199133](https://zenodo.org/records/8199133), based on Talking With Hands 16.2M. Licensed under **CC BY-NC 4.0**; see the original `LICENSE.txt` and `SOURCE_README.txt` in this directory. Noncommercial use only under that license. This notice applies to the derived catalog, stylized trajectories and their visualizations, not as a replacement for other repository components' licenses.

Attribution requested by the data creators:

- Taras Kucherenko, Rajmund Nagy, Youngwoo Yoon, Jieyeon Woo, Teodor Nikolov, Mihail Tsakov, and Gustav Eje Henter. 2023. *The GENEA Challenge 2023: A large-scale evaluation of gesture generation models in monadic and dyadic settings*. ACM ICMI 2023.
- Gilwoo Lee, Zhiwei Deng, Shugao Ma, Takaaki Shiratori, Siddhartha S. Srinivasa, and Yaser Sheikh. 2019. *Talking With Hands 16.2M: A large-scale dataset of synchronized body-finger motion and audio for conversational motion analysis and synthesis*. IEEE/CVF ICCV 2019, 763–772.

Changes: deterministic selection of 24 training main-agent recordings; TWH skeleton mapped to A3 upper-body directions; 30 Hz joint trajectories; per-joint limit projection, smoothing, opening and tempo/amplitude repair with source-fidelity gates; automatic activity cuts; quality filtering and rule-based semantic proposals; generated style variants. No source human semantic labels are claimed. `admission.json`, `atom_admission.json`, catalog metadata and `results/execution_20261027/genea_subset_manifest.json` provide source IDs, timestamps, transformations and integrity checks. Source audio/transcriptions remain local and unmodified. Demo speech is newly synthesized Chinese speech, not an original GENEA recording.
