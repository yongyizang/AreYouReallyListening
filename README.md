# Are you really listening? Boosting Perceptual Awareness in Music-QA Benchmarks

## Getting Started
If you are only using this repository for evaluation, you don't need to install anything. Just download the `RUL-MuchoMusic.json` file, and reference the python scripts in `ExampleAudioEval` folder; we provide example scripts to run evaluation on both Qwen2-Audio and Qwen-Audio. We recommend using the `leave-one-out` evaluation method; additionally, we recommend reporting evaluation results on random gaussian noise alongside with the original audio.

If you are looking to use this repository for filtering some other dataset based on the proposed Perceptual Index (PI), see the minimal example provided in `ExampleFilteringBasedOnPI` folder. We provide the `distractors.json` as the list of distractors generated using Deepseek-V3, and you could modify `filter_based_on_PI.py` to use any vLLM-supported model for filtering. By default, it uses Qwen2.5-7B.

## Citation
If you use this code, the dataset, or our proposed method, please cite our paper:

```
@article{rulmuchomusic,
  title={Are you really listening? boosting perceptual awareness in music-qa benchmarks},
  author={Zang, Yongyi and O'Brien, Sean and Berg-Kirkpatrick, Taylor and McAuley, Julian and Novack, Zachary},
  journal={International Society of Music Information Retrieval (ISMIR) 2025},
  year={2025}
}
```