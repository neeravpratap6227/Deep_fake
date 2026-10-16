# Deepfake detection model

## How to run:

Import the ipynb file in Google Colab. Make sure to use a GPU runtime (T4). Run the cells in order, uploading files when required.
There are a few checkpoint cells to upload zip files after downloading some beforehand. Ignore these if you're running for the first time.

## Based on:

### [FACTOR: Detecting Deepfakes without Seeing Any](https://arxiv.org/pdf/2311.01458.pdf) (Tal Reiss et al.)

Practical recipe for deepfake fact checking and demonstrate its power in critical
attack settings: face swapping and audio-visual synthesis. Although it is training-
free, relies exclusively on off-the-shelf features, is very easy to implement, and
does not see any deepfakes, it achieves better than state-of-the-art accuracy.

Citation:
```
@article{reiss2023detecting,
  title={Detecting Deepfakes Without Seeing Any},
  author={Reiss, Tal and Cavia, Bar and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2311.01458},
  year={2023}
}
```

### [DFDC Solution](https://github.com/selimsef/dfdc_deepfake_challenge/tree/master) (Selim Seferbekov)

Based on:

- [The Deepfake Detection Challenge (DFDC) Preview Dataset](https://arxiv.org/abs/1910.08854)
- [Deep Fake Image Detection Based on Pairwise Learning](https://www.mdpi.com/2076-3417/10/1/370)
- [DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection](https://arxiv.org/abs/2001.03024)
- [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)
- [Real or Fake? Spoofing State-Of-The-Art Face Synthesis Detection Systems](https://arxiv.org/abs/1911.05351)
- [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/abs/1912.11035)
- [FakeSpotter: A Simple yet Robust Baseline for Spotting AI-Synthesized Fake Faces](https://arxiv.org/abs/1909.06122)
- [FakeLocator: Robust Localization of GAN-Based Face Manipulations via Semantic Segmentation Networks with Bells and Whistles](https://arxiv.org/abs/2001.09598)
- [Media Forensics and DeepFakes: an overview](https://arxiv.org/abs/2001.06564)
- [Face X-ray for More General Face Forgery Detection](https://arxiv.org/abs/1912.13458)
