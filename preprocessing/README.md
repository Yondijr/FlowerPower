### Pre-processing

Run mouth cropping script to save grayscale mouth ROIs. We assume you save cropped mouths to *`$TCN_LIPREADING_ROOT/datasets/`*. You can choose `--testset-only` to produce testing set.

```Shell
python crop_mouth_from_video.py --video-direc <LRW-DIREC> \
                                --landmark-direc <LANDMARK-DIREC> \
                                --save-direc <OUTPUT-DIREC> \
                                --convert-gray \
                                --testset-only
```
