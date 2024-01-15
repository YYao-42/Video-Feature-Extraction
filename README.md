Code for video processing in the paper "Identifying Temporal Correlations Between Natural Single-shot Videos and EEG Signals" [1].

- To extract object-based optical flow (of a video): `python maskof.py --input path/to/video --output path/to/output` 
- To extract object-based temporal contrast (of videos in a folder): `python tempcontrast.py --input path/to/video/folder --output path/to/output --mask`
- To generate videos for experiment from a list of video clips: `python video_merge.py --input path/to/video/folder --output path/to/output --fps 30`

Please download the folder necessary for using Mask-RCNN from [here](https://drive.google.com/file/d/1cshnv6gNhMh-nT-qpUNQC978IFQQ3E23/view?usp=sharing) and put it in the same directory as the code.

**Reference:**

[1] Yao, Y., Stebner, A., Tuytelaars, T., Geirnaert, S., & Bertrand, A. (2023). Identifying Temporal Correlations Between Natural Single-shot Videos and EEG Signals. bioRxiv. doi:10.1101/2023.09.19.558394