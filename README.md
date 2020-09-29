# Install libraries
- Create coda env: `conda create env -n multisensory python=2.7`
- pip `install myrequirements.txt`
# How to generate masks on server
`./download_models.sh`

`cd src`

```
python sep_video_multiprocess.py --cam \
                    --videosegment_dir /media/Databases/preprocess_avspeech/segment \
                    --start_clip_index 6 \
                    --n_process 16
```