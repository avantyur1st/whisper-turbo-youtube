# **YouTube Video Transcriptions with OpenAI Whisper Featuring the Turbo Model**

[![blog post shield](https://img.shields.io/static/v1?label=&message=Blog%20post&color=blue&style=for-the-badge&logo=openai&link=https://openai.com/blog/whisper)](https://openai.com/blog/whisper)
[![notebook shield](https://img.shields.io/static/v1?label=&message=Notebook&color=blue&style=for-the-badge&logo=googlecolab&link=https://colab.research.google.com/github/avantyur1st/whisper-turbo-youtube/blob/main/whisper_turbo_youtube.ipynb)](https://colab.research.google.com/github/avantyur1st/whisper-turbo-youtube/blob/main/whisper_turbo_youtube.ipynb)
[![repository shield](https://img.shields.io/static/v1?label=&message=Repository&color=blue&style=for-the-badge&logo=github&link=https://github.com/openai/whisper)](https://github.com/openai/whisper)
[![paper shield](https://img.shields.io/static/v1?label=&message=Paper&color=blue&style=for-the-badge&link=https://cdn.openai.com/papers/whisper.pdf)](https://cdn.openai.com/papers/whisper.pdf)
[![model card shield](https://img.shields.io/static/v1?label=&message=Model%20card&color=blue&style=for-the-badge&link=https://github.com/openai/whisper/blob/main/model-card.md)](https://github.com/openai/whisper/blob/main/model-card.md)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

This notebook will guide you through the transcription of a Youtube video using Whisper. You'll be able to explore most inference parameters or use the Notebook as-is to store the transcript and the audio of the video in your Google Drive.


# **Check GPU type** 🕵️

The type of GPU you get assigned in your Colab session defined the speed at which the video will be transcribed.
The higher the number of floating point operations per second (FLOPS), the faster the transcription.
But even the least powerful GPU available in Colab is able to run any Whisper model.
Make sure you've selected `GPU` as hardware accelerator for the Notebook (Runtime &rarr; Change runtime type &rarr; Hardware accelerator).

|  GPU   |  GPU RAM   | FP32 teraFLOPS |     Availability   |
|:------:|:----------:|:--------------:|:------------------:|
|  T4    |    16 GB   |       8.1      |   Free / Colab Pro |
|  L4    |    24 GB   |      30.3      |     Colab Pro      |
| A100   | 40 / 80 GB |      19.5      |  Colab Pro / Pro+  |

---
**Factory reset your Notebook's runtime if you want to get assigned a new GPU.**


```
GPU 0: Tesla T4 (UUID: GPU-0079ab15-5b72-04e1-d928-1f3bf288393d)
Mon Dec  2 11:17:06 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
| N/A   40C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```


# **Install libraries** 🏗️
This cell will take a little while to download several libraries, including Whisper.

---

# **Optional:** Save images in Google Drive 💾
Enter a Google Drive path and run this cell if you want to store the results inside Google Drive.

---

```drive_path = "Colab Notebooks/Whisper Youtube"```

---
**Run this cell again if you change your Google Drive path.**



# **Model selection** 🧠

There are 6 pre-trained options to play with:

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

The discussion about which model is better to choose:
https://github.com/openai/whisper/discussions/2363
---

```Model = 'turbo'```

---
**Run this cell again if you change the model.**

```
    100%|█████████████████████████████████████| 1.51G/1.51G [00:21<00:00, 75.2MiB/s]
```

**turbo model is selected.**



# **Video selection** 📺

Enter the URL of the Youtube video you want to transcribe, whether you want to save the audio file in your Google Drive, and run the cell.

---

```URL = "https://youtu.be/dQw4w9WgXcQ"```

```store_audio = True```

---
**Run this cell again if you change the video.**

# **Run the model** 🚀

Run this cell to execute the transcription of the video. This can take a while and is very much based on the length of the video and the number of parameters of the model selected above.

---

```Language = "English"```

```Output_type = '.vtt'```

---

```
    [00:00.000 --> 00:22.000]  We're no strangers to love.
    [00:22.000 --> 00:27.000]  You know the rules, and so do I.
    [00:27.000 --> 00:31.000]  Our full commitments while I'm thinking of.
    [00:31.000 --> 00:35.000]  You wouldn't get this from any other guy.
    [00:35.000 --> 00:40.000]  I just wanna tell you how I'm feeling.
    [00:40.000 --> 00:43.000]  Gotta make you understand.
    [00:43.000 --> 00:45.000]  Never gonna give you up.
    [00:45.000 --> 00:47.000]  Never gonna let you down.
    [00:47.000 --> 00:51.000]  Never gonna run around and desert you.
    [00:51.000 --> 00:53.000]  Never gonna make you cry.
    [00:53.000 --> 00:55.000]  Never gonna say goodbye.
    [00:55.000 --> 01:00.000]  Never gonna tell a lie and hurt you.
    [01:00.000 --> 01:04.000]  We've known each other for so long.
    [01:04.000 --> 01:09.000]  Your heart's been aching, but you're too shy to say it.
    [01:09.000 --> 01:13.000]  Inside we both know what's been going on.
    [01:13.000 --> 01:17.000]  We know the game and we're gonna play it.
    [01:17.000 --> 01:22.000]  And if you ask me how I'm feeling.
    [01:22.000 --> 01:25.000]  Don't tell me you're too blind to see.
    [01:25.000 --> 01:27.000]  Never gonna give you up.
    [01:27.000 --> 01:29.000]  Never gonna let you down.
    [01:29.000 --> 01:33.000]  Never gonna run around and desert you.
    [01:33.000 --> 01:35.000]  Never gonna make you cry.
    [01:35.000 --> 01:38.000]  Never gonna say goodbye.
    [01:38.000 --> 01:41.000]  Never gonna tell a lie and hurt you.
    [01:41.000 --> 01:43.000]  Never gonna give you up.
    [01:43.000 --> 01:46.000]  Never gonna let you down.
    [01:46.000 --> 01:50.000]  Never gonna run around and desert you.
    [01:50.000 --> 01:59.000]  Never gonna make you cry, never gonna say goodbye, never gonna tell a lie and hurt you
    [01:59.000 --> 02:07.000]  Give you love, give you love
    [02:07.000 --> 02:16.000]  Never gonna give, never gonna give, give you love
    [02:16.000 --> 02:25.000]  We've known each other for so long, your heart's been aching but you're too shy to say it
    [02:25.000 --> 02:33.000]  Inside we both know what's been going on, we know the game and we're gonna play it
    [02:33.000 --> 02:41.000]  I just wanna tell you how I'm feeling, gotta make you understand
    [02:41.000 --> 02:49.000]  Never gonna give you up, never gonna let you down, never gonna run around and desert you
    [02:49.000 --> 02:57.000]  Never gonna make you cry, never gonna say goodbye, never gonna tell a lie and hurt you
    [02:57.000 --> 03:06.000]  Never gonna give you up, never gonna let you down, never gonna run around and desert you
    [03:06.000 --> 03:14.500]  Never gonna make you cry, never gonna say goodbye, never gonna tell a lie, and hurt you.
    [03:14.500 --> 03:23.000]  Never gonna give you up, never gonna let you down, never gonna run around and desert you.
    [03:23.000 --> 03:27.500]  We're gonna make you cry, we're gonna say goodbye,
    [03:27.500 --> 03:53.400]  we're gonna say goodbye.
```

**Transcript file created: /content/drive/My Drive/Colab Notebooks/Whisper Youtube/dQw4w9WgXcQ.vtt**

## Acknowledgments

* The original code can be found [here](https://github.com/ArthurFDLR/whisper-youtube). Thanks to the authors for their work!
