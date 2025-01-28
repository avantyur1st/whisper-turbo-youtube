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
|  T4    |    16 GB   |       8.1      |         Free       |
| P100   |    16 GB   |      10.6      |      Colab Pro     |
| V100   |    16 GB   |      15.7      |  Colab Pro (Rare)  |

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

```
    Collecting git+https://github.com/openai/whisper.git
      Cloning https://github.com/openai/whisper.git to /tmp/pip-req-build-h5rg7qi_
      Running command git clone --filter=blob:none --quiet https://github.com/openai/whisper.git /tmp/pip-req-build-h5rg7qi_
      Resolved https://github.com/openai/whisper.git to commit 90db0de1896c23cbfaf0c58bc2d30665f709f170
      Installing build dependencies ... done
      Getting requirements to build wheel ... done
      Preparing metadata (pyproject.toml) ... done
    Requirement already satisfied: numba in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20240930) (0.60.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20240930) (1.26.4)
    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20240930) (2.5.1+cu121)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20240930) (4.66.6)
    Requirement already satisfied: more-itertools in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20240930) (10.5.0)
    Requirement already satisfied: tiktoken in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20240930) (0.8.0)
    Requirement already satisfied: triton>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20240930) (3.1.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from triton>=2.0.0->openai-whisper==20240930) (3.16.1)
    Requirement already satisfied: llvmlite<0.44,>=0.43.0dev0 in /usr/local/lib/python3.10/dist-packages (from numba->openai-whisper==20240930) (0.43.0)
    Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken->openai-whisper==20240930) (2024.9.11)
    Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken->openai-whisper==20240930) (2.32.3)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20240930) (4.12.2)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20240930) (3.4.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20240930) (3.1.4)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20240930) (2024.10.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20240930) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch->openai-whisper==20240930) (1.3.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken->openai-whisper==20240930) (2024.8.30)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->openai-whisper==20240930) (3.0.2)
    Requirement already satisfied: yt-dlp in /usr/local/lib/python3.10/dist-packages (2024.11.18)
    Using device: cuda:0
```

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
