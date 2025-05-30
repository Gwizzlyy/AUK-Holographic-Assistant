
# AUK AI Holographic Assistant
[![MIT License](https://img.shields.io/badge/License-Apache2.0-red.svg)](https://choosealicense.com/licenses/apache-2.0/)

An innovative Holographic Assistant delivering intuitive interactions through NLP and immersive 3D visuals. Enhances Human-Computer interaction (HCI) via intuitive voice-based interactions. Provides real-time, intelligent access to academic and administrive services at AUK.




## Authors

- [Homer Alnowaif](https://www.github.com/Gwizzlyy)
- [Maryam AlMulla](https://www.github.com/maryamalmullax)
- [Shahd Abdouh](https://www.github.com/Shahd-R)
- [Jasem AlSanea](https://www.github.com/jals27)
- Under Supervision of [Dr. Marwa Sharawi](https://www.researchgate.net/profile/Marwa-Sharawi-2)


## Features

- Local fine-tuned and trained LLaMA model on AUK data.
- Voice Activation using OpenWakeWord & RealtimeSTT.
- Speech-to-text and text-to-speech.
- Fallback to web and OpenAI GPT-4 for fact-checks.
- MPV-controlled avatar with idle/talking animations.


## Installation

Clone the Repository

```bash
  git clone https://github.com/Gwizzlyy/AUK-Holographic-Assistant.git
```
Create a Python Environment

```bash
    python -m venv env
    source env/bin/activate # Windows: env\Scripts\activate
```
## Deployment

To deploy this project, first download the [AI Model](https://huggingface.co/Gwizzly/model500).

Create a Python 3.12 Environment and run
```bash
  pip install -r requirements.txt
```
Download [mpv](https://mpv.io/) and add it to PATH (User PATH Variables) and setup the pipe socket .

Launch ```python main.py``` after secrets are configured.

Change ```gpt_recorder()``` to other modes like Voice Activation.
```python
if __name__ == "__main__":
    mpv_process = start_mpv_player()
    try:
        gpt_recorder() # Call the modes here like manual_recorder()
    except KeyboardInterrupt:
        print("\nTerminated.")
        if mpv_process:
            mpv_process.terminate()
```

Manual Mode:
`m` for typing
`[ENTER]` for push-to-talk

Voice Activation Mode: say "Hello Wolfy!" and ask your question.



## Environment Variables / Secrets

To run this project, you will need to add the following environment variables.

`ELEVENLABS_KEY`

`OPENAI_KEY`

`onnx_path`

`model_path`


## Example Questions
- "How many credits do I need to graduate from AUK?"
- "How can I pay my tuiton?"
- "What documents do I need to apply?"
## Project Structure

```
.
├── main.py                 # Main script
├── model.gguf              # LLaMA model file
├── Hellow_wolfy!.onnx      # Wakeword model
├── p_idle.mp4              # Idle animation video
├── p_talking.mp4           # Talking animation video
├── requirements.txt        # Includes all the dependencies
└── README.md
```
## Acknowledgements

 - [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
 - [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit)



## License

[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

