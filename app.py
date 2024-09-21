import gradio as gr
import numpy as np

from transformers import pipeline

pipeline_en = pipeline(task="text-to-speech", model="facebook/mms-tts-eng")
pipeline_pt = pipeline(task="text-to-speech", model="facebook/mms-tts-por")

def tts(lang):
  pipeline = pipeline_en if lang == "en" else pipeline_pt
  def tts_lang(txt):
    res = pipeline(txt)
    audio = (res['audio'].reshape(-1) * 2 ** 15).astype(np.int16)
    return res['sampling_rate'], audio
  return tts_lang

with gr.Blocks() as demo:
  gr.Interface(
    tts("en"),
    inputs=gr.Textbox(
      lines=1,
      value="one two three four",
    ),
    outputs="audio",
    allow_flagging="never",
  )

  gr.Interface(
    tts("pt"),
    inputs=gr.Textbox(
      lines=1,
      value="um dois tres quatro",
    ),
    outputs="audio",
    allow_flagging="never",
  )


if __name__ == "__main__":
   demo.launch()
