
from fastai.vision.all import *
import gradio as gr

learn = load_learner('export.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


gr.Interface(
    fn=predict, 
    inputs=gr.Image(), # Changed .inputs.Image to gr.Image and shape to size
    outputs=gr.Label(num_top_classes=3) # Changed .outputs.Label to gr.Label
).launch(share=True)