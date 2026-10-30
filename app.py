"""
Titanic Survival Predictor (Gradio demo).

Run:  python app.py
Fill in the passenger details and press "Predict".
If the saved model is missing, it is trained automatically the first time.
"""
import os
import json
import joblib
import pandas as pd
import gradio as gr

import train_model

if not os.path.exists(train_model.MODEL_PATH):
    print("No saved model found. Training now (about a minute)...")
    train_model.train(verbose=False)

model = joblib.load(train_model.MODEL_PATH)

try:
    with open(train_model.METRICS_PATH) as f:
        _m = json.load(f)
    MODEL_NOTE = (f"{_m['final_model']} model, {_m['accuracy']:.0%} accurate on unseen test passengers. "
                  "It can be wrong, so treat the result as an estimate.")
except Exception:
    MODEL_NOTE = "This model can be wrong, so treat the result as an estimate."


# ---------------------------------------------------------------- result cards
CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#header { text-align: center; padding: 18px 0 6px 0; }
#header h1 { font-size: 2.1rem; margin: 0; }
#header p  { opacity: .75; margin: 6px 0 0 0; font-size: 1.02rem; }
.result-card { border-radius: 16px; padding: 28px 24px; text-align: center;
               border: 2px solid var(--border-color-primary); min-height: 300px;
               display: flex; flex-direction: column; justify-content: center; }
.result-card .verdict { font-size: 2.2rem; font-weight: 800; margin: 6px 0 4px 0; }
.result-card .sub { opacity: .8; font-size: 1.05rem; margin-bottom: 18px; }
.result-card .bar { height: 16px; border-radius: 999px; background: rgba(127,127,127,.25);
                    overflow: hidden; margin: 6px auto 8px auto; max-width: 420px; }
.result-card .bar > div { height: 100%; border-radius: 999px; }
.result-card .legend { display: flex; justify-content: space-between; max-width: 420px;
                       margin: 0 auto; font-size: .9rem; opacity: .85; }
.result-card .note { margin-top: 22px; font-size: .85rem; opacity: .65; }
.result-card.ok  { border-color: #16a34a; background: rgba(22,163,74,.10); }
.result-card.bad { border-color: #dc2626; background: rgba(220,38,38,.10); }
.result-card.err { border-color: #d97706; background: rgba(217,119,6,.12); }
.result-card .ok-text  { color: #16a34a; }
.result-card .bad-text { color: #dc2626; }
.result-card .err-text { color: #d97706; }
.section-title { font-weight: 700; font-size: 1.02rem; margin: 2px 0 6px 0; }
#footer { text-align: center; opacity: .6; font-size: .85rem; padding: 10px 0; }
"""

PLACEHOLDER = """
<div class="result-card">
  <div style="font-size:3rem">🚢</div>
  <div class="verdict">Ready when you are</div>
  <div class="sub">Fill in the passenger details on the left<br>and press <b>Predict</b>.</div>
  <div class="note">""" + MODEL_NOTE + """</div>
</div>
"""


def _card(verdict_html, sub, p, css_class, text_class, color):
    pct = round(p * 100)
    return f"""
<div class="result-card {css_class}">
  <div style="font-size:3rem">{'✅' if css_class == 'ok' else '❌'}</div>
  <div class="verdict {text_class}">{verdict_html}</div>
  <div class="sub">{sub}</div>
  <div class="bar"><div style="width:{pct}%; background:{color}"></div></div>
  <div class="legend"><span>Chance of survival: <b>{pct}%</b></span><span>Not surviving: <b>{100 - pct}%</b></span></div>
  <div class="note">{MODEL_NOTE}</div>
</div>
"""


def _error_card(msg):
    return f"""
<div class="result-card err">
  <div style="font-size:3rem">⚠️</div>
  <div class="verdict err-text">Something went wrong</div>
  <div class="sub">{msg}<br>Please check that every field is filled in.</div>
</div>
"""


def predict(pclass, sex, age, fare, sibsp, parch, embarked, has_cabin):
    try:
        if age is None or fare is None:
            raise ValueError("Age and fare must be filled in.")
        sibsp, parch = int(sibsp or 0), int(parch or 0)
        family_size = sibsp + parch + 1
        row = pd.DataFrame([{
            "pclass": int(pclass),
            "sex": sex,
            "age": float(age),
            "sibsp": sibsp,
            "parch": parch,
            "fare": float(fare),
            "embarked": embarked,
            "family_size": family_size,
            "is_alone": int(family_size == 1),
            "has_cabin": int(bool(has_cabin)),
        }])
        p = float(model.predict_proba(row)[0][1])
        if p >= 0.5:
            return _card("SURVIVED", "The model predicts this passenger survived.",
                         p, "ok", "ok-text", "#16a34a")
        return _card("DID NOT SURVIVE", "The model predicts this passenger did not survive.",
                     p, "bad", "bad-text", "#dc2626")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _error_card(str(e))


# --------------------------------------------------------------------- the UI
EXAMPLES = {
    "👩 First-class woman":  [1, "female", 28, 80, 0, 0, "Southampton", True],
    "👨 Third-class man":    [3, "male", 22, 7, 0, 0, "Southampton", False],
    "👩‍👧 Mother with family": [2, "female", 30, 21, 1, 1, "Southampton", False],
    "👦 Boy, large family":  [3, "male", 8, 21, 3, 1, "Southampton", False],
}
PORTS = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}


def run(pclass, sex, age, fare, sibsp, parch, port, has_cabin):
    return predict(pclass, sex, age, fare, sibsp, parch, PORTS[port], has_cabin)


def build_app():
    with gr.Blocks(title="Titanic Survival Predictor") as demo:
        gr.HTML("""
        <div id="header">
          <h1>🚢 Titanic Survival Predictor</h1>
          <p>Enter a passenger's details to see whether the model predicts they would have survived.</p>
        </div>""")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("<div class='section-title'>👤 Passenger</div>")
                with gr.Group():
                    sex = gr.Radio(["male", "female"], value="male", label="Sex")
                    age = gr.Slider(0, 80, value=30, step=1, label="Age (years)")
                gr.Markdown("<div class='section-title'>🎫 Ticket</div>")
                with gr.Group():
                    pclass = gr.Radio([("1st class", 1), ("2nd class", 2), ("3rd class", 3)],
                                      value=3, label="Ticket class")
                    fare = gr.Number(value=32, minimum=0, label="Fare paid (£, typically 7 to 512)")
                    port = gr.Dropdown(list(PORTS.keys()), value="Southampton", label="Port of embarkation")
                    has_cabin = gr.Checkbox(value=False, label="Had a cabin")
                gr.Markdown("<div class='section-title'>👪 Family on board</div>")
                with gr.Group():
                    with gr.Row():
                        sibsp = gr.Number(value=0, precision=0, minimum=0, label="Siblings / spouse")
                        parch = gr.Number(value=0, precision=0, minimum=0, label="Parents / children")
                with gr.Row():
                    clear_btn = gr.Button("Reset", variant="secondary")
                    go_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("<div class='section-title'>📊 Prediction</div>")
                result = gr.HTML(PLACEHOLDER)
                gr.Markdown("<div class='section-title'>⚡ Try an example</div>")
                with gr.Row():
                    ex_buttons = [gr.Button(label, size="sm") for label in EXAMPLES]

        gr.HTML("<div id='footer'>Learning project built with scikit-learn and Gradio. "
                "Trained on 891 Titanic passengers. Not for real decisions.</div>")

        inputs = [pclass, sex, age, fare, sibsp, parch, port, has_cabin]
        go_btn.click(run, inputs, result, api_name="predict")

        for btn, values in zip(ex_buttons, EXAMPLES.values()):
            btn.click(lambda v=values: v, None, inputs).then(run, inputs, result)

        clear_btn.click(lambda: [3, "male", 30, 32, 0, 0, "Southampton", False, PLACEHOLDER],
                        None, inputs + [result])
    return demo


if __name__ == "__main__":
    build_app().launch(inbrowser=True, css=CSS, theme=gr.themes.Soft())
