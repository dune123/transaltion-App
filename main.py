from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import streamlit as st

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


article_en=st.text_input("Enter the text you want to translate : ")

model_inputs=tokenizer(article_en,return_tensors="pt")

#translator from english to hindi
generated_tokens=model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
)
translator=tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)

st.write("translated hindi text : ")
st.write(translator)