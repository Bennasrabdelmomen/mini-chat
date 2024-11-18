from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import wikipediaapi
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

# Initialize models
nlp = spacy.load("en_core_web_sm")
wiki_wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': 'chatbot/1.0 (contact@example.com)'})
summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
conversation_history = []

"""

General Description: 
user inserts query, passes through the extract_entities function, if there are entites that could be searched on wikipedia in the user's query
the BART model responds to the user and stores the conversation inside conversation_memory, if there are no entities found in the query, the DialoGPT model
responds to the conversation and stores the utterance inside the conversation_memory

"""

app = FastAPI()


class UserInput(BaseModel):
    message: str


def extract_entities(question):
    doc = nlp(question)
    entities = [ent.text for ent in doc.ents if
                ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]]
    return entities


def search_wikipedia_page(entity):
    page = wiki_wiki.page(entity)
    if page.exists():
        return page, page.fullurl
    return None, None


def summarize_page(page, max_length=100):
    text = page.text[:1000]
    inputs = summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0,
                                            num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def generate_conversational_response(user_input):
    entities = extract_entities(user_input)
    context_summary = ""

    if entities:
        for entity in entities:
            if entity not in [e.split(":")[0] for e in conversation_history]:
                page, url = search_wikipedia_page(entity)
                if page:
                    summary = summarize_page(page, max_length=100)
                    context_summary += summary

        if context_summary:
            response = f"Here's some information about {entities[0]}:\n{context_summary}"
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")
            return response

    input_ids = chat_tokenizer.encode(user_input + chat_tokenizer.eos_token, return_tensors="pt", max_length=1024,
                                      truncation=True)
    chat_history_ids = chat_model.generate(input_ids, max_new_tokens=50, min_length=20, temperature=0.7,
                                           attention_mask=torch.ones(input_ids.shape, dtype=torch.long),
                                           pad_token_id=chat_tokenizer.eos_token_id)
    response = chat_tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Assistant: {response}")
    return response


@app.post("/chat")
async def chat_with_assistant(user_input: UserInput):
    response = generate_conversational_response(user_input.message)
    return {"response": response}


@app.get("/conversation_history")
async def get_conversation_history():
    return {"history": conversation_history}

@app.get("/Delete_conversation_history")
async def reset():
    conversation_history=[]
    return conversation_history


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
