import gradio as gr
import random
import time
import json
import time
import urllib.request
import sys
sys.path.append("/")
import json
from typing import List, Optional, Tuple, Union
from config import *
from load_model import get_model


request_rewriter, searcher, passage_extractor, answer_generator, fact_checker = get_model()
def predict(input, history=[], history_rewrite_input=[], history_url=[]):
    #STEP 1 Query Rewriting
    # revised_request = request_rewriter.request_rewrite(history_rewrite_input, input)
    revised_request = input
    history_rewrite_input.append(revised_request) #record all the revised request
    print("revised_request:", revised_request)

    #STEP 2 Doc Retrieval
    raw_reference_list = searcher.search(revised_request)
    print("raw_reference_list:", raw_reference_list)

    #collect the retrieved urls
    urls = ""
    for raw_reference in raw_reference_list :
        if "url" in raw_reference:
            urls = urls + "- {}: {}\n".format(raw_reference['title'], raw_reference['url']) 
        else:
            urls = urls + "- {} \n".format(raw_reference['title']) 
    #recorde all historical retrieved urls
    history_url.append(urls) 
    
    #STEP 3 Passage Extractor
    reference = ""
    for raw_reference in raw_reference_list:
        reference = reference + passage_extractor.extract(raw_reference, revised_request, if_extract) + "\n"
    #truncate the references
    reference = reference[:cutoff] 

    print("reference:", reference)

    #STEP 4 Answer Generation
    output = answer_generator.answer_generate(reference, revised_request)
    
    #STEP 5 Fact Verification
    if not (fact_checker.fact_check(reference, output)) :
        output = "I cannot answer this request."

    history.append((input, reference, output))
    print((output + urls))
    return (output + "\n参考信息：\n" + urls)



with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center"> Chat with Knowledges  </h1>""")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_message = predict(message)
        chat_history.append((message, bot_message))
        time.sleep(1)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share = True)
# demo.launch()

