from flask import Flask , redirect , request , render_template , jsonify , json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def getChatResponse(input:str,step = 0) -> str:
     # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(str(input)+ tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('chat.html')



@app.route('/get', methods = ['GET','POST'])
def chat():
    data = {}
    print(request)
    if request.method == 'POST':
        print('request is POST')
        chat = request.json
        data['bot'] = getChatResponse(input = chat)
        return jsonify(data)
    else:
        print('it is not a POST')

if __name__ == '__main__':
    app.run(debug=True)
