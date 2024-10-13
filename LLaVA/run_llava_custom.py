import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import json

import ast




def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



def eval_model(model_path,image_url,query,tokenizer, model, image_processor, context_len):
    # Model
    disable_torch_init()

#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    qs = query
    print(qs)
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

#     if args.conv_mode is not None and conv_mode != args.conv_mode:
#         print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
#     else:
#         args.conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
    prompt = qs
    image = load_image(image_url)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


if __name__ == "__main__":

    data_path = '/workspace/iclr/test_set.json'
    model_path = "/workspace/pavan/multimodal/LLaVA/llava-v1.5-13b"
    df = pd.read_json(data_path)
    # df_temp = pd.read_csv(data_path)
    # df['outputs'] = df_temp['outputs']
    # df['outputs_1hop'] = df_temp['outputs_1hop']
    # df['outputs_2hop'] = df_temp['outputs_2hop']
    # df['outputs_gpt'] = df_temp['outputs_gpt']
    df.dropna()
    df = shuffle(df)
    # df = df[:10]
    lenth = len(df)
    print(df.columns)
    
    images_path = '/workspace/iclr/images'
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    outputs, outputs_1hop, outputs_2hop, outputs_gpt = [], [], [],[]
    count = 0
    results = {}
    for _, row in tqdm(df.iterrows(),total=df.shape[0]):
        count = count + 1
        image_url = images_path + "/" + row['image_id']
        kg_triples_1_hop, kg_triples_2_hop, gpt_triples = [],[],[]
        query_no_kg, query_1_hop, query_2_hop, query_gpt_triples = "","","",""
        kg_triples_1_hop = row['ques_1hop_triples']
        kg_triples_2_hop = row['ques_2hop_triples']
        gpt_triples = row['gpt_triples']

        
#         Prompt for no KG
        # query_no_kg += row['question'] + "\n"
        # for choice in row["answer choices"]:
        #     query_no_kg += choice + "\n"
        # query_no_kg += "\nGiven the image, choose one answer out of A,B,C,D. Also, provide a reason supporting your answer choice. Answer: "
        # query_no_kg+="\nThe correct answer is " + row['outputs'][0] + "\nBased on the image, give a detailed explanation for why this is the correct answer out of the given answer choices. Reason: "
        #Prompt for 1 hop triples
        query_1_hop += "Below are facts in the form of triples that might be meaningful to answer the question:\n"
        for triple in kg_triples_1_hop:
            query_1_hop += f"{triple}\n"
        query_1_hop += "Focus less on the given triples.\n\n"        
        query_1_hop += row['question'] + "\n"
        for choice in row["answer choices"]:
            query_1_hop += choice + "\n"
        query_1_hop += "\nGiven the image, choose one answer out of A,B,C,D. Also, provide a reason supporting your answer choice. Answer: "
#         query_1_hop+="\nThe correct answer is " + row['outputs_1hop'][0] + "\nBased on the image, give a detailed explanation for why this is the correct answer out of the given answer choices. Reason: "
        
#         #Prompt for 2 hop triples
        query_2_hop += "Below are facts in the form of triples that might be meaningful to answer the question:\n"
        for triple in kg_triples_2_hop:
            query_2_hop += f"{triple}\n"
        query_2_hop += "Focus less on the given triples.\n\n"        
        query_2_hop += row['question'] + "\n"
        for choice in row["answer choices"]:
            query_2_hop += choice + "\n"
        query_2_hop += "\nGiven the image, choose one answer out of A,B,C,D. Answer: "
#         query_2_hop+="\nThe correct answer is " + row['outputs_2hop'][0] + "\nBased on the image, give a detailed explanation for why this is the correct answer out of the given answer choices. Reason: "
        
        # prompt for gpt triples
        query_gpt_triples += "Below are facts in the form of triples that might be meaningful to answer the question:\n"
        for triple in gpt_triples:
            query_gpt_triples += f"{triple}\n"
        query_gpt_triples += "Focus less on the given triples.\n\n"        
        query_gpt_triples += row['question'] + "\n"
        for choice in row["answer choices"]:
            query_gpt_triples += choice + "\n"
        query_gpt_triples += "\nGiven the image, choose one answer out of A,B,C,D. Also, provide a reason supporting the choice. Answer: "
#         query_gpt_triples+="\nThe correct answer is " + row['outputs_gpt'][0] + "\nBased on the image, give a detailed explanation for why this is the correct answer out of the given answer choices. Reason: "

        
        # do inference
        # output = eval_model(model_path,image_url,query_no_kg,tokenizer, model, image_processor, context_len)
        # results[count] = {'result': row['correct answer'], 'model answer': output}
        # with open('op_no_kg.json', 'w') as f:
        #     json.dump(results, f)
        output_1hop = eval_model(model_path,image_url,query_1_hop,tokenizer, model, image_processor, context_len)
        results[count] = {'result': row['correct answer'], 'model answer': output_1hop}
        with open('op_1_hop.json', 'w') as f:
            json.dump(results, f)
        output_2hop = eval_model(model_path,image_url,query_2_hop,tokenizer, model, image_processor, context_len)
        results[count] = {'result': row['correct answer'], 'model answer': output_2hop}
        with open('op_1_2_hop.json', 'w') as f:
            json.dump(results, f)
        output_gpt = eval_model(model_path,image_url,query_gpt_triples,tokenizer, model, image_processor, context_len)
        results[count] = {'result': row['correct answer'], 'model answer': output_gpt}
        with open('op_gpt.json', 'w') as f:
            json.dump(results, f)

        # save output
        # outputs.append(output)
        outputs_1hop.append(output_1hop)
        outputs_2hop.append(output_2hop)
        outputs_gpt.append(output_gpt)
        
    # df['outputs'] = outputs
    df['outputs_1hop'] = outputs_1hop
    df['outputs_2hop'] = outputs_2hop
    df['outputs_gpt'] = outputs_gpt
    df.to_csv("llava_new_all.csv")
    

    
        
        


        
        
    