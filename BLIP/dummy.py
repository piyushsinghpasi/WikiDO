import torch.nn as nn
import torch


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

# k, s = 16, 16
# m = nn.Conv2d(3, 768, k, stride=s)
# m1 = nn.Conv2d(768, 768, 4, 4)
# print(get_param_num(m))

# input = torch.randn(64, 3, 256, 256)
# output = m(input)
# print(output.shape)
# x = output.flatten(2).transpose(1, 2)
# print(x.shape)
# x = x.transpose(1,2)
# print(x.shape)
# x = x.unflatten(2,(16,16))
# print(x.shape)
# x = m1(x)
# print(x.shape)

import spacy

def run_dependency_parser(sentences):
    phrases_list = []
    nlp = spacy.load("en_core_web_sm")

    for sentence in sentences:
        # Process the sentence with spaCy
        doc = nlp(sentence)

        # Extract noun chunks (phrases) from the dependency parse
        phrases = [chunk.text for chunk in doc.noun_chunks]
        phrases_list.append(phrases)

    return phrases_list

# Example usage
sentences = ["A train station in Japan with a train on the tracks and a parking lot nearby", 
             "The Carr Building, which was once the Trinity College School of Law, is a large brick building with a white doorway. The building has a symmetrical design with a row of windows on each side and a row of windows on the top floor. The building is situated on a grassy area, and there is a bench located in front of it.",
             "A cemetery with a large white building and a tall tower, which is the tomb of Mohamed Bachir El Ibrahimi.",
             "The ruins of the Kashi Vishwanath temple are shown, with a Manastambha outside the temple."]

phrases_list = run_dependency_parser(sentences)

print(phrases_list)