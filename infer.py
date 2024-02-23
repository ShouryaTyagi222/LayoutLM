import torch
from PIL import Image
from transformers import AutoModelForQuestionAnswering
from transformers import LayoutLMv2Processor
import argparse


q_map={
    'q1':"Which organization issued this given circular?",
    'q2': "What is the Address of the Issuing Authority of the given Circular?",
    'q3': "What is the Serial No./ID of the Given Circular?",
    'q4': "What is the Date of Issuance of the Circular?",
    'q5': "What is the Subject of the given Circular?",
    'q6': "Who has this circular been addressed to?",
    'q7': "To Whom has the circular been forwarded to?",
    'q8': "Who Has Forwarded This Circular?",
    'q9': "What is the Designation of the Person who Forwarded this Circular?",
    'q10': "Who has signed the Given Circular?",
    'q11': "What is the Designation of the Person who Signed this Circular?"
}

def layoutlm(model, processor, question, image):
    encoding = processor(image, question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    for k,v in encoding.items():
        encoding[k] = v.to(model.device)

    # Perform inference
    outputs = model(**encoding)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get the answer
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()
    
    if start_index<end_index:
      answer = encoding["input_ids"][0][start_index : end_index + 1]
      answer = processor.decode(answer)
    else:
      answer = encoding["input_ids"][0][end_index : start_index + 1]
      answer = processor.decode(answer)

    return answer


def infer(args):
    image_path=args.image_input
    model_path=args.model_path
    q=args.question

    # load the models
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    processor = LayoutLMv2Processor.from_pretrained(model_path)
    
    image = Image.open(image_path).convert("RGB")
    output=layoutlm(model,processor,q,image)

    print(output)

def parse_args():
    parser = argparse.ArgumentParser(description="Model INFER", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--image_input", type=str, default=None, help="path to the image dir")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="path to the layoutLM model")
    parser.add_argument("-q", "--question", type=str, default=None, help="question")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    infer(args)