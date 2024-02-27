from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import AdamW

import numpy as np
import os
import json
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import wandb

from config import Config
from data.data_loader import *

def get_output(batch,model,model_type,device):
    if model_type.lower()=='layoutlmv2':
        input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
        bbox = batch["bbox"].to(device=device, dtype=torch.long)  # No need to specify data type
        image = batch["image"].to(device=device, dtype=torch.float32)  # Assuming image data type is float32
        start_positions = batch["start_positions"].to(device=device, dtype=torch.long)  # No need to specify data type
        end_positions = batch["end_positions"].to(device=device, dtype=torch.long)  # No need to specify data type

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)

    elif model_type.lower()=='layoutlmv3':
        # get the inputs;
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        image = batch["image"].to(device, dtype=torch.float)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image, start_positions=start_positions, end_positions=end_positions)

    else:
        print('ENTER THE CORRECT NAME OF THE MODEL !!')
    
    return outputs
    
def main(args):
    print('Loading Configs')
    model=args.model
    image_dir=args.image_dir_path
    output_dir=os.path.join(args.output_path,args.model)
    model_checkpoint = args.init_checkpoint
    tokenizer_checkpoint=args.tokenizer_checkpoint
    batch_size = args.batch_size
    banned_txt_path = args.banned_txt_path
    input_file=args.data_path
    n_epochs= args.epochs
    learning_rate=args.learning_rate
    data_split=args.data_split
    gpu_device=args.device
    wandb_key=args.wandb_key
    wandb_project=args.wandb_project_desc
    wandb_model=args.wandb_model_desc
    wandb_name=args.wandb_name
    wandb_flag=args.wandb_flag

    if wandb_flag:
        wandb.init(project=wandb_project, name=wandb_name)
        wandb.login(key=wandb_key)
        wandb.config.update({"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": n_epochs, "model": wandb_model})
        # Log in to WandB
        wandb.login()

    print(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # A Text file contains all the files that are not to be processed
    with open(banned_txt_path) as f:
        banned_files = f.readlines()

    banned_files = [x.strip() for x in banned_files]

    print('Loading the Data')

    encoded_dataset = json.load(open(input_file))
    encoded_dataset = convert_to_custom_format(encoded_dataset,image_dir,banned_files)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    # tokenizer.decode(encoded_dataset[0]["input_ids"])

    train_dataset, test_dataset = train_test_split(encoded_dataset, test_size=data_split, random_state=42)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate)

    # Print the size of both datasets
    print("Length of Train Set", len(train_dataset))
    print("Length of Test Set", len(test_dataset))

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        device = "cuda" 
        torch.cuda.set_device(gpu_device)
    else:
        device="cpu"
    model.to(device)
    # rouge = Rouge()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction()
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    checkpoint_path = os.path.join(output_dir,"checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

    print('Train Start')

    with open(os.path.join(output_dir,f"{args.model}_logs.txt"), "w") as f:
        f.write("")
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        progbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}, Train Loss = 0, current loss = 0 , Bleu Score = 0', unit='batch')
        Loss=[]
        i=0
        bleu_scores=[0]
        o_answers=[]
        p_answers=[]
        for batch in progbar:
            # get the inputs;
            # if i==3:
            #     break
            # i+=1
            input_ids = batch["input_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = get_output(batch,model,args.model,device)

            # zero the parameter gradients
            optimizer.zero_grad()

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            start_pos = start_positions[0].item()
            end_pos = end_positions[0].item()
            ori_answer = input_ids[0][start_pos : end_pos + 1]
            ori_answer = tokenizer.decode(ori_answer)
            
            # Predicted Answer
            start_index = torch.argmax(outputs.start_logits[0], dim=-1).item()
            end_index = torch.argmax(outputs.end_logits[0], dim=-1).item()

            # Slice the input_ids tensor using scalar indices
            if start_index < end_index:
                p_answer = input_ids[0][start_index:end_index + 1]
            else:
                p_answer = input_ids[0][end_index:start_index + 1]

            # print('original answer :',ori_answer)

            # Decode the predicted answer
            p_answer = tokenizer.decode(p_answer)
            # print('predicted answer :',p_answer)

            bleu_score = corpus_bleu([ori_answer.split()], [p_answer.split()], smoothing_function=smoother.method1)
            bleu_scores.append(bleu_score)

            o_answers.append(ori_answer)
            p_answers.append(p_answer)

            Loss.append(loss.item())
            progbar.set_description("Epoch : %s/%s, Train Loss : %0.3f, current Loss : %0.3f, BLEU Score : %0.3f," % (epoch+1, n_epochs, np.mean(Loss), loss.item(), np.mean(bleu_scores)))
            
        rouge1_p=[]
        rouge1_r=[]
        rouge1_f=[]
        rouge2_p=[]
        rouge2_r=[]
        rouge2_f=[]
        rougel_p=[]
        rougel_r=[]
        rougel_f=[]

        for hypothesis, reference in zip(o_answers, p_answers):
            scores = scorer.score(hypothesis, reference)
            rouge1_p.append(scores['rouge1'].precision)
            rouge1_r.append(scores['rouge1'].recall)
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_p.append(scores['rouge2'].precision)
            rouge2_r.append(scores['rouge2'].recall)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougel_p.append(scores['rougeL'].precision)
            rougel_r.append(scores['rougeL'].recall)
            rougel_f.append(scores['rougeL'].fmeasure)
        
        rouge1_p=np.mean(rouge1_p)
        rouge1_r=np.mean(rouge1_r)
        rouge1_f=np.mean(rouge1_f)
        rouge2_p=np.mean(rouge2_p)
        rouge2_r=np.mean(rouge2_r)
        rouge2_f=np.mean(rouge2_f)
        rougel_p=np.mean(rougel_p)
        rougel_r=np.mean(rougel_r)
        rougel_f=np.mean(rougel_f)

        print(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f},')

        with open(os.path.join(output_dir,f"{args.model}_logs.txt"), "a") as f:
            f.write("Epoch = %s/%s Test Loss = %0.3f, BLEU Score = %0.3f \n" % (epoch+1, n_epochs, np.mean(Loss), np.mean(bleu_scores)))
            f.write(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f},')

        scheduler.step()
        print(f'Epoch : {epoch+1}, learning rate : {optimizer.param_groups[0]["lr"]}')

        train_loss=np.mean(Loss)

        model.eval()
        Loss = []
        bleu_scores=[]
        o_answers=[]
        p_answers=[]
        progbar = tqdm(test_dataloader, desc=f'Epoch {epoch}/{n_epochs}', unit='batch')
        for batch in progbar:
            # get the inputs;
            input_ids = batch["input_ids"].to(device)
            # bbox = batch["bbox"].to(device)
            # image = batch["image"].to(device, dtype=torch.float)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # # forward + backward + optimize
            # outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image, start_positions=start_positions, end_positions=end_positions)

            outputs = get_output(batch,model,args.model,device)

            loss = outputs.loss

            start_pos = start_positions[0].item()
            end_pos = end_positions[0].item()
            ori_answer = input_ids[0][start_pos : end_pos + 1]
            ori_answer = tokenizer.decode(ori_answer)
            # Predicted Answer
            start_index = torch.argmax(outputs.start_logits[0], dim=-1).item()
            end_index = torch.argmax(outputs.end_logits[0], dim=-1).item()

            # Slice the input_ids tensor using scalar indices
            if start_index < end_index:
                p_answer = input_ids[0][start_index:end_index + 1]
            else:
                p_answer = input_ids[0][end_index:start_index + 1]

            # Decode the predicted answer
            p_answer = tokenizer.decode(p_answer)

            bleu_score = corpus_bleu([ori_answer.split()], [p_answer.split()], smoothing_function=smoother.method1)
            o_answers.append(ori_answer)
            p_answers.append(p_answer)
            Loss.append(loss.item())
            bleu_scores.append(bleu_score)
            progbar.set_description("Epoch : %s/%s Test Loss : %0.3f, BLEU Score : %0.3f," % (epoch+1, n_epochs, np.mean(Loss), np.mean(bleu_scores)))
        
        rouge1_p=[]
        rouge1_r=[]
        rouge1_f=[]
        rouge2_p=[]
        rouge2_r=[]
        rouge2_f=[]
        rougel_p=[]
        rougel_r=[]
        rougel_f=[]

        for hypothesis, reference in zip(o_answers, p_answers):
            scores = scorer.score(hypothesis, reference)
            rouge1_p.append(scores['rouge1'].precision)
            rouge1_r.append(scores['rouge1'].recall)
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_p.append(scores['rouge2'].precision)
            rouge2_r.append(scores['rouge2'].recall)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougel_p.append(scores['rougeL'].precision)
            rougel_r.append(scores['rougeL'].recall)
            rougel_f.append(scores['rougeL'].fmeasure)
        
        rouge1_p=np.mean(rouge1_p)
        rouge1_r=np.mean(rouge1_r)
        rouge1_f=np.mean(rouge1_f)
        rouge2_p=np.mean(rouge2_p)
        rouge2_r=np.mean(rouge2_r)
        rouge2_f=np.mean(rouge2_f)
        rougel_p=np.mean(rougel_p)
        rougel_r=np.mean(rougel_r)
        rougel_f=np.mean(rougel_f)

        print(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f},')

        with open(os.path.join(output_dir,f"{args.model}_logs.txt"), "a") as f:
            f.write("Epoch = %s/%s Test Loss = %0.3f, BLEU Score = %0.3f \n" % (epoch+1, n_epochs, np.mean(Loss), np.mean(bleu_scores)))
            f.write(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f},')

        if wandb_flag:
            wandb.log({
                "Epoch":epoch+1,
                "Training Loss": train_loss,
                "Testing Loss": np.mean(Loss),
                "Bleu Score":  np.mean(bleu_scores),
                "Rouge-1 Recall": rouge1_p,
                "Rouge-1 Precision": rouge1_r,
                "Rouge-1 F1 Score": rouge1_f,
                "Rouge-2 Recall": rouge2_p,
                "Rouge-2 Precision": rouge2_r,
                "Rouge-2 F1 Score": rouge2_f,
                "Rouge-l Recall":rougel_p,
                "Rouge-l Precision": rougel_r,
                "Rouge-l F1 Score": rougel_f,
                
            })

        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': Loss,
        }
        torch.save(checkpoint, checkpoint_path)


    # Save the model
    model.save_pretrained(os.path.join(output_dir,f"{args.model}-finetuned"))

if __name__ == "__main__":
    # args = parse_args()
    args=Config()

    main(args)