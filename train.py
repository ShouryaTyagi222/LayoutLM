from transformers import AutoModelForQuestionAnswering
from transformers import AdamW

import numpy as np
import os
import json
from tqdm import tqdm

import torch
from sklearn.model_selection import train_test_split

from config import Config
from data.data_loader import *

    
def main(args):
    print('Loading Configs')
    image_dir=args.image_dir_path
    output_dir=os.path.join(args.output_path,args.model)
    model_checkpoint = args.init_checkpoint
    batch_size = args.batch_size
    banned_txt_path = args.banned_txt_path
    input_file=args.data_path
    n_epochs= args.epochs
    learning_rate=args.learning_rate
    data_split=args.data_split
    gpu_device=args.device

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

    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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

    checkpoint_path = os.path.join(output_dir,"checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

    print('Train Start')

    if args.model=='LayoutLMv2':
        # Log Losses to a file
        with open(os.path.join(output_dir,"losses_lmv2_combined.txt"), "w") as f:
            f.write("")

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            error_file_count=0
            model.train()
            Loss = []
            progbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{n_epochs}', unit='batch')
            for batch in progbar:
                try:
                    # get the inputs;
                    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
                    bbox = batch["bbox"].to(device=device, dtype=torch.long)  # No need to specify data type
                    image = batch["image"].to(device=device, dtype=torch.float32)  # Assuming image data type is float32
                    start_positions = batch["start_positions"].to(device=device, dtype=torch.long)  # No need to specify data type
                    end_positions = batch["end_positions"].to(device=device, dtype=torch.long)  # No need to specify data type


                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(input_ids=input_ids, bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
                    loss = outputs.loss
                    # print("Loss:", loss.item())
                    loss.backward()
                    optimizer.step()
                    Loss.append(loss.item())
                    progbar.set_description("Train Loss = %0.3f," % (np.mean(Loss)))
                except Exception as e:
                    print(e)
                    error_file_count+=1

                
            print('THE THE NUMBER OF FILES WHICH GOT ERROR :',error_file_count,'OUT OF :',len(train_dataloader))
            
            Loss = np.mean(Loss)
            # Print the loss
            print("Epoch:", epoch, "Loss:", Loss)

            with open(os.path.join(output_dir,"losses_lmv2_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Train_Loss: {Loss}\n")

            model.eval()
            Test_Loss=[]
            error_file_count=0
            progbar = tqdm(test_dataloader, desc=f'Epoch {epoch}/{n_epochs}', unit='batch')
            for batch in progbar:
                try:
                    # get the inputs;
                    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
                    bbox = batch["bbox"].to(device=device, dtype=torch.long)  # No need to specify data type
                    image = batch["image"].to(device=device, dtype=torch.float32)  # Assuming image data type is float32
                    start_positions = batch["start_positions"].to(device=device, dtype=torch.long)  # No need to specify data type
                    end_positions = batch["end_positions"].to(device=device, dtype=torch.long)  # No need to specify data type

                    # forward + backward + optimize
                    outputs = model(input_ids=input_ids, bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
                    loss = outputs.loss
                    # print("Loss:", loss.item())
                    Test_Loss.append(loss.item())
                    progbar.set_description("Val Loss = %0.3f," % (np.mean(Test_Loss)))
                except:
                    error_file_count+=1

            print('THE THE NUMBER OF FILES WHICH GOT ERROR :',error_file_count,'OUT OF :',len(train_dataloader))
            
            Test_Loss = np.mean(Test_Loss)
            # Print the loss
            print("Epoch:", epoch, "Loss:", Test_Loss)

            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
            }
            torch.save(checkpoint, checkpoint_path)

            # Log the loss
            with open(os.path.join("losses_lmv2_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Test_Loss: {Loss}\n")

        # Save the model
        model.save_pretrained(os.path.join(output_dir,"layoutlmv2b-finetuned"))
    
    elif args.model=='LayoutLMv3':
        # Log Losses to a file
        with open(os.path.join(output_dir,"losses_lmv3_combined.txt"), "w") as f:
            f.write("")
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            model.train()
            progbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{n_epochs}', unit='batch')
            Loss=[]
            for batch in progbar:
                # get the inputs;
                input_ids = batch["input_ids"].to(device=device, dtype=torch.long).long()
                bbox = batch["bbox"].to(device=device, dtype=torch.long)
                image = batch["image"].to(device, dtype=torch.float)
                start_positions = batch["start_positions"].to(device=device, dtype=torch.long)
                end_positions = batch["end_positions"].to(device=device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                # print("Loss:", loss.item())
                loss.backward()
                optimizer.step()
                Loss.append(loss.item())
                progbar.set_description("Train Loss = %0.3f," % (np.mean(Loss)))
            
            Loss = np.mean(Loss)
            # Print the loss
            print("Epoch:", epoch, "Loss:", Loss)

            with open(os.path.join(output_dir,"losses_lmv3_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Train_Loss: {Loss}\n")

            model.eval()
            Test_Loss = []
            progbar = tqdm(test_dataloader, desc=f'Epoch {epoch}/{n_epochs}', unit='batch')
            for batch in progbar:
                # get the inputs;
                input_ids = batch["input_ids"].to(device=device, dtype=torch.long).long()
                bbox = batch["bbox"].to(device=device, dtype=torch.long)
                image = batch["image"].to(device, dtype=torch.float)
                start_positions = batch["start_positions"].to(device=device, dtype=torch.long)
                end_positions = batch["end_positions"].to(device=device, dtype=torch.long)

                # forward + backward + optimize
                outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                # print("Loss:", loss.item())
                Test_Loss.append(loss.item())
                progbar.set_description("Val Loss = %0.3f," % (np.mean(Test_Loss)))
            
            Test_Loss = np.mean(Test_Loss)
            # Print the loss
            print("Epoch:", epoch, "Loss:", Test_Loss)

            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
            }
            torch.save(checkpoint, checkpoint_path)

            # Log the loss
            with open(os.path.join("losses_lmv3_combined.txt"), "a") as f:
                f.write(f"Epoch: {epoch} Test_Loss: {Loss}\n")

        # Save the model
        model.save_pretrained(os.path.join(output_dir,"layoutlmv3b-finetuned"))


    else:
        print('ENTER AN APPROPRIATE MODEL NAME')
        

if __name__ == "__main__":
    # args = parse_args()
    args=Config()

    main(args)