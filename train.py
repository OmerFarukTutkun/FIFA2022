import torch
from torch import nn
from torch import optim
import time
import loaddata
import predict
BATCH_SIZE = 64
LR = 0.001
NUM_EPOCH = 10
def main():
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    train_dataset =  loaddata.FIFADataset(filename='results.csv')
    train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=BATCH_SIZE, shuffle=True )
  
    model = nn.Sequential(
                nn.Linear(2*train_dataset.get_numOfTeams() , 128),
                nn.Linear(128 , 1),
                nn.Sigmoid(),
            ).to(device=device)

    loss_fn = nn.BCELoss()
    learning_rate = LR
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, NUM_EPOCH+1):
        loss_list =  []
        start = time.time()
        for x, y in train_loader:
            
            outputs = model(x.to(device))
            loss = loss_fn(outputs, y.to(device))
            optimizer.zero_grad()

            loss_list.append( float( loss) )
            loss.backward()
            optimizer.step()
        end = time.time()
        print("Epoch: %d Number of batch: %d, Loss: %f Speed: %f sample/s" % (epoch, len(loss_list),  sum(loss_list)/ len(loss_list) , len(loss_list)*BATCH_SIZE/(end-start)))
        if epoch % 10 == 0:
            predict.FifaPredict(model, train_dataset , device)
if __name__ == "__main__":
    main()