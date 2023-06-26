import torch
from icecream import ic

class CustomCrossEntropyLoss(torch.nn.Module):
    
    def __init__(self, weights) -> None:
        super(CustomCrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weights.float())
    
    def forward(self, output, target, accuracy):
        prediction = torch.argmax(accuracy)

        target[:,0] = (target[:,-1] == False) * target[:,0] + torch.where(prediction == 0, 1, 1) * target[:,0]
        target[:,1] = (target[:,-1] == False) * target[:,1] + torch.where(prediction == 1, 1, 1) * target[:,1]
        target[:,2] = (target[:,-1] == False) * target[:,2] + torch.where(prediction == 2, 1, 1) * target[:,2]
        target[:,3] = (target[:,-1] == False) * target[:,3] + torch.where(prediction == 3, 1, 1) * target[:,3]
        target[:,4] = (target[:,-1] == False) * target[:,4] + torch.where(prediction == 4, 1, 1) * target[:,4]

        target[:,5] = torch.sum(target[:,:-1], dim=1) == 0
        
        adapted_target = torch.argmax(target, dim=1)
        
        return self.cross_entropy_loss(output, adapted_target), adapted_target

if __name__ == "__main__":
    prediction = torch.tensor([
                            [2, 2, 4, 0, 0, 0, 4, 4],
                            [4, 2, 4, 5, 5, 5, 4, 4],
                            [5, 2, 4, 5, 0, 5, 4, 4],
                            [2, 2, 4, 2, 2, 1, 1, 5],
                            [2, 0, 0, 2, 2, 2, 1, 1],
                            [2, 0, 3, 3, 3, 1, 3, 3],
                            [2, 3, 3, 3, 0, 0, 3, 3],
                            [4, 0, 0, 0, 0, 0, 3, 3]])
    
    target = torch.tensor([
                           [[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]],
                           [[1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1]]])
    
    ic(target.shape, prediction.shape)

    target[0] = (target[-1] == False) * target[0] + torch.where(prediction == 0, 1, 0) * target[0]
    target[1] = (target[-1] == False) * target[1] + torch.where(prediction == 1, 1, 0) * target[1]
    target[2] = (target[-1] == False) * target[2] + torch.where(prediction == 2, 1, 0) * target[2]
    target[3] = (target[-1] == False) * target[3] + torch.where(prediction == 3, 1, 0) * target[3]
    target[4] = (target[-1] == False) * target[4] + torch.where(prediction == 4, 1, 0) * target[4]

    target[5] = torch.sum(target[:-1], dim=0) == 0

    ic(target)

    # ic(torch.argmin(target, dim=0), torch.argmax(target, dim=0))
    # ic(torch.eq(prediction, torch.argmin(target,dim=0)))
    # ic(torch.eq(prediction,  torch.argmax(target, dim=0)))