import torch

def calcRelationGrad(rel_action, rel_actprob, alpha=0.3):
    lenth = len(rel_action)
    grads = torch.cuda.FloatTensor(1, ).fill_(0)
    for i in range(lenth)[::-1]:
        to_grad = -torch.log(rel_actprob[i])
        if rel_action[i] == 0:
            to_grad *= alpha
        grads = grads + to_grad
    return grads

def calcEntityGrad(rel_action, ent_action, ent_actprob, beta=0.3):
    grads = torch.cuda.FloatTensor(1, ).fill_(0)
    j = 0
    for i in range(len(rel_action)):
        if rel_action[i] > 0:
            for k in range(len(ent_action[j]))[::-1]:
                to_grad = -torch.log(ent_actprob[j][k])
                if ent_action[j][k] == 0:
                    to_grad *= beta
                else:
                    to_grad *= 1.0
                grads = grads + to_grad
            j += 1
    
    return grads

def calcTripleGrad(rel_action, tri_action, tri_actprob, gamma):
    grads = torch.cuda.FloatTensor(1, ).fill_(0)
    j = 0
    for i in range(len(rel_action)):
        if rel_action[i] > 0:
            for k in range(len(tri_action[j]))[::-1]:
                to_grad = -torch.log(tri_actprob[j][k]) 
                if tri_action[j][k] == 0:
                    to_grad *= gamma
                else:
                    to_grad *= 1.0
                grads = grads + to_grad
            j += 1
    
    return grads

def optimize(rel_action, rel_actprob, ent_action, ent_actprob, tri_action, tri_actprob, mode, alpha=0.3, beta=0.4, gamma=0.8):
    if "NER" in mode:
        grads = calcEntityGrad(rel_action, ent_action, ent_actprob, beta)
        grads += calcTripleGrad(rel_action, tri_action, tri_actprob, gamma)
    else:
        grads = torch.cuda.FloatTensor(1, ).fill_(0)
    if "RE" in mode:
        grads += calcRelationGrad(rel_action, rel_actprob, alpha)
    loss = grads.cpu().item()
    grads.backward()
    return loss

def optimize_round(rel_action, rel_actprob, ent_action, ent_actprob, tri_action, tri_actprob, mode, alpha, beta, gamma):
    loss = optimize(rel_action, rel_actprob, ent_action, \
                ent_actprob, tri_action, tri_actprob, mode, alpha, beta, gamma)
    return loss
