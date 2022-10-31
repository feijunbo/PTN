import torch
import torch.nn as nn
import torch.nn.functional as F

from datamanager import recombine_fusion

class RelModel(nn.Module):
    def __init__(self):
        super(RelModel, self).__init__()
        self.classifier = nn.Linear(768, 2)

    def forward(self, s_pool, training):
        inp = F.dropout(s_pool, training=training)
        prob = F.softmax(self.classifier(inp), dim=1)
        prob = prob.mean(0)
        return prob

class TriModel(nn.Module):
    def __init__(self):
        super(TriModel, self).__init__()
        self.classifier = nn.Linear(768, 2)

    def forward(self, s_pool, training):
        inp = F.dropout(s_pool, training=training)
        prob = F.softmax(self.classifier(inp), dim=1)
        prob = prob.mean(0)
        return prob

class EntModel(nn.Module):
    def __init__(self):
        super(EntModel, self).__init__()
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(768, 768, dtype=torch.float)),
                              requires_grad=True)

    def forward(self, protos, word_vec, training):
        inp1 = torch.mm(protos.view(3, 768), self.W)
        inp2 = torch.mm(inp1, word_vec.view(768, 1))
        prob = F.softmax(inp2, dim=0)
        return prob 

class Model(nn.Module):
    def __init__(self, bert, N, K, tokenizer, na=False, multi=False):
        super(Model, self).__init__()
        self.N = N
        self.K = K
        self.na = na
        self.multi = multi
        self.relModel = RelModel()
        self.entModel = EntModel()
        self.triModel = TriModel()
        self.bert = bert
        self.tokenizer = tokenizer
    
    def sample(self, prob, training, preoptions, position):
        if not training:
            return torch.argmax(prob, 0, keepdim=True)
        else:
            return torch.cuda.LongTensor(1, ).fill_(preoptions[position])

    def forward(self, mode, text, preoptions=None, preactions=None):
        support = text['support']
        fusion = text['fusion']
        query = text['query']
        s_token = torch.cuda.LongTensor(fusion['token'])
        s_mask = torch.cuda.LongTensor(fusion['mask'])
        s_seg = torch.cuda.LongTensor(fusion['seg'])
        s_indexed_mask = fusion['indexed_mask']
        s_tag = support['tag']
        

        rel_action, rel_actprob = [], []
        ent_action, ent_actprob = [], []
        tri_action, tri_actprob = [], []
        training = True if "test" not in mode else False

        #-----------------------------------------------------------------
        # Prepare
        s_sent = []
        s_pool = []
        idx = 0
        while idx < len(s_token):
            sent, pool = self.bert(s_token[idx:min(idx+5,len(s_token))], token_type_ids=s_seg[idx:min(idx+5,len(s_token))], attention_mask=s_mask[idx:min(idx+5,len(s_token))], return_dict=False)
            s_sent.append(sent)
            s_pool.append(pool)
            idx += 5
        s_sent = torch.cat(s_sent, dim=0)
        s_pool = torch.cat(s_pool, dim=0)
        s_pool = s_pool.view(self.N, self.K, -1)

        #-----------------------------------------------------------------
        # Relation Perspective
        action = torch.cuda.LongTensor(1, ).fill_(0)
        best_actprob = 0
        best_rel = 0
        cnt = 0
        for x in range(len(s_pool)):                   
            prob = self.relModel(s_pool[x], training)
            action = self.sample(prob.view(-1), training, preoptions, x)
            actprob = prob[action]
            rel_action.append(action.cpu().item())
            rel_actprob.append(actprob)

            if not training:
                if action == 0:
                    actprob = 1 - actprob
                if actprob > best_actprob:
                    best_actprob = actprob
                    best_rel = x
            #----------------------------------------------------------------
            # Entity Perspective
            if "NER" in mode and (action.item() > 0 or (not training)):
                c_s_mask = s_mask[x*self.K:(x+1)*self.K]
                c_s_seg = s_seg[x*self.K:(x+1)*self.K][c_s_mask==1]
                c_s_sent = s_sent[x*self.K:(x+1)*self.K][c_s_mask==1][c_s_seg==0]
                c_q_sent = s_sent[x*self.K:(x+1)*self.K][c_s_mask==1][c_s_seg==1]
                t_s_indexed_mask = []
                for i in range(x*self.K,(x+1)*self.K):
                    t_s_indexed_mask.append(torch.cuda.LongTensor(s_indexed_mask[i]))
                t_s_indexed_mask = torch.cat(t_s_indexed_mask, 0)
                c_s_indexed_mask = t_s_indexed_mask[c_s_seg==0]
                c_q_indexed_mask = t_s_indexed_mask[c_s_seg==1]

                c_s_tag = []
                for i in range(x*self.K,(x+1)*self.K):
                    c_s_tag.append(torch.cuda.LongTensor(s_tag[i]))
                c_s_tag = torch.cat(c_s_tag, 0)

                c_s_sent = c_s_sent[c_s_indexed_mask == 1]
                c_q_sent = c_q_sent[c_q_indexed_mask == 1].view(self.K, -1, 768)
                c_q_sent = c_q_sent[0]
                head_ent = c_s_sent[c_s_tag == 1].mean(0)
                tail_ent = c_s_sent[c_s_tag == 2].mean(0)
                none_ent = c_s_sent[c_s_tag == 0].mean(0)
                # none_ent = self.noent(torch.cuda.LongTensor(1, ).fill_(0))
                protos = torch.stack([none_ent, head_ent, tail_ent])
                actionb = torch.cuda.LongTensor(1, ).fill_(0)
                actions, actprobs = [], []
                true_actions, true_actprobs = [], []
                for y in range(len(c_q_sent)):
                    probb = self.entModel(protos, c_q_sent[y], training)
                    actionb = self.sample(probb.view(-1), training, preactions[x] if preactions is not None else None, y)
                    true_actionb = self.sample(probb.view(-1), False, None, y)
                    actprobb = probb[actionb]
                    true_actprobb = probb[true_actionb]
                    actions.append(actionb.cpu().item())
                    actprobs.append(actprobb)
                    true_actions.append(true_actionb.cpu().item())
                    true_actprobs.append(true_actprobb)
                
                # Triple Perspective
                new_actions, new_actprobs = [], []
                new_fusion = recombine_fusion(true_actions, true_actprobs, support['token'][x*self.K:(x+1)*self.K], query, self.tokenizer, cnt, training)
                new_s_token = torch.cuda.LongTensor(new_fusion['token'])
                new_s_mask = torch.cuda.LongTensor(new_fusion['mask'])
                new_s_seg = torch.cuda.LongTensor(new_fusion['seg'])
                new_s_pool = []
                idx = 0
                while idx < len(new_s_token):
                    _, pool = self.bert(new_s_token[idx:min(idx+5,len(new_s_token))], token_type_ids=new_s_seg[idx:min(idx+5,len(new_s_token))], attention_mask=new_s_mask[idx:min(idx+5,len(new_s_token))], return_dict=False)
                    new_s_pool.append(pool)
                    idx += 5
                new_s_pool = torch.cat(new_s_pool, dim=0)
                new_s_pool = new_s_pool.view(-1, self.K, 768)
                for x2 in range(len(new_s_pool)):
                    new_prob = self.triModel(new_s_pool[x2], training)
                    new_action = self.sample(new_prob.view(-1), training, new_fusion['new_rel'], x2)
                    new_actprob = new_prob[new_action]
                    new_actions.append(new_action.cpu().item())
                    new_actprobs.append(new_actprob)
                ent_action.append(actions)
                ent_actprob.append(actprobs)
                tri_action.append(new_actions)
                tri_actprob.append(new_actprobs)
                cnt += 1
        
        if not training:
            for i in range(len(rel_action))[::-1]:
                if (self.multi == False and i != best_rel) or (self.multi == True and rel_action[i] == 0 and i != best_rel):
                # if i != best_rel:
                    rel_action[i] = 0
                    if "NER" in mode:
                        ent_action.pop(i)
                        tri_action.pop(i)
                elif not self.na:
                    rel_action[i] = 1
        return rel_action, rel_actprob, ent_action, ent_actprob, tri_action, tri_actprob

