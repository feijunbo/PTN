from numpy import mean


def calcF1(acc, cnt, tot, beta=1.0):
    if cnt == 0 or tot == 0:
        return 0
    precision = float(acc) / float(cnt)
    recall = float(acc) / float(tot)
    if precision + recall < 1e-5:
        return 0
    return (1+beta*beta) * precision * recall / (beta*beta*precision + recall)

def calc_acc(top_action, bot_action, gold_labels, mode):
    types = gold_labels['type']
    tags = gold_labels['tags']
    acc, cnt, tot = 0, 0, 0
    for type in types:
        if type == 1:
            tot += 1
    j = 0
    k = 0
    for i in range(len(top_action)):
        ok = 0
        if types[i] > 0:
            if top_action[i] == types[i]:
                if "NER" in mode:
                    if tags[k] == bot_action[j]:
                        ok = 1
                else:
                    ok = 1
            k += 1
        if top_action[i] > 0:
            j += 1
            cnt += 1
        acc += ok
    return acc, tot, cnt	

def calc_rel_acc(top_action, bot_action, gold_labels, mode):
    types = gold_labels['type']
    acc, cnt, tot = 0, 0, 0
    for type in types:
        if type == 1:
            tot += 1
    for i in range(len(top_action)):
        ok = 0
        if types[i] > 0:
            if top_action[i] == types[i]:
                    ok = 1
        if top_action[i] > 0:
            cnt += 1
        acc += ok
    return acc, tot, cnt

def __get_class_span_dict__(label, is_string=False):
    '''
    return a dictionary of each class label/tag corresponding to the entity positions in the sentence
    {label:[(start_pos, end_pos), ...]}
    '''
    class_span = {}
    current_label = None
    i = 0
    if not is_string:
        # having labels in [0, num_of_class] 
        while i < len(label):
            if label[i] > 0:
                start = i
                current_label = label[i]
                i += 1
                while i < len(label) and label[i] == current_label:
                    i += 1
                if current_label in class_span:
                    class_span[current_label].append((start, i))
                else:
                    class_span[current_label] = [(start, i)]
            else:
                assert label[i] == 0
                i += 1
    else:
        # having tags in string format ['O', 'O', 'person-xxx', ..]
        while i < len(label):
            if label[i] != 'O':
                start = i
                current_label = label[i]
                i += 1
                while i < len(label) and label[i] == current_label:
                    i += 1
                if current_label in class_span:
                    class_span[current_label].append((start, i))
                else:
                    class_span[current_label] = [(start, i)]
            else:
                i += 1
    return class_span

def __get_intersect_by_entity__(pred_class_span, label_class_span):
    '''
    return the count of correct entity
    '''
    cnt = 0
    for label in label_class_span:
        cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
    return cnt

def __get_cnt__(label_class_span):
    '''
    return the count of entities
    '''
    cnt = 0
    for label in label_class_span:
        cnt += len(label_class_span[label])
    return cnt

def calc_ent_acc(top_action, bot_action, gold_labels, mode):
    types = gold_labels['type']
    tags = gold_labels['tags']
    acc, cnt, tot = 0, 0, 0

    j = 0
    k = 0
    for i in range(len(top_action)):
        label_cnt = 0
        pred_cnt = 0
        correct_cnt = 0
        label_class_span = {}
        pred_class_span = {}
        if types[i] > 0:
            label_class_span = __get_class_span_dict__(tags[k])
            label_cnt = __get_cnt__(label_class_span)
            k += 1
        if top_action[i] > 0:
            pred_class_span = __get_class_span_dict__(bot_action[j])
            pred_cnt = __get_cnt__(pred_class_span)
            j += 1
        correct_cnt = __get_intersect_by_entity__(pred_class_span, label_class_span)
        tot += label_cnt
        cnt += pred_cnt
        acc += correct_cnt
    return acc, tot, cnt

def calc_post_acc(top_action, bot_action, gold_labels, bot_prob, mode):
    types = gold_labels['type']
    tags = gold_labels['tags']
    acc, cnt, tot = 0, 0, 0
    for type in types:
        if type == 1:
            tot += 1
    j = 0
    k = 0
    for i in range(len(top_action)):
        ok = 0
        if types[i] > 0:
            if top_action[i] == types[i]:
                if "NER" in mode:
                    label_class_span = __get_class_span_dict__(tags[k])
                    pred_class_span = __get_class_span_dict__(bot_action[j])
                    for label in label_class_span:
                        max_prob = 0
                        max_prob_span = None
                        for span in pred_class_span.get(label,[]):
                            prob = mean([item.cpu().detach().item() for item in bot_prob[j][span[0]:span[1]]])
                            if prob > max_prob:
                                max_prob = prob
                                max_prob_span = span
                            bot_action[j][span[0]:span[1]] = [0] * (span[1] - span[0])
                        if max_prob_span is not None:
                            bot_action[j][max_prob_span[0]:max_prob_span[1]] = [label] * (max_prob_span[1] - max_prob_span[0])
                    if tags[k] == bot_action[j]:
                        ok = 1
                else:
                    ok = 1
            k += 1
        if top_action[i] > 0:
            j += 1
            cnt += 1
        acc += ok
    return acc, tot, cnt

def error_analysis(top_action, bot_action, gold_labels, mode):
    acc, cnt, tot = 0, 0, 0
    types = gold_labels['type']
    tags = gold_labels['tags']
    statictis = {'right':0, 'entity_error':0, 'relation_error':0}

    for type in types:
        if type == 1:
            tot += 1
    j = 0
    k = 0
    for i in range(len(top_action)):
        pred_cnt = 0
        correct_cnt = 0
        label_class_span = {}
        pred_class_span = {}
        ok = 0
        if types[i] > 0:
            if top_action[i] == types[i]:
                if tags[k] == bot_action[j]:
                    ok = 1
                    statictis['right'] += 1
                else:
                    statictis['entity_error'] += 1
                    pred_class_span = __get_class_span_dict__(bot_action[j])
                    pred_cnt = __get_cnt__(pred_class_span)
                    label_class_span = __get_class_span_dict__(tags[k])
                    correct_cnt = __get_intersect_by_entity__(pred_class_span, label_class_span)
                    key = str(correct_cnt) + '-' + str(pred_cnt - correct_cnt)
                    if not key in statictis:
                        statictis[key] = 0
                    statictis[key] += 1
            else:
                statictis['relation_error'] += 1
            k += 1
        if top_action[i] > 0:
            j += 1
            cnt += 1
        acc += ok
    if tot == 0:
        statictis[str(tot)+'_tot'] = 1
        statictis[str(tot)+'_cnt'] = 1
        statictis[str(tot)+'_acc'] = 1 if cnt == 0 else 0
    else:
        statictis[str(tot)+'_tot'] = tot
        statictis[str(tot)+'_cnt'] = cnt
        statictis[str(tot)+'_acc'] = acc
    return statictis

def error_detail(top_action, bot_action, gold_labels, bot_prob):
    types = gold_labels['type']
    tags = gold_labels['tags']
    raw_token = gold_labels['raw_token']
    detail = {}

    j = 0
    k = 0
    for i in range(len(top_action)):
        label_class_span = {}
        pred_class_span = {}
        if types[i] > 0:
            if top_action[i] == types[i]:
                pred_class_span = __get_class_span_dict__(bot_action[j])
            label_class_span = __get_class_span_dict__(tags[k])
            for label in label_class_span:
                detail[label] = {}
                detail[label]['true'] = label_class_span[label]
                detail[label]['true_token'] = []
                for span in label_class_span[label]:
                    detail[label]['true_token'].append(raw_token[0][span[0]:span[1]])
                right = list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[]))))
                wrong = list(set(pred_class_span.get(label,[])).difference(set(right)))
                detail[label]['rigth'] = right
                detail[label]['rigth_token'] = []
                detail[label]['rigth_prob'] = []
                detail[label]['rigth_prob_avg'] = []
                for span in right:
                    detail[label]['rigth_token'].append(raw_token[0][span[0]:span[1]])
                    detail[label]['rigth_prob'].append([item.cpu().detach().item() for item in bot_prob[j][span[0]:span[1]]])
                    detail[label]['rigth_prob_avg'].append(mean([item.cpu().detach().item() for item in bot_prob[j][span[0]:span[1]]]))
                detail[label]['wrong'] = wrong
                detail[label]['wrong_token'] = []
                detail[label]['wrong_prob'] = []
                detail[label]['wrong_prob_avg'] = []
                for span in wrong:
                    detail[label]['wrong_token'].append(raw_token[0][span[0]:span[1]])
                    detail[label]['wrong_prob'].append([item.cpu().detach().item() for item in bot_prob[j][span[0]:span[1]]])
                    detail[label]['wrong_prob_avg'].append(mean([item.cpu().detach().item() for item in bot_prob[j][span[0]:span[1]]]))
            k += 1
        if top_action[i] > 0:
            j += 1

    return detail

def rule_actions(gold_labels):
    options = gold_labels['type']
    actions = [[] for i in range(len(options))]
    j = 0
    for i in range(len(options)):
        if options[i] == 1:
            actions[i] = gold_labels['tags'][j]
            j += 1
    return options, actions

def update_statictis(statictis, error_statictis):
    ks = list(statictis.keys())
    eks = list(error_statictis.keys())
    allk = list(set(ks + eks))
    for k in allk:
        if k in ks and k in eks:
            error_statictis[k] += statictis[k]
        elif k in ks and k not in eks:
            error_statictis[k] = statictis[k]
        else:
            pass
