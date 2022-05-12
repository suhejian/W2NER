""" 
估算一个实体的置信度
"""


def convert_entity_prediction_to_relations(entity_prediction):
    """
    将解码出来的字符串转换成关系矩阵中的位置, 这边的字符串仅仅是一个实体对应的字符串
    input:
        entity_prediction (str): 解码出来的字符串
    output:
        relation_list (list): 关系矩阵中有关系元素的列表
    """
    indexes = entity_prediction.split("-")
    assert indexes[-2] == "#", "Check the decoded string"
    
    relation_list = []
    
    tag_type = int(indexes[-1])
    start_to_end = [int(item) for item in indexes[: -2]]
    if len(start_to_end) == 1:
        # 该实体只有一个Token
        # 如果该实体的位置为i，那么矩阵中(i,i)位置对应的关系就是tag_type
        relation_list = [(start_to_end[0], start_to_end[-1], tag_type)]
        return relation_list
    # 否则，实体的长度是大于1的
    start, end = start_to_end[0], start_to_end[-1]
    for i in range(len(start_to_end) - 1):
        cur = start_to_end[i]
        nxt = start_to_end[i + 1]
        # 矩阵中(cur, nxt)位置对应的关系就是1，表示succession
        relation_list.append((cur, nxt, 1))
    relation_list.append((end, start, tag_type))
    
    return relation_list


def get_entity_prob(model_output, relation_list):
    """ 
    根据模型的输出和关系列表中的元素，估算一个实体的概率
    input:
        model_output: 模型的输出
        relation_list: 一个实体对应的关系列表
    output:
        prob: 该实体的置信度
    """
    import torch.nn.functional as F
    
    total_prob = 0
    for relation in relation_list:
        start, end, tag = relation
        logits = model_output[start][end]   # tag nums维度的向量
        tag_prob = F.softmax(logits)[tag]
        # print(tag_prob)
        total_prob += tag_prob
    
    entity_prob = total_prob / len(relation_list)
    # print(entity_prob)
    entity_prob = entity_prob.detach().cpu().item()
    
    return entity_prob


def estimate_entity_prob(model_outputs, predictions):
    """ 
    根据模型的输出和预测出的实体(以索引关系字符串形式给出)估计每个实体的置信度
    input: 
        model_outputs (tensor): 模型输出, argmax之前的结果 [batch_size, seq_len, seq_len, tag_num]
        predictions (list): 每句话对应的解码结果，长度为batch_size，每个元素是字符串集合
    """
    
    # 存储实体和对应的置信度
    entity_and_prob = []
    for i in range(len(predictions)):
        # sent_prediction是每个句子的解码结果
        sent_prediction = predictions[i]
        # model_output是该句子对应的模型输出
        model_output = model_outputs[i]
        
        # 存储每个句子的结果
        res = []
        for entity_prediction in sent_prediction:
            # entity_prediction是每个实体对应的解码字符串
            # 得到该实体对应的关系列表
            relation_list = convert_entity_prediction_to_relations(entity_prediction)
            
            # 根据模型对 该句子的输出结果 和 该实体对应的关系列表 估计这个实体的置信度
            entity_prob = get_entity_prob(model_output, relation_list)
            
            # 把该实体对应的结果添加到句子级别
            res.append((entity_prediction, entity_prob))
        
        # 把该句子对应的结果添加到batch级别
        entity_and_prob.append(res)
       
    return entity_and_prob
        
    
    