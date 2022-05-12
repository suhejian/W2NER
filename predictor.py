import argparse
import config
from transformers import AutoTokenizer
import data_loader
from torch.utils.data import DataLoader
from model import Model
import torch
import os
import json
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import utils
from estimate_entity_prob import estimate_entity_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/conll03.json')
    parser.add_argument('--save_path', type=str, default='./model.pt')
    parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--log_path', type=str)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)
    logger = utils.get_logger(config.log_path)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)
    
    train_loader, dev_loader, test_loader = (
    DataLoader(dataset=dataset,
                batch_size=config.batch_size,
                collate_fn=data_loader.collate_fn,
                shuffle=i == 0,
                num_workers=4,
                drop_last=i == 0)
    for i, dataset in enumerate(datasets))
    
    with open("./data/{}/test.json".format(config.dataset), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    
    model = Model(config)
    model.load_state_dict(torch.load(config.save_path))
    
    
    # 评估模式
    model.eval()
    model.cuda()

    pred_result = []
    label_result = []

    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            entity_text = data_batch[-1]
            label_result.append(entity_text)
            data_batch = [data.cuda() for data in data_batch[:-1]]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            arg_outputs = torch.argmax(outputs, -1)
            predictions = utils.get_predictions(arg_outputs.cpu().numpy(), sent_length.cpu().numpy())
            
            entity_and_prob = estimate_entity_prob(model_outputs=outputs, predictions=predictions)
            pred_result.append(entity_and_prob)
            
    labels = []
    preds = []
    for i in range(len(pred_result)):
        labels.extend(label_result[i])
        preds.extend(pred_result[i])
        
        
    if not os.path.exists("./predictions/"):
        os.makedirs("./predictions/")
    
    with open(config.predict_path, "w", encoding="utf-8") as f:
        for i in range(len(test_data)):
            f.write("raw sentence: \n")
            f.write(" ".join(test_data[i]['sentence']) + "\n")
            f.write("true entities: \n")
            f.write(str(utils.get_entities(test_data[i]['sentence'], labels[i], config.vocab.id2label)) + "\n")
            f.write("predicted entities: \n")
            pred = [item[0] for item in preds[i]]
            f.write(str(utils.get_entities(test_data[i]['sentence'], pred, config.vocab.id2label)) + "\n")
            f.write("predicted entities prob: \n")
            prob = [item[1] for item in preds[i]]
            f.write(str(prob) + "\n")
            f.write("\n")
            
