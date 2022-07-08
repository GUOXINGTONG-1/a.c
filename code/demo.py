# -*- coding: utf-8 -*-
import torch
from sqlnet.nl2sql import *
from modle.sqlbert import SQLBert, BertAdam, BertTokenizer
import argparse
import os.path as osp
import json

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--gpu', action='store_true', help='Whether use gpu')
	parser.add_argument('--batch_size', type=int, default=12)

	parser.add_argument('--data_dir', type=str, default='/tcdata/')
	parser.add_argument('--bert_model_dir', type=str, default='./model/chinese-bert_chinese_wwm_pytorch/')
	parser.add_argument('--restore_model_path', type=str, default='./model/best_bert_model')

    
	args = parser.parse_args()
	gpu = args.gpu
	batch_size = args.batch_size
	data_dir = args.data_dir
	bert_model_dir = args.bert_model_dir
	restore_model_path = args.restore_model_path
	test_sql_path = osp.join(data_dir, 'final_test_1.json')
	answer_path = osp.join(data_dir, 'answer_1.json')
	test_table_path = osp.join(data_dir, 'final_test.tables.json')
	test_sql, test_table = loading_data(test_sql_path, test_table_path)
	test_db_path = osp.join(data_dir, 'final_test.db')
	print(test_db_path)    
	'''
	#tid_x = int(input("请输入要查询表格的类型：1：金融，2：贸易，3：巴拉巴拉……\n"))
	tid_x = 1
	with open(test_sql_path, 'w') as f:
		if tid_x == 1:
			tid = "c98c2ca1332111e99c14542696d6e445"
			
		#question = input("问题：")
		question = "2012年都有哪些涨价楼盘"
		fi = f'{{"table_id": "{tid}", "question": "{question}"}}'
		f.writelines(str(fi))
		f.close()
	'''
	with open(test_sql_path, 'r') as f:
		tid_dict = json.load(f)
		tid = tid_dict['table_id']
		f.close()
	print(tid)
	tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
	model = SQLBert.from_pretrained(bert_model_dir)
	print("Loading from %s" % restore_model_path)
	model.load_state_dict(torch.load(restore_model_path))
	print("Loaded model from %s" % restore_model_path)
	sql = get_sql(model, batch_size, test_sql, test_table, test_db_path, tokenizer=tokenizer)
	anwser = answer(model, batch_size, test_sql, test_table, tid, test_db_path, tokenizer=tokenizer)
	test_dict = {'SQL': f'{sql}', 'answer': anwser}
	print(test_dict)
	with open(answer_path, 'w') as write_f:
		json.dump(test_dict, write_f, indent=4, ensure_ascii=False)
	
	print("SQL::",str(sql))
	for item in anwser:
		list2 = [str(i) for i in item]
		item = ' '.join(list2)
		print("答案::", item)


