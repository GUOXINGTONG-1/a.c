# -*- coding: utf-8 -*-
# encoding=utf8 
import json
import records
import numpy as np
import copy
import re
import difflib
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from tqdm import tqdm
from fuzzywuzzy import process
from fuzzywuzzy.utils import StringProcessor
from collections import defaultdict
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertEncoder, BertAttention
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam, BertConfig
from collections import Counter



@lru_cache(None)
def my_scorer(t, c):
	return (1 - abs(len(t) - len(c)) / max(len(t), len(c))) * process.default_scorer(t, c)


def my_process(s):
    """Process string by
		-- removing all but letters and numbers
		-- trim whitespace
		-- force to lower case
		if force_ascii == True, force convert to ascii"""
    # Force into lowercase.
    string_out = StringProcessor.to_lower_case(s)
    # Remove leading and trailing whitespaces.
    # string_out = StringProcessor.strip(string_out)
    return StringProcessor.strip(string_out)  ## 直接返回


def pos_in_tokens(target_str, tokens, type=None, header=None):
    if not tokens:
        return -1, -1
    tlen = len(target_str)
    copy_target_str = target_str
    q = ''.join(tokens).replace('##', '')
    header = ''.join(header).replace('##', '').replace('[UNK]', '')
    ngrams = []
    for l in range(max(1, tlen - 25), min(tlen + 5, len(tokens))):
        ngrams.append(l)
    candidates = {}
    unit_flag = 0
    tback_flag = 0
    unit_r = 0
    if type == 'real':
        units = re.findall(r'[(（-](.*)', str(header))
        if units:
            unit = units[0]
            # unit_keys = re.findall(r'[百千万亿]{1,}',str(header))
            unit_keys = re.findall(r'百万|千万|万|百亿|千亿|万亿|亿', unit)
            unit_other = re.findall(r'元|米|平|套|枚|册|张|辆|个|股|户|m²|亩|人', unit)
            if unit_keys:
                unit_flag = 1  # col中有[万|百万|千万|亿]单位，
                unit_key = unit_keys[0]
                v, unit_r = chs_to_digits(unit_key)
                # print('--unit--',unit_key, target_str)
                target_str = target_str + unit_key
                target_str = string_PreProcess(target_str)
                target_str = unit_transform(target_str)
            # print('--target_str--', target_str, header)
            elif unit_other:
                unit_flag = 2  # col中有[元|米|平] 单位为个数
            else:
                unit_flag = 3  # 无单位，可能为个数，可能与ques中单位相同
    for l in ngrams:
        cur_idx = 0
        while cur_idx <= len(tokens) - l:
            cur_str = []
            st, ed = cur_idx, cur_idx
            i = st
            while i != len(tokens) and len(cur_str) < l:
                cur_tok = tokens[i]
                cur_str.append(cur_tok)
                i += 1
                ed = i
            cur_str = ''.join(cur_str)
            if '##' in cur_str:
                cur_str = cur_str.replace('##', '')
            if '[UNK]' in cur_str:
                cur_str = cur_str.replace('[UNK]', '')
            if '-' in cur_str:
                cur_str = cur_str.replace('-', '')

            if unit_flag == 1:
                if cur_str == target_str:  # ques 无单位 默认为个数 target_str为unit_convert()后的
                    cur_str = str(int(cur_str) / unit_r)
                    unit_flag = 0  # target_str回到初始状态，
                    tback_flag = 0
                # elif cur_str == copy_target_str: #ques 无单位 默认与target_str 相同
                # 	tback_flag = 1 #标志位 默认与target_str 单位相同
                else:
                    cur_str = cur_str_(cur_str)

            elif unit_flag == 2:
                cur_str = cur_str_(cur_str)
            elif unit_flag == 3:
                if unit_transform(cur_str) == target_str:
                    cur_str = cur_str_(cur_str)
            if type == 'text':
                for item in list(thesaurus_dic.keys()):
                    if item in cur_str:
                        cur_str_the = re.sub(item, thesaurus_dic[item], cur_str)
                        candidates[cur_str_the] = (st, ed)
            candidates[cur_str] = (st, ed)
            cur_idx += 1
    # if tback_flag:
    # 	target_str = copy_target_str

    if list(candidates.keys()) is None or len(list(candidates.keys())) == 0:
        # print('-----testnone----',target_str, tokens,ngrams)
        return -1, -1

    target_str = str(target_str).replace('-', '')
    resultsf = process.extract(target_str, list(candidates.keys()), limit=10, processor=my_process, scorer=my_scorer)
    results = extact_sort(target_str, list(candidates.keys()), limit=10)
    if not results or not resultsf:
        return -1, -1
    dchosen, dcscore = results[0]
    fchosen, fcscore = resultsf[0]
    if fcscore > dcscore:
        cscore = fcscore
        chosen = fchosen
    else:
        cscore = dcscore
        chosen = dchosen

    if cscore != 100:
        pass
    if cscore <= 50:
        q = ''.join(tokens).replace('##', '')
        score = '%d' % (cscore)
    return candidates[chosen]

def cur_str_(cur_str):
    cur_str = unit_transform(cur_str)
    return cur_str


def justify_col_type(table):
	def get_real_col_type(col_idx):
		ret_type = 'text'
		if 'rows' not in table.keys():
			if 'types' in table.keys():
				ret_type = table['types'][col_idx]
		else:
			na_set = {'None', 'none', 'N/A', '', 'nan', '-', '/', 'NAN'}
			col_data = list(filter(lambda x: x not in na_set, [r[col_idx] for r in table['rows']]))
			if col_data:
				isreal = True
				try:
					_ = list(map(float, col_data))
				except:
					isreal = False
				if isreal:
					ret_type = 'real'
				if ('ISBN' in table['header'][col_idx]) or ('号' in table['header'][col_idx]) or ('ID' in table['header'][col_idx]):
					ret_type = 'text'
				# if ret_type != table['types'][col_idx]:
				# 	print(table['header'][col_idx], col_data)

		return ret_type

	if 'types' not in table.keys():
		table['types'] = ['text'] * len(table['header'])
	for i in range(len(table['header'])):
		table['types'][i] = get_real_col_type(i)
	return table


def loading_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths,)
    if not isinstance(table_paths, list):
        table_paths = (table_paths,)
    sql_data = []
    table_data = {}

    for SQL_PATH in sql_paths:
        with open(SQL_PATH, encoding='utf-8') as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                if use_small and idx >= 1000:
                    break
                sql_data.append(sql)
        print("Loaded %d data from %s" % (len(sql_data), SQL_PATH))

    for TABLE_PATH in table_paths:
        with open(TABLE_PATH, encoding='utf-8') as inf:
            for line in inf:
                tab = json.loads(line.strip())
                tab = justify_col_type(tab)
                table_data[tab[u'id']] = tab
        print("Loaded %d data from %s" % (len(table_data), TABLE_PATH))

    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:
            ret_sql_data.append(sql)

    return ret_sql_data, table_data


def loading_dataset(data_dir='../data', toy=False, use_small=False, mode='train'):
    print("Loading dataset")
    import os.path as osp
    data_dirs = {}
    for name in ['train', 'val', 'test']:
        data_dirs[name] = {}
        data_dirs[name]['data'] = osp.join(data_dir, name, name + '.json')
        data_dirs[name]['tables'] = osp.join(data_dir, name, name + '.tables.json')
        data_dirs[name]['db'] = osp.join(data_dir, name, name + '.db')

    dev_sql, dev_table = loading_data(data_dirs['val']['data'], data_dirs['val']['tables'], use_small=use_small)
    dev_db = data_dirs['val']['db']
    if mode == 'train':
        train_sql, train_table = loading_data(data_dirs['train']['data'], data_dirs['train']['tables'],
                                              use_small=use_small)
        train_db = data_dirs['train']['db']
        return train_sql, train_table, train_db, dev_sql, dev_table, dev_db
    elif mode == 'test':
        test_sql, test_table = loading_data(data_dirs['test']['data'], data_dirs['test']['tables'], use_small=use_small)
        test_db = data_dirs['test']['db']
        return dev_sql, dev_table, dev_db, test_sql, test_table, test_db


def batch_seq(sql_data, table_data, idxes, st, ed, tokenizer=None, ret_vis_data=False):
    q_seq = []  # 问题内容
    col_seq = []  # 一张表所有表头
    col_num = []  # 表头数量
    ans_seq = []  # sql的答案列表
    gt_cond_seq = []  # 条件列--列号，类型，值
    vis_seq = []  # （）tuple，问题和对应表所有表头
    sel_num_seq = []  # sel列的数量
    header_type = []  # 对应表所有列的数据类型
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        sel_num = len(sql['sql']['sel'])
        sel_num_seq.append(sel_num)
        conds_num = len(sql['sql']['conds'])

        if tokenizer:
            q = tokenizer.tokenize(string_PreProcess(sql['question']))
            col = [tokenizer.tokenize(header) for header in table_data[sql['table_id']]['header']]

        else:
            q = [char for char in sql['question']]
            col = [[char for char in header] for header in table_data[sql['table_id']]['header']]
        q_seq.append(q)
        col_seq.append(col)
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append(
            (
                len(sql['sql']['agg']),
                sql['sql']['sel'],
                sql['sql']['agg'],
                conds_num,
                tuple(x[0] for x in sql['sql']['conds']),
                tuple(x[1] for x in sql['sql']['conds']),
                sql['sql']['cond_conn_op'],
            ))
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'], table_data[sql['table_id']]['header']))
        header_type.append(table_data[sql['table_id']]['types'])
    # q_seq: char-based sequence of question
    # gt_sel_num: number of selected columns and aggregation functions
    # col_seq: char-based column name
    # col_num: number of headers in one table
    # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
    # gt_cond_seq: ground truth of conds
    if ret_vis_data:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, header_type, vis_seq
    else:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, header_type


def pad_batch_seqs(seqs, pad=None, max_len=None):
    if not max_len:
        max_len = max([len(s) for s in seqs])
    if not pad:
        pad = 0
    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            seqs[i] = seqs[i][:max_len]
        else:
            seqs[i].extend([pad] * (max_len - len(seqs[i])))

    return seqs


def gen_batch_bert_seq(tokenizer, q_seq, col_seq, header_type, max_len=230):
    input_seq = []  # 输入编号
    q_mask = []  # NL mask
    sel_col_mask = []  # columns mask
    sel_col_index = []  # columns starting index
    where_col_mask = []
    where_col_index = []
    token_type_ids = []  # sentence A/B
    attention_mask = []  # length mask

    col_end_index = []

    q_lens = []
    sel_col_nums = []
    where_col_nums = []

    batch_size = len(q_seq)
    for i in range(batch_size):
        text_a = ['[CLS]'] + q_seq[i] + ['[SEP]']
        text_b = []
        for col_idx, col in enumerate(col_seq[i]):
            new_col = []
            if header_type[i][col_idx] == 'text':
                type_token1 = '[unused1]'
                type_token2 = '[unused4]'
                type_token3 = '[unused7]'
            elif header_type[i][col_idx] == 'real':
                type_token1 = '[unused2]'
                type_token2 = '[unused5]'
                type_token3 = '[unused8]'
            else:
                type_token1 = '[unused3]'
                type_token2 = '[unused6]'
                type_token3 = '[unused9]'
            new_col.extend(col)
            new_col.append(type_token2)  # type特征 用来分类第一次作为条件
            new_col.append(type_token3)  # type特征 用来分类第二次作为条件
            # TODO: 可以再加入新的标签来支持更多的列
            new_col.append(type_token1)  # type特征 用来分类sel, 同时分隔列名

            if len(text_a) + len(text_b) + len(new_col) >= max_len:
                break
            text_b.extend(new_col)

        text_b.append('[SEP]')

        inp_seq = text_a + text_b
        input_seq.append(inp_seq)
        q_mask.append([1] * (len(text_a) - 2))
        q_lens.append(len(text_a) - 2)
        token_type_ids.append([0] * len(text_a) + [1] * len(text_b))
        attention_mask.append([1] * len(inp_seq))

        sel_col = []
        where_col = []
        col_ends = []
        for i in range(len(text_a) - 1, len(inp_seq)):
            if inp_seq[i] in ['[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]']:
                where_col.append(i)
            if inp_seq[i] in ['[unused1]', '[unused2]', '[unused3]']:
                sel_col.append(i)
                col_ends.append(i)

        sel_col_mask.append([1] * len(sel_col))
        where_col_mask.append([1] * len(where_col))
        sel_col_nums.append(len(sel_col))
        where_col_nums.append(len(where_col))
        sel_col_index.append(sel_col)
        where_col_index.append(where_col)
        col_end_index.append(col_ends)

    # 规范输入为同一长度，pad = ’[pad]‘ | 0
    input_seq = pad_batch_seqs(input_seq, '[PAD]')
    input_seq = [tokenizer.convert_tokens_to_ids(sq) for sq in input_seq]  # 字符token转化为词汇表里的编码id
    q_mask = pad_batch_seqs(q_mask)
    sel_col_mask = pad_batch_seqs(sel_col_mask)
    sel_col_index = pad_batch_seqs(sel_col_index)
    where_col_mask = pad_batch_seqs(where_col_mask)
    where_col_index = pad_batch_seqs(where_col_index)
    token_type_ids = pad_batch_seqs(token_type_ids)
    attention_mask = pad_batch_seqs(attention_mask)
    col_end_index = pad_batch_seqs(col_end_index)
    return (input_seq, q_mask, sel_col_mask, sel_col_index, where_col_mask, where_col_index, col_end_index,
            token_type_ids, attention_mask), q_lens, sel_col_nums, where_col_nums


def to_batch_seq_test(sql_data, table_data, idxes, st, ed, tokenizer=None):
    q_seq = []
    col_seq = []
    col_num = []
    raw_seq = []
    table_ids = []
    header_type = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]

        if tokenizer:
            q = tokenizer.tokenize(string_PreProcess(sql['question']))
            col = [tokenizer.tokenize(header) for header in table_data[sql['table_id']]['header']]
        else:
            q = [char for char in sql['question']]
            col = [[char for char in header] for header in table_data[sql['table_id']]['header']]
        q_seq.append(q)
        col_seq.append(col)
        col_num.append(len(table_data[sql['table_id']]['header']))
        raw_seq.append(sql['question'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
        header_type.append(table_data[sql['table_id']]['types'])
    return q_seq, col_seq, col_num, raw_seq, table_ids, header_type


def generate_gt_where_seq_test(q, gt_cond_seq):
    ret_seq = []
    for cur_q, ans in zip(q, gt_cond_seq):
        temp_q = u"".join(cur_q)
        cur_q = [u'<BEG>'] + cur_q + [u'<END>']
        record = []
        record_cond = []
        for cond in ans:
            if cond[2] not in temp_q:
                record.append((False, cond[2]))
            else:
                record.append((True, cond[2]))
        for idx, item in enumerate(record):
            temp_ret_seq = []
            if item[0]:
                temp_ret_seq.append(0)
                temp_ret_seq.extend(list(range(temp_q.index(item[1]) + 1, temp_q.index(item[1]) + len(item[1]) + 1)))
                temp_ret_seq.append(len(cur_q) - 1)
            else:
                temp_ret_seq.append([0, len(cur_q) - 1])
            record_cond.append(temp_ret_seq)
        ret_seq.append(record_cond)
    return ret_seq


def gen_bert_labels(q_seq, q_lens, sel_col_nums, where_col_nums, ans_seq, gt_cond_seq, header_type, col_seq):
    q_max_len = max(q_lens)
    sel_col_max_len = max(sel_col_nums)
    where_col_max_len = max(where_col_nums)  # 2col

    # labels init
    where_conn_label = np.array([x[6] for x in ans_seq])  # (None, )
    sel_num_label = np.array([0 for _ in ans_seq])  # (None, )
    where_num_label = np.array([0 for _ in ans_seq])  # (None, )
    sel_col_label = np.array([[0] * sel_col_max_len for _ in ans_seq], dtype=np.float)  # (None, col_max_len)
    sel_agg_label = np.array([[-1] * sel_col_max_len for _ in ans_seq])  # (None, col_max_len)
    where_col_label = np.array([[0] * where_col_max_len for _ in ans_seq], dtype=np.float)  # (None, 2col_max_len)
    where_op_label = np.array([[-1] * where_col_max_len for _ in ans_seq])  # (None, 2col_max_len)

    where_start_label = np.array([[-1] * where_col_max_len for _ in ans_seq])
    where_end_label = np.array([[-1] * where_col_max_len for _ in ans_seq])
    for b in range(len(gt_cond_seq)):  # batch_size
        num_conds = len(gt_cond_seq[b])  # 条件数量
        if num_conds == 0:
            where_col_label[b] = 1.0 / sel_col_nums[b]  # 分散
            mass = 0
        else:
            mass = 1 / num_conds
        col_cond_count = {}
        for cond in gt_cond_seq[b]:
            if cond[0] >= sel_col_nums[b]:
                continue

            if cond[0] in col_cond_count:
                col_cond_count[cond[0]] += 1
            else:
                col_cond_count[cond[0]] = 0

            col_idx = 2 * cond[0] + col_cond_count[cond[0]] % 2
            where_op_label[b][col_idx] = cond[1]
            where_col_label[b][col_idx] += mass
            s, e = pos_in_tokens(cond[2], q_seq[b], header_type[b][cond[0]], col_seq[b][cond[0]])
            if s >= 0:
                s = min(s, q_lens[b] - 1)
                e = min(e - 1, q_lens[b] - 1)
                where_start_label[b][col_idx] = s
                where_end_label[b][col_idx] = e

        if num_conds > 0:
            where_num_label[b] = (where_col_label[b] > 0).sum()

        for b in range(len(ans_seq)):
            _sel = ans_seq[b][1]
            _agg = ans_seq[b][2]
            sel, agg = [], []
            for i in range(len(_sel)):
                if _sel[i] < sel_col_nums[b]:
                    sel.append(_sel[i])
                    agg.append(_agg[i])
            sel_num_label[b] = len(sel)
            mass = 1 / sel_num_label[b]
            if sel_num_label[b] == 0:
                mass = 1 / sel_col_nums[b]
            sel_col_label[b][sel] = mass
            sel_agg_label[b][sel] = agg

    return where_conn_label, sel_num_label, where_num_label, sel_col_label, sel_agg_label, \
           where_col_label, where_op_label, where_start_label, where_end_label


def to_batch_query(sql_data, idxes, st, ed):
	query_gt = []
	table_ids = []
	for i in range(st, ed):
		sql_data[idxes[i]]['sql']['conds'] = sql_data[idxes[i]]['sql']['conds']
		query_gt.append(sql_data[idxes[i]]['sql'])
		table_ids.append(sql_data[idxes[i]]['table_id'])
	return query_gt, table_ids


def epoch_train(model, optimizer, batch_size, sql_data, table_data, tokenizer=None):
	model.train()
	perm = np.random.permutation(len(sql_data))
	cum_loss = 0.0
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		st = st * batch_size
		if isinstance(model, SQLBert):
			# bert training
			q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type = batch_seq(sql_data, table_data,
																								  perm, st, ed,
																								  tokenizer=tokenizer)

			bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																				   header_type)
			logits = model.forward(bert_inputs)  # condconn_logits, condop_logits, sel_agg_logits, q2col_logits

			# gen label
			labels = gen_bert_labels(q_seq, q_lens, sel_col_nums, where_col_nums, ans_seq, gt_cond_seq, header_type, col_seq)
			# q_seq  (12,q_lens) 问题内容
			# q_lens  (12,1)问题长度
			# sel_col_nums (12,1) col 长度
			# where_col_nums (12,1)2col长度
			# ans_seq   [(1, [6], [0], 1, (1,), (2,), 0),] len(agg),sel_col,agg,len(con),con_col,con_type,con_op
			# gt_cond_seq (12,3)条件列--列号，类型，值

			# compute loss
			loss = model.loss(logits, labels, q_lens, sel_col_nums)
		else:

			q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type = batch_seq(sql_data, table_data,
																								  perm, st, ed)
			# q_seq: char-based sequence of question
			# gt_sel_num: number of selected columns and aggregation functions
			# col_seq: char-based column name
			# col_num: number of headers in one table
			# ans_seq: (sel, number of conds, sel list in conds, op list in conds)
			# gt_cond_seq: ground truth of conds
			gt_where_seq = generate_gt_where_seq_test(q_seq, gt_cond_seq)
			gt_sel_seq = [x[1] for x in ans_seq]
			score = model.forward(q_seq, col_seq, col_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq,
								  gt_sel=gt_sel_seq,
								  gt_sel_num=gt_sel_num)
			# sel_num_score, sel_col_score, sel_agg_score, cond_score, cond_rela_score

			# compute loss
			loss = model.loss(score, ans_seq, gt_where_seq)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		cum_loss += loss.data.cpu().numpy() * (ed - st)
	return cum_loss / len(sql_data)


def get_sql(model, batch_size, sql_data, table_data, db_path, tokenizer=None):
	engine = DBEngine(db_path)
	model.eval()
	perm = list(range(len(sql_data)))
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		st = st * batch_size
		with torch.no_grad():
			if isinstance(model, SQLBert):
				q_seq, col_seq, col_num, raw_q_seq, table_ids, header_type = to_batch_seq_test(sql_data, table_data,
																							   perm, st, ed,
																							   tokenizer=tokenizer)

				bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																					   header_type)
				score = model.forward(bert_inputs, re_turn=False)
				sql_preds = model.query(score, q_seq, col_seq, sql_data, table_data, perm, st, ed)
			else:
				q_seq, col_seq, col_num, raw_q_seq, table_ids, header_type = to_batch_seq_test(sql_data, table_data,
																							   perm, st, ed)
				score = model.forward(q_seq, col_seq, col_num)
				sql_preds = model.query(score, q_seq, col_seq, raw_q_seq)
			sql_preds = post_process(sql_preds, sql_data, table_data, perm, st, ed)
			sql_gt = sql_preds[0]
			sql = engine.sql_get(table_ids, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
	return sql

def answer(model, batch_size, sql_data, table_data, tid, db_path, tokenizer=None):
	engine = DBEngine(db_path)
	model.eval()
	perm = list(range(len(sql_data)))
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		st = st * batch_size
		with torch.no_grad():
			if isinstance(model, SQLBert):
				q_seq, col_seq, col_num, raw_q_seq, table_ids, header_type = to_batch_seq_test(sql_data, table_data,
																							   perm, st, ed,
																							   tokenizer=tokenizer)

				bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																					   header_type)
				score = model.forward(bert_inputs, re_turn=False)
				sql_preds = model.query(score, q_seq, col_seq, sql_data, table_data, perm, st, ed)
			else:
				q_seq, col_seq, col_num, raw_q_seq, table_ids, header_type = to_batch_seq_test(sql_data, table_data,
																							   perm, st, ed)
				score = model.forward(q_seq, col_seq, col_num)
				sql_preds = model.query(score, q_seq, col_seq, raw_q_seq)
			sql_preds = post_process(sql_preds, sql_data, table_data, perm, st, ed)
			sql_gt = sql_preds[0]
			anwser = engine.anwser_get(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
	return anwser


def epoch_acc(model, batch_size, sql_data, table_data, db_path, tokenizer=None):
	engine = DBEngine(db_path)
	model.eval()
	perm = list(range(len(sql_data)))
	badcase = 0
	one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
	total_error_cases = []
	total_gt_cases = []
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		st = st * batch_size

		q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type, raw_data = \
			batch_seq(sql_data, table_data, perm, st, ed, tokenizer=tokenizer, ret_vis_data=True)
		query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
		# query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
		raw_q_seq = [x[0] for x in raw_data]  # original question

		# try:
		with torch.no_grad():
			if isinstance(model, SQLBert):
				bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																					   header_type)
				score = model.forward(bert_inputs, re_turn=False)
				pred_queries = model.query(score, q_seq, col_seq, sql_data, table_data, perm, st, ed)
			else:
				score = model.forward(q_seq, col_seq, col_num)
				# generate predicted format
				pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq)

		pred_queries_post = copy.deepcopy(pred_queries)
		pred_queries_post = post_process(pred_queries_post, sql_data, table_data, perm, st, ed)
		one_err, tot_err, error_idxs = check_acc(raw_data, pred_queries_post, query_gt)
		error_cases, gt_cases = gen_batch_error_cases(error_idxs, query_gt, pred_queries_post, pred_queries, raw_data)
		total_error_cases.extend(error_cases)
		total_gt_cases.extend(gt_cases)

		# except:
		# 	badcase += 1
		# 	print('badcase', badcase)
		# 	continue
		one_acc_num += (ed - st - one_err)
		tot_acc_num += (ed - st - tot_err)

		# Execution Accuracy
		for sql_gt, sql_pred, tid in zip(query_gt, pred_queries_post, table_ids):
			ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
			try:
				ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'],
										  sql_pred['cond_conn_op'])
			except:
				ret_pred = None
			ex_acc_num += (ret_gt == ret_pred)
	save_error_case(total_error_cases, total_gt_cases)
	print(ret_pred)
	return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)


def epoch_ensemble(model, batch_scores, batch_size, sql_data, table_data, tokenizer=None):
	model.eval()
	perm = list(range(len(sql_data)))

	if not batch_scores:
		gen_new = True
		batch_scores = []
	else:
		gen_new = False

	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		batch_id = st
		st = st * batch_size

		q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type, raw_data = \
			batch_seq(sql_data, table_data, perm, st, ed, tokenizer=tokenizer, ret_vis_data=True)

		with torch.no_grad():

			bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																				   header_type)
			score = model.forward(bert_inputs, re_turn=False)

			score = model.gen_ensemble(score, q_seq, col_seq, sql_data, table_data, perm, st, ed)
		if not gen_new:
			for i in range(7):
				batch_scores[batch_id][i] += score[i]
			for item in range(len(score[7])):
				for col in range(len(score[7][item])):
					batch_scores[batch_id][7][item][col] += score[7][item][col]
		else:
			batch_scores.append(score)
	return batch_scores

def epoch_ensemble_test(model, batch_scores, batch_size, sql_data, table_data, tokenizer=None):
	model.eval()
	perm = list(range(len(sql_data)))

	if not batch_scores:
		gen_new = True
		batch_scores = []
	else:
		gen_new = False

	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		batch_id = st
		st = st * batch_size

		q_seq, col_seq, col_num, raw_q_seq, table_ids, header_type = to_batch_seq_test(sql_data, table_data,
																					   perm, st, ed,
																					   tokenizer=tokenizer)

		with torch.no_grad():

			bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																				   header_type)
			score = model.forward(bert_inputs, re_turn=False)

			score = model.gen_ensemble(score, q_seq, col_seq, sql_data, table_data, perm, st, ed)
		if not gen_new:
			for i in range(7):
				batch_scores[batch_id][i] += score[i]
			for item in range(len(score[7])):
				for col in range(len(score[7][item])):
					batch_scores[batch_id][7][item][col] += score[7][item][col]
		else:
			batch_scores.append(score)
	return batch_scores

def ensemble_predict(batch_scores, batch_size, sql_data, table_data, output_path):
	perm = list(range(len(sql_data)))
	fw = open(output_path, 'w', encoding='utf-8')
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		batch_idx = st
		st = st * batch_size

		score = batch_scores[batch_idx]
		sql_preds = gen_ensemble_query(score, sql_data, table_data, perm, st, ed)
		sql_preds = post_process(sql_preds, sql_data, table_data, perm, st, ed)
		for sql_pred in sql_preds:
			sql_pred = eval(str(sql_pred))
			fw.writelines(json.dumps(sql_pred, ensure_ascii=False) + '\n')
		# fw.writelines(json.dumps(sql_pred,ensure_ascii=False).encode('utf-8')+'\n')
	fw.close()


def ensemble_acc(batch_scores, batch_size, sql_data, table_data, db_path):
	engine = DBEngine(db_path)
	perm = list(range(len(sql_data)))
	badcase = 0
	one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
	total_error_cases = []
	total_gt_cases = []
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		batch_idx = st
		st = st * batch_size
		query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
		raw_data = [(sql_data[i]['question'], table_data[sql_data[i]['table_id']]['header']) for i in perm[st: ed]]
		score = batch_scores[batch_idx]
		pred_queries = gen_ensemble_query(score, sql_data, table_data, perm, st, ed)

		pred_queries_post = copy.deepcopy(pred_queries)
		pred_queries_post = post_process(pred_queries_post, sql_data, table_data, perm, st, ed)
		one_err, tot_err, error_idxs = check_acc(None, pred_queries_post, query_gt)
		error_cases, gt_cases = gen_batch_error_cases(error_idxs, query_gt, pred_queries_post, pred_queries, raw_data)
		total_error_cases.extend(error_cases)
		total_gt_cases.extend(gt_cases)

		# except:
		# 	badcase += 1
		# 	print('badcase', badcase)
		# 	continue
		one_acc_num += (ed - st - one_err)
		tot_acc_num += (ed - st - tot_err)

		# Execution Accuracy
		for sql_gt, sql_pred, tid in zip(query_gt, pred_queries_post, table_ids):
			ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
			try:
				ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'],
										  sql_pred['cond_conn_op'])
			except:
				ret_pred = None
			ex_acc_num += (ret_gt == ret_pred)
	save_error_case(total_error_cases, total_gt_cases)
	return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)


def post_process(pred, sql_data, table_data, perm, st, ed):
	for i in range(st, ed):
		sql = sql_data[perm[i]]
		table = table_data[sql['table_id']]
		for c in range(len(pred[i - st]['conds'])):
			col_idx = pred[i - st]['conds'][c][0]
			col_val = pred[i - st]['conds'][c][2]
			if col_idx > len(table['header']) or col_val == "" or table['types'][col_idx] == 'real':
				continue

			col_data = []
			for r in table['rows']:
				if col_idx < len(r) and r[col_idx] not in {'None', 'none'}:#, 'N/A', '-', '/', ''}:
					col_data.append(r[col_idx])
			if not col_data:
				continue

			is_real = True
			try:
				_ = list(map(float, col_data))
			except:
				is_real = False
			if is_real:
				continue

			score_c = 0
			for item in list(thesaurus_dic.keys()):
				if item in col_val:
					col_val_the = re.sub(item, thesaurus_dic[item], col_val)
					match_c, score_c = process.extractOne(col_val_the, col_data, processor=my_process)

			match, score = process.extractOne(col_val, col_data, processor=my_process)
			if score_c > score:
				match = match_c
				score = score_c

			if score < 30:
				continue
			pred[i - st]['conds'][c][2] = match
	return pred


def check_acc(vis_info, pred_queries, gt_queries):

	tot_err = sel_num_err = agg_err = sel_err = 0.0
	cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
	bad_sample_idxs = []
	for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
		good = True
		sel_pred, agg_pred, where_rela_pred = pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op']
		sel_gt, agg_gt, where_rela_gt = gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op']

		if where_rela_gt != where_rela_pred:
			good = False
			cond_rela_err += 1

		if len(sel_pred) != len(sel_gt):
			good = False
			sel_num_err += 1

		pred_sel_dict = {k: v for k, v in zip(list(sel_pred), list(agg_pred))}
		gt_sel_dict = {k: v for k, v in zip(sel_gt, agg_gt)}
		if set(sel_pred) != set(sel_gt):
			good = False
			sel_err += 1
		agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
		agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
		if agg_pred != agg_gt:
			good = False
			agg_err += 1

		cond_pred = list(sorted(pred_qry['conds'], key=lambda x: (x[0], x[1], x[2])))
		cond_gt = list(sorted(gt_qry['conds'], key=lambda x: (x[0], x[1], x[2])))
		if len(cond_pred) != len(cond_gt):
			good = False
			cond_num_err += 1
		else:
			cond_op_pred, cond_op_gt = {}, {}
			cond_val_pred, cond_val_gt = {}, {}
			for p, g in zip(cond_pred, cond_gt):
				cond_op_pred[p[0]] = p[1]
				cond_val_pred[p[0]] = p[2]
				cond_op_gt[g[0]] = g[1]
				cond_val_gt[g[0]] = g[2]

			if set(cond_op_pred.keys()) != set(cond_op_gt.keys()):
				cond_col_err += 1
				good = False

			where_op_pred = [cond_op_pred[x] for x in sorted(cond_op_pred.keys())]
			where_op_gt = [cond_op_gt[x] for x in sorted(cond_op_gt.keys())]
			if where_op_pred != where_op_gt:
				cond_op_err += 1
				good = False

			where_val_pred = [cond_val_pred[x] for x in sorted(cond_val_pred.keys())]
			where_val_gt = [cond_val_gt[x] for x in sorted(cond_val_gt.keys())]
			if where_val_pred != where_val_gt:
				cond_val_err += 1
				good = False

		if not good:
			tot_err += 1
			bad_sample_idxs.append(b)
	return np.array(
		(sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err, cond_rela_err)), tot_err, bad_sample_idxs




def gen_batch_error_cases(error_idxs, query_gt, pred_queries_post, pred_queries, raw_data):
    error_cases = []
    gt_cases = []
    for idx in error_idxs:
        single_error = {}
        single_gt = {}
        single_error['q_raw'] = raw_data[idx][0]
        single_error['q_normed'] = string_PreProcess(single_error['q_raw'])
        single_gt['q_raw'] = raw_data[idx][0]
        single_gt['q_normed'] = string_PreProcess(single_error['q_raw'])
        single_error['cols'] = raw_data[idx][1]
        single_gt['cols'] = raw_data[idx][1]
        single_gt['sql'] = query_gt[idx]
        single_error['sql'] = copy.deepcopy(pred_queries_post[idx])
        for i in range(len(single_error['sql']['conds'])):
            single_error['sql']['conds'][i].append(pred_queries[idx]['conds'][i][2])
        error_cases.append(single_error)
        gt_cases.append(single_gt)
    return error_cases, gt_cases


def save_error_case(error_case, gt_cases, dir='../log/'):
    import os.path as osp
    error_fn = osp.join(dir, 'error_cases.json')
    gt_fn = osp.join(dir, 'gt_cases.json')
    with open(error_fn, "w", encoding='utf-8') as f:
        json.dump(error_case, f, ensure_ascii=False, indent=4)
    with open(gt_fn, "w", encoding='utf-8') as f:
        json.dump(gt_cases, f, ensure_ascii=False, indent=4)


def load_word_emb(file_name):
    print('Loading word embedding from %s' % file_name)
    f = open(file_name)
    ret = json.load(f)
    f.close()
    # ret = {}
    # with open(file_name, encoding='latin') as inf:
    #     ret = json.load(inf)
    #     for idx, line in enumerate(inf):
    #         info = line.strip().split(' ')
    #         if info[0].lower() not in ret:
    #             ret[info[0]] = np.array([float(x) for x in info[1:]])
    return ret

thesaurus_dic = {
	'没有要求': '不限',
	'达标': '合格',
	'不': '否',
	'鄂': '湖北',
	'豫': '河南',
	'皖': '安徽',
	'冀': '河北',
	'inter': '因特尔',
	'samsung': '三星',
	'芒果TV': '湖南卫视',
	'湖南台': '芒果TV',
	'企鹅公司': '腾讯',
	'鹅厂': '腾讯',
	'宁': '南京',
	'Youku': '优酷',
	'荔枝台': '江苏卫视',
	'周一': '星期一',
	'周二': '星期二',
	'周三': '星期三',
	'周四': '星期四',
	'周五': '星期五',
	'周六': '星期六',
	'周日': '星期天',
	'周天': '星期天',
	'电视剧频道': '中央台八套',
	'沪': '上海',
	'闽': '福建',
	'增持': '买入'

}

# diff2
s = "导游（见习）"
w = "见习导游"


def similar_str(s1, s2):  ## string_similar 改为  similar_str
    s2_low = s2.lower()
    score1 = difflib.SequenceMatcher(None, s1, s2).quick_ratio()
    score2 = difflib.SequenceMatcher(None, s1, s2_low).quick_ratio()
    val_max = max(score1, score2)

    return val_max


# target_str, list(candidates.keys()),
def search_abbreviation(w, s, ngram=10):  ##search_abbr 改为 search_abbreviation
    slist = []  ## sl 改为 slist
    wlist = []  ## wl 改为 wlist
    for idx in range(len(s)):
        if w.startswith(s[idx]):
            i = 0
            while i < ngram:
                if idx + i > len(s):
                    break
                word = s[idx:idx + i]
                i += 1
                sc = similar_str(word, w)
                wlist.append(word)
                slist.append(sc)
    return wlist[np.argmax(slist)]


def extact_sort(target, candlist, limit=10):  ## target 改为 goal  limit 改为 lim
    wlist = []
    for item in candlist:
        score = similar_str(target, item) * 100
        wlist.append((item, score))
    return heapq.nlargest(limit, wlist, key=lambda i: i[1]) if limit is not None else sorted(wlist, key=lambda i: i[1],
                                                                                             reverse=True)


def digit_distance_search(target, candidates, limit=10):  ## target 改为 goal  limit 改为 lim
    target = abs(float(target))
    if target == 0:
        target = sum(target, 0.01)  ##target + 0.01改为sum(target,0.01)
    candlist = list(candidates.keys())

    wlist = []
    wlists = []  ## wls  改为 wlists
    score = 0
    for item in candlist:
        try:
            float(item)
            wlist.append(item)

        except ValueError:
            pass

    if len(wlist) > 1:
        for i in range(len(wlist) - 2):
            for j in range(i + 1, len(wlist) - 1):
                try:
                    if candidates[wlist[i]][0] == candidates[wlist[j]][0]:
                        wlists.append(wlist[i])
                except:
                    print('keyerror', candidates[wlist[i]][0], candidates[wlist[j]][0])
                    pass
    wlist = [x for x in wlist if x not in wlists]
    wlists.clear()
    wlistt = []  # wlt  改为 wlistt
    for item in wlist:
        if float(item) != 0:
            score = divide(min(target, abs(float(item))), max(target, abs(float(
                item))))  ##min(target, abs(float(item)))/max(target, abs(float(item)))改为divide(min(target, abs(float(item))),max(target, abs(float(item))))
        else:
            score = divide((float(item) + 0.01), target)
        wlistt.append((score * 100, item))
    return heapq.nlargest(limit, wlistt, key=lambda i: i[1]) if limit is not None else sorted(wlistt,
                                                                                              key=lambda i: i[1],
                                                                                              reverse=True)


def divide(a, b):
    div = (a / b)
    return div


# Db


agg_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
cond_op_dict = {0: ">", 1: "<", 2: "==", 3: "!="}
rela_dict = {0: '', 1: ' AND ', 2: ' OR '}


class DBEngine:
    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def sql_get(self, table_id, sel_index, agg_index, con, con_re):
        table_id = 'Table_{}'.format(table_id)
        # 条件数>1 而 条件关系为''
        if con_re == 0 and len(con) > 1:
            return 'Errorone'
        """ Error1 to Errorone """
        # 选择列或条件列为0
        if len(sel_index) == 0 or len(con) == 0 or len(agg_index) == 0:
            return 'Error2'
        """ Error2 to Errortwo """
        con_re = rela_dict[con_re]
        sel_part = ""  ## select_part to sel_part
        for sel, agg in zip(sel_index, agg_index):
            sel_str = 'col_{}'.format(sel + 1)  ## select_str to sel_str
            agg_str = agg_dict[agg]
            if agg:
                sel_part += '{}({}),'.format(agg_str, sel_str)
            else:
                sel_part += '({}),'.format(sel_str)
        sel_part = sel_part[:-1]

        where_part = []
        for col_index, op, val in con:
            # val = val.encode().decode()
            where_part.append('col_{} {} "{}"'.format(col_index + 1, cond_op_dict[op], val))
        where_part = 'WHERE ' + con_re.join(where_part)

        query = 'SELECT {} FROM {} {}'.format(sel_part, table_id, where_part)
        # print("========================")
        encode_query = query
        return encode_query

    def anwser_get(self, table_id, sel_index, agg_index, con, con_re):
        """
        table_id: id of the queried table.
        select_index: list of selected column index, like [0,1,2]
        aggregation_index: list of aggregation function corresponding to selected column, like [0,0,0], length is equal to select_index
        conditions: [[condition column, condition operator, condition value], ...]
        condition_relation: 0 or 1 or 2
        select_index 改为 sel_index
        aggregation_index 改为 agg_index
        conditions 改为 con
        condition_relation 改为 con_re
        """
        table_id = 'Table_{}'.format(table_id)

        # 条件数>1 而 条件关系为''
        if con_re == 0 and len(con) > 1:
            return 'Error1'
        # 选择列或条件列为0
        if len(sel_index) == 0 or len(con) == 0 or len(agg_index) == 0:
            return 'Error2'

        con_re = rela_dict[con_re]

        sel_part = ""  ## select_part to sel_part
        for sel, agg in zip(sel_index, agg_index):
            sel_str = 'col_{}'.format(sel + 1)  ## select_str to sel_str
            agg_str = agg_dict[agg]
            if agg:
                sel_part += '{}({}),'.format(agg_str, sel_str)
            else:
                sel_part += '({}),'.format(sel_str)
        sel_part = sel_part[:-1]

        where_part = []
        for col_index, op, val in con:
            where_part.append('col_{} {} "{}"'.format(col_index + 1, cond_op_dict[op], val))
        where_part = 'WHERE ' + con_re.join(where_part)

        query = 'SELECT {} FROM {} {}'.format(sel_part, table_id, where_part)
        encode_query = query.encode('utf-8')
        try:
            out = self.conn.query(query).as_dict()
        except:
            return 'Error3'
        answer = [tuple(sorted(i.values(), key=lambda x: str(x))) for i in out]
        return answer

    def execute(self, table_id, sel_index, agg_index, con, con_re):
        """
        table_id: id of the queried table.
        select_index: list of selected column index, like [0,1,2]
        aggregation_index: list of aggregation function corresponding to selected column, like [0,0,0], length is equal to select_index
        conditions: [[condition column, condition operator, condition value], ...]
        condition_relation: 0 or 1 or 2
        select_index 改为 sel_index
        aggregation_index 改为 agg_index
        conditions 改为 con
        condition_relation 改为 con_re
        """
        table_id = 'Table_{}'.format(table_id)

        # 条件数>1 而 条件关系为''
        if con_re == 0 and len(con) > 1:
            return 'Error1'
        # 选择列或条件列为0
        if len(sel_index) == 0 or len(con) == 0 or len(agg_index) == 0:
            return 'Error2'

        con_re = rela_dict[con_re]

        sel_part = ""  ## select_part to sel_part
        for sel, agg in zip(sel_index, agg_index):
            sel_str = 'col_{}'.format(sel + 1)  ## select_str to sel_str
            agg_str = agg_dict[agg]
            if agg:
                sel_part += '{}({}),'.format(agg_str, sel_str)
            else:
                sel_part += '({}),'.format(sel_str)
        sel_part = sel_part[:-1]

        where_part = []
        for col_index, op, val in con:
            where_part.append('col_{} {} "{}"'.format(col_index + 1, cond_op_dict[op], val))
        where_part = 'WHERE ' + con_re.join(where_part)

        query = 'SELECT {} FROM {} {}'.format(sel_part, table_id, where_part)
        # print("========================")
        # encode_query = query.encode('utf-8')
        # print(encode_query)
        # print(encode_query.decode("gbk"))
        try:
            out = self.conn.query(query).as_dict()
        # print(out)

        except:
            return 'Error3'
        # print("========================")
        # result_set = [tuple(set(i.values())) for i in out]
        result_set = [tuple(sorted(i.values(), key=lambda x: str(x))) for i in out]
        return result_set


# lookahead


class Lookahead(Optimizer):
    def __init__(self, op, k=5, alpha=0.5):  ## optimizer 改为 op
        self.optimizer = op
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dictionary = self.optimizer.state_dict()  ## fast_state_dict 改为 fast_state_dictionary
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dictionary["state"]
        param_groups = fast_state_dictionary["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dictionary = {  ## slow_state_dict 改为 slow_state_dictionary
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dictionary = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dictionary)
        self.optimizer.load_state_dict(fast_state_dictionary)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


# strPreProcess

def string_PreProcess(que):  ##strPreProcess  改为 string_PreProcess     question 改为 que
    val = que  ## value 改为 val
    try:
        if re.search(r'为负值|为负|是负', val):
            val = re.sub(r'为负值|为负|是负', '小于0', val)
        if re.search(r'为正值|为正|是正', val):
            val = re.sub(r'为正值|为正|是正', '大于0', val)
        # X.x块钱  X毛钱
        val = val.replace('块钱', '块')
        val = val.replace('千瓦', 'kw')
        val = val.replace('个', '')
        val = val.replace(' ', '')
        money = re.compile(
            r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}点[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}')  ## patten_money 改为 money
        mon = money.findall(val)  ## k  改为 mon
        if mon:
            for item in mon:
                listm = item.split('点')
                front, rf = chs_to_digits(listm[0])
                end, rn = chs_to_digits(listm[1])
                val = str(front) + '.' + str(end)
                val = val.replace(item, val, 1)
        k = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}块[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{,1}')  ## patten_kuai  改为 k
        kk = k.findall(val)  ## km  改为 kk
        if kk:
            for item in kk:
                listm = item.split('块')
                front, rf = chs_to_digits(listm[0])
                end, rn = chs_to_digits(listm[1])
                if end:
                    val = str(front) + '.' + str(end) + '元'
                else:
                    val = str(front) + '元'
                val = val.replace(item, val, 1)
            # value = value.replace('毛钱', '元',)
            # value = value.replace('毛', '元')
        m = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1}毛|[0-9]毛')  ## patten_mao  改为 m
        km = m.findall(val)  ## kmao  改为 km
        if km:
            for item in km:
                strm = item.replace('毛', '')  ## strmao 改为 strm
                valm, rm = chs_to_digits(strm)  ## valmao 改为 valm
                mflo = str(float(valm) / 10) + '元'  ## maoflo 改为 mflo
                val = val.replace(item, mflo, 1)
        val = val.replace('元毛', '元')

        j = re.compile(r'[一二两三四五六七八九123456789]角')  ## patten_jao  改为 j
        kj = j.findall(val)  ## kjao  改为 kj
        if kj:
            for item in kj:
                strj = item.replace('角', '')  ## strjao 改为 strj
                valj, rm = chs_to_digits(strj)  ## valjao 改为 valj

                jflo = str(float(valj) / 10) + '元'  ## jaoflo 改为 jflo
                val = val.replace(item, jflo, 1)

        val = val.replace('元毛', '元')
        #        patten_datec = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百|0|1|2|3|4|5|6|7|8|9]{1,}月[零|一|幺|二|两|三|四|五|六|七|八|九|十|百|0|1|2|3|4|5|6|7|8|9]{,2}')
        #        kmonthday = patten_datec.findall(value)
        #        if kmonthday:
        #            print('kmonthday',kmonthday)

        # 更改中文数字--阿拉伯数字
        mm = re.findall(r'[〇零一幺二两三四五六七八九十百千]{2,}', val)
        if mm:
            for item in mm:
                v, r = chs_to_digits(item)
                if r == 1 and v // 10 + 1 != len(item):
                    v = str(v).zfill(len(item) - v // 10)
                val = val.replace(item, str(v), 1)
        mmd = re.findall(r'123456789千]{2,}', val)
        if mmd:
            for item in mmd:
                v, r = chs_to_digits(item)
                val = val.replace(item, str(v), 1)

        mmm = re.findall(r'[一二两三四五六七八九十123456789]{1,}万[二两三四五六七八九23456789]', val)
        if mmm:
            for item in mmm:
                sv = item.replace('万', '')
                v, r = chs_to_digits(sv)
                val = val.replace(item, str(v * 1000), 1)
            # print('--mmm--',mmm,value)
            # 2万2--22000
        mmw = re.findall(r'[一二两三四五六七八九十]万', val)
        if mmw:
            for item in mmw:
                iv = item.replace('万', '')
                v, r = chs_to_digits(iv)
                val = re.sub(item, str(v) + '万', val)
        '''
        mmw  = re.findall(r'[一幺二两三四五六七八九十]万',value)
        if mmw:
            for item in mmw:
                v, r = chinese_to_digits(item)
                value = re.sub(item, str(v), value)
        mmy = re.findall(r'[一幺二两三四五六七八九十百]亿', value)
        if mmy:
            for item in mmy:
                v, r = chinese_to_digits(item)
                value = re.sub(item, str(v), value)

        mmf = re.findall(r'\d*\.?\d+[百千万亿]{1,}',value)
        if mmf:
            for item in mmf:
                v, r = chinese_to_digits(item)
                v_item = re.sub(r'[百千万亿]{1,}','',item)
                v =float(v_item) * r
                value = re.sub(item, str(v), value)
        '''
        mm2 = re.findall(r'[〇零一幺二两三四五六七八九十百千]{1,}[倍|个|元|人|名|位|周|亿|以上|年|盒|册|天|集|宗]', val)
        if mm2:
            for item in mm2:
                mm22 = re.findall(r'[〇零一幺二两三四五六七八九十百千]{1,}', item)
                for item2 in mm22:
                    v2, r2 = chs_to_digits(item2)
                    itemvalue = item.replace(item2, str(v2), 1)
                val = val.replace(item, itemvalue, 1)

            # 百分之几----\d%
        if re.search(r'百分之', val):
            items = re.findall(r'百分之[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}', val)
            # items= re.findall(r'百分之\d*?}', value)
            if items:
                for item in items:
                    item_t = item.replace('百分之', '')
                    mon, r = chs_to_digits(item_t)
                    item_t = str(mon) + '%'
                    val = re.sub(str(item), str(item_t), val)
                # print('1--',items,value)
            items_two = re.findall(r'百分之\d{1,}\.?\d*', val)
            if items_two:
                for item in items_two:
                    item_t = item.replace('百分之', '') + '%'
                    val = re.sub(str(item), str(item_t), val)
                # print('2--', items_two, value)
        if re.search(r'百分点', val):
            items_we = re.findall(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}.??百分点', val)
            if items_we:
                for item in items_we:
                    item_t = re.sub('.??百分点', '', item)
                    mon, r = chs_to_digits(item_t)
                    item_t = str(mon) + '%'
                    val = re.sub(str(item), str(item_t), val)
                # print('百分点-中',items_we,value)
            items_se = re.findall(r'\d+?\.??\d*.??百分点', val)
            if items_se:
                for item in items_se:
                    item_t = re.sub('.??百分点', '', item) + '%'
                    val = re.sub(str(item), str(item_t), val)
                # print('百分点-ala', items_se, value)

        mm3 = re.findall(r'[大于|小于|前|超过|第|破][〇零一幺二两三四五六七八九十百千]{1,}', val)
        if mm3:
            for item in mm3:
                mm33 = re.findall(r'[〇零一幺二两三四五六七八九十百千]{1,}', item)
                for item2 in mm33:
                    v3, r3 = chs_to_digits(item2)
                    itemvalue = item.replace(item2, str(v3), 1)
                # v, r = chinese_to_digits(item)

                val = val.replace(item, itemvalue, 1)

        mm4 = re.findall(r'[排名|排行|达到|排在|排|列|率]{1,}前[0123456789]{1,}', val)
        if mm4:

            for item in mm4:
                # print('qian_val',item,value)
                v = re.sub(r'[排名|排行|达到|排在|排|列|率]{1,}前', '', item)
                s1 = item.replace('前', '大于', 1)
                vs = s1.replace(v, str(int(v) + 1), 1)
                val = val.replace(item, vs, 1)
            # print('--前n--',item,value)

        # 更改中文年份并补充完整
        pattern_date1 = re.compile(r'(\d{2,4}年)')
        # pattern_date1 = re.compile(r'(.{1}月.{,2})日|号')
        date1 = pattern_date1.findall(val)
        dateList1 = list(set(date1))
        if dateList1:
            for item in dateList1:
                v = string_to_date(item)
                val = re.sub(str(item), str(v), val)

        pattern_date2 = re.compile(r'(\d+)(\-|\.)(\d+)(\-|\.)(\d+)')
        date2 = pattern_date2.findall(val)
        dateList2 = list(set(date2))
        if dateList2:
            for item in dateList2:
                v = string_to_date(item)
                val = re.sub(str(item), str(v), val)
        pattern_date3 = re.compile(
            r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|0|1|2|3|4|5|6|7|8|9]{1,}月[零|一|幺|二|两|三|四|五|六|七|八|九|十|0|1|2|3|4|5|6|7|8|9]{1,2}')
        date3 = pattern_date3.findall(val)
        if date3:
            nflag = 0
            for item in date3:
                listm = item.split('月')
                if listm[0].isdigit():
                    front = listm[0]
                else:
                    front, rf = chs_to_digits(listm[0])
                    nflag = 1
                if listm[1].isdigit():
                    end = listm[1]
                else:
                    end, rn = chs_to_digits(listm[1])
                    nflag = 1
                if nflag:
                    kv = str(front) + '月' + str(end)

                    val = val.replace(item, kv, 1)

        pattern_date4 = re.compile(r'\d*?年[\D]{1}月')
        date4 = pattern_date4.findall(val)
        if date4:
            for item in date4:
                kitem = re.findall(r'([\D]{1})月', item)
                mon, v = chs_to_digits(kitem[0])
                mm = item.replace(kitem[0], str(mon))
                val = re.sub(item, mm, val)

        if re.search(r'1下|1共|.1元股|1线', val):
            val = val.replace('1下', '一下')
            val = val.replace('.1元股', '元一股')
            val = val.replace('1共', '一共')
            val = val.replace('1线', '一线')

    except Exception as exc:
        # print('strPreProcess_error', exc,'--',value)
        pass

    return val


# 汉字数字转阿拉伯数字
def chs_to_digits(uchars_chs):  ## chinese_to_digits 改为 chs_to_digits   uchars_chinese 改为 uchars_chs
    total = 0

    num_tmp = {  ## common_used_numerals_tmp 改为 num_tmp
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '〇': 0,
        '零': 0,
        '一': 1,
        '幺': 1,
        '二': 2,
        '两': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '十': 10,
        '百': 100,
        '千': 1000,
        '万': 10000,
        '百万': 1000000,
        '千万': 10000000,
        '亿': 100000000,
        '百亿': 10000000000
    }
    r = 1  # 表示单位：个十百千...
    try:

        for i in range(len(uchars_chs) - 1, -1, -1):
            # print(uchars_chinese[i])
            val = num_tmp.get(uchars_chs[i])

            if val is not None:
                # print('val', val)
                if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
                    if val > r:
                        r = val
                        total = total + val
                    else:
                        r = r * val
                        # total = total + r * x
                elif val >= 10:
                    if val > r:
                        r = val
                    else:
                        r = r * val
                elif val == 0 and i != 0:
                    r = r * 10
                elif r == 1:
                    total = total + pow(10, len(uchars_chs) - i - 1) * val
                else:
                    total = total + r * val
    except Exception as exc:
        print(uchars_chs)
        print('chinese_to_digits_error', exc)
    return total, r


# 日期字符转日期
def string_to_date(date_str):  ## str_to_date 改为 string_to_date
    try:
        # 是数字 有年月日三位
        date_search = re.search('(\d+)(\-|\.)(\d+)(\-|\.)(\d+)', date_str)
        if date_search:
            year_str = date_search.group(1)
            month_str = date_search.group(3)
            day_str = date_search.group(5)
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_date = '{}-{}-{}日'.format(year_str, month_str, day_str)
            return date_date

        # 是数字 只有年月
        # 辅导公告 默认是月底
        date_search = re.search('(\d+)(\-|\.)(\d+)', date_str)
        if date_search:
            year_str = date_search.group(1)
            month_str = date_search.group(3)
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_date = '%s-%s月' % (year_str, month_str)
            return date_date

        # 以下包含汉字
        date_str = date_str.replace('号', '日')
        # 有年月日三位
        date_search = re.search('(.{2,4})年(.*?)月(.*?)日', date_str)
        if date_search:
            if date_search.group(1).isdigit():  # 不能用isnumeric 汉字一二三四会被认为是数字
                # 只有年月日是汉字 数字还是阿拉伯数字
                year_str = date_search.group(1)
                month_str = date_search.group(2)
                day_str = date_search.group(3)
            # 年份不足4位 把前面的补上
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_str = '%s-%s-%s日' % (year_str, month_str, day_str)
            return date_str

        # 只有两位
        date_search = re.search('(.{2,4})年(.*?)月', date_str)
        if date_search:
            if date_search.group(1).isdigit():
                year_str = date_search.group(1)
                month_str = date_search.group(2)
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_str = '%s-%s月' % (year_str, month_str)
            return date_str
        # 只有一位

        date_search = re.search('(\d{2,4})年', date_str)
        if date_search:
            if date_search.group(1).isdigit():
                year_str = date_search.group(1)
            if len(year_str) == 2 and int(year_str[0]) < 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_str = '%s年' % (year_str)
            return date_str

    except Exception as exc:
        pass
    return None


def unit_transform(ques):  ##  unit_convert 改为 unit_transform
    val = ques  ## value 改为 val
    try:
        mmw = re.findall(r'[一幺二两三四五六七八九十]万', val)
        if mmw:
            for item in mmw:
                v, r = chs_to_digits(item)
                val = re.sub(item, str(v), val)
        mmy = re.findall(r'[一幺二两三四五六七八九十百]亿', val)
        if mmy:
            for item in mmy:
                v, r = chs_to_digits(item)
                val = re.sub(item, str(v), val)

        mmf = re.findall(r'\d*\.?\d+万|\d*\.?\d+百万|\d*\.?\d+千万|\d*\.?\d+亿', val)
        if mmf:

            for item in mmf:
                mmf_v = re.sub(r'万|百万|千万|亿', '', item)
                mmf_r = re.sub(mmf_v, '', item)
                v, r = chs_to_digits(mmf_r)
                # print('dig', mmf,v,'--',r)
                val = re.sub(item, str(int(float(mmf_v) * r)), val)

    except Exception as exc:
        print('unit_convert_error', exc, '---', ques)

    return val


str_test1 = '11和2012年,19年1月7日到十九日周票房超过一千万的影投公司,幺九年一月十四到十九播放数大于三千万的剧集,18年同期'
str_test2 = '市值是不超过百亿元,股价高于十块钱,增长超过两块五,或者上涨幅度大于百分之八的股票'
str_test3 = '涨幅为正，年收益为正值，税后利率不为负，税后利润不为负值的股票'
str_test4 = '2019年第1周超过一千万并且占比高于百分之十的,百分之16，百分之几,百分之92.5,百分之0.2，十五个百分点，八个百分点'
str_test5 = '请问有哪些综艺节目它的收视率超过百分之0.2或者市场的份额超过百分之2的'
str_test6 = '中国国航的指标是什么啊，它的油价汇率不足3.5个百分点'
str_test7 = '你知道零七年五月三号,一五年七月，分别领人名币三千块钱，改革开放三十年,给你十块'
str_test8 = '三块五毛钱，四块五毛钱，三千万，六点五块钱，八点五块钱，五毛钱'
'''
你好啊请问一下上海哪些楼盘的价格在2012年的五月份超过了一万五一平-----你好啊请问一下上海哪些楼盘的价格在2012年的五月份超过了一万51平
请问一下有没有什么股票交易交割高于七块钱一股的-----请问一下有没有什么股票交易交割高于七元一股的
二月四号到十号，排名前十的院线有几个总票房大于四亿的-----2月4号到十号，排名前十的院线有几个总票房大于四亿的
保利地产公司股11年每股盈余超过一元，那它12年的每股盈余又会是多少呀-----保利地产公司股2011年每股盈余超过一元，那它2012年的每股盈余又会是多少呀
想知道有多少家影投公司第四周的票房是超过一千五百万？-----想知道有多少家影投公司第四周的票房是超过一千500万？
我想咨询一下有哪些地产股票股价是不低于十块而且在11年每股税后利润还高于一块一股-----我想咨询一下有哪些地产股票股价是不低于十块而且在2011年每股税后利润还高于1.1元股
贷款年限10年假设降20个基点调整前后是什么情况才能使每月减少还款不足100元-----贷款年限2010年假设降20个基点调整前后是什么情况才能使每月减少还款不足100元

'''


# patten = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}块[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}')
def datacontinous(strcofig):
    question = strcofig
    p = re.compile(
        r'([零一幺二两三四五六七八九十百0123456789]+.?)([零一幺二两三四五六七八九十百0123456789]+.?)到([零一幺二两三四五六七八九十百0123456789]+.?[零一幺二两三四五六七八九十百0123456789]+)')
    plist = p.findall(question)
    if plist:
        for item in plist:
            front = '{}{}'.format(item[0], item[1])
            end = str(item[2])
    # print('---到---',plist,front,end)
    pdig = re.compile(r'([零一幺二两三四五六七八九].?)+')
    plist = pdig.findall(question)
    if plist:
        print('plist--', plist)


''''
with open("F:\\天池比赛\\nl2sql_test_20190618\\test.json", "r", encoding='utf-8') as fr,open("F:\\天池比赛\\nl2sql_test_20190618\\log.txt", "w+", encoding='utf-8') as fw:
    count = 0
    for line in fr.readlines():
        lines = eval(line)
        value_re = strPreProcess(lines['question'])
        value_re = datacontinous(value_re)
        count += 1
        #if value_re != lines['question']:
        #    string = lines['question'] + '-----' + value_re + '\n'
        #    fw.write(str(string))
    print('count',count)

'''

# value_re = strPreProcess(str_test7)
# print('----',value_re)

'''
        if re.search(r'1下|1共|1句|1线|哪1年|哪1天', value):
            value = value.replace('1下', '一下')
            value = value.replace('1句', '一句')
            value = value.replace('1共', '一共')
            value = value.replace('1线', '一线')
            value = value.replace('哪1年', '哪一年')
            value = value.replace('哪1天', '哪一天')
        if re.search(r'1手房|2手房|2线|2办', value):
            value = value.replace('1手房', '一手房')
            value = value.replace('2手房', '二手房')
            value = value.replace('2线', '二线')
            value = value.replace('2办', '两办')
'''

# sqlbert


class SQLBert(BertPreTrainedModel):
	def __init__(self, config, hidden=150, gpu=True, dropout_prob=0.2, bert_cache_dir=None):
		super(SQLBert, self).__init__(config)
		self.OP_SQL_DIC = {0: ">", 1: "<", 2: "==", 3: "!="}
		self.AGG_DIC = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
		self.CONN_DIC = {0: "", 1: "and", 2: "or"}
		self.SQL_TOK = ['WHERE', 'AND', 'OR', '==', '>', '<', '!=']

		self.bert_cache_dir = bert_cache_dir

		self.bert = BertModel(config)
		self.apply(self.init_bert_weights)
		self.bert_hidden_size = self.config.hidden_size
		self.W_w_conn = nn.Linear(self.bert_hidden_size, 3)
		self.W_s_num = nn.Linear(self.bert_hidden_size, 5)
		self.W_w_num = nn.Linear(self.bert_hidden_size, 5)
		self.W_s_col = nn.Linear(self.bert_hidden_size, 1)
		self.W_s_agg = nn.Linear(self.bert_hidden_size, 6)
		self.W_w_col = nn.Linear(self.bert_hidden_size, 1)
		self.W_w_op = nn.Linear(self.bert_hidden_size, 4)

		self.W_q_s = nn.Linear(self.bert_hidden_size, hidden)
		self.W_col_s = nn.Linear(self.bert_hidden_size, hidden)
		self.W_q_e = nn.Linear(self.bert_hidden_size, hidden)
		self.W_col_e = nn.Linear(self.bert_hidden_size, hidden)
		self.W_w_s = nn.Linear(hidden, 1)
		self.W_w_e = nn.Linear(hidden, 1)

		self.dropout = nn.Dropout(dropout_prob)

		self.softmax = nn.Softmax(dim=-1)
		self.log_softmax = nn.LogSoftmax(dim=-1)

		self.kl_loss = nn.KLDivLoss(reduction='batchmean')

		self.gpu = gpu

		if gpu:
			self.cuda()

	def forward(self, inputs, re_turn=True):#return_logits改为re_turn

		input_seq, q_mask, sel_col_mask, sel_col_index, where_col_mask, \
		where_col_index, col_end_index, token_type_ids, attention_mask = self.transform_inputs(inputs, dtype=torch.long)
		out_seq, pooled_output = self.bert(input_seq, token_type_ids, attention_mask, output_all_encoded_layers=False)

		out_seq = self.dropout(out_seq)  # 获取token的output 输出[12, seq_length, 768]
		cls_raw = out_seq[:, 0]
		cls_emb = self.dropout(pooled_output)  # 获取句子的output (12, 768)
		max_qlen = q_mask.shape[-1]
		max_col_num = sel_col_mask.shape[-1]

		qs = out_seq[:, 1:1 + max_qlen]  # (12,max_qlen,768) q_seq改为qs
		sel_col_seq = out_seq.gather(dim=1, index=sel_col_index.unsqueeze(-1).expand(-1, -1,
																					 out_seq.shape[-1]))  # (12,col,768)
		where_col_seq = out_seq.gather(dim=1, index=where_col_index.unsqueeze(-1).expand(-1, -1, out_seq.shape[
			-1]))  # (12,2col,768)

		# B = out_seq.shape[0]
		# out_seq = out_seq.cumsum(dim=1)
		# col_end_index = col_end_index.unsqueeze(-1).expand(-1, -1, out_seq.shape[-1])
		# col_ctx_seq = out_seq.gather(dim=1, index=col_end_index[:, 0:1])  # 之前的和 (None, 1, 768)
		# for i in range(1, col_end_index.shape[1]):
		# 	next_sum = out_seq.gather(dim=1, index=col_end_index[:, i: i+1, :])  #  (None, 1, 768)
		# 	interval = (col_end_index[:, i: i+1] - col_end_index[:, i-1: i]).float().clamp(1.0)
		# 	col_ctx_seq[:, i-1: i, :] = (cls_raw.unsqueeze(1) + next_sum - col_ctx_seq[:, i-1: i, :]) / (interval + 1)
		# 	if i != col_end_index.shape[1] - 1:
		# 		col_ctx_seq = torch.cat([col_ctx_seq, next_sum], dim=1)
		#
		# where_ctx_seq = col_ctx_seq.repeat(1, 1, 2).contiguous().view(B, col_ctx_seq.shape[1], self.bert_hidden_size, 2)
		# where_ctx_seq = where_ctx_seq.permute(0, 3, 1, 2).contiguous().view(B, -1, self.bert_hidden_size)

		# sel_col_seq = (sel_col_seq + col_ctx_seq) / 2
		# where_col_seq = (where_col_seq + where_ctx_seq) / 2

		wcl = self.W_w_conn(cls_emb)  # (12,3)where_conn_logit改为wcl
		snl = self.W_s_num(cls_emb)  # (12,5)sel_num_logit改为snl
		wnl = self.W_w_num(cls_emb)  # (12,5)where_num_logit改为wnl
		scl = self.W_s_col(sel_col_seq).squeeze(-1)  # (12,col)sel_col_logit改为scl
		sal = self.W_s_agg(sel_col_seq)  # (12,col,6)sel_agg_logit改为sal
		wcoll = self.W_w_col(where_col_seq).squeeze(-1)  # (12,2col)where_col_logit改为wcoll
		wol = self.W_w_op(where_col_seq)  # (12,2col,4)where_op_logit改为wol

		q_col_s = F.leaky_relu(
			self.W_q_s(qs).unsqueeze(2) + self.W_col_s(where_col_seq).unsqueeze(1))  # (12, q_max, 2col, 768)
		q_col_e = F.leaky_relu(
			self.W_q_e(qs).unsqueeze(2) + self.W_col_e(where_col_seq).unsqueeze(1))  # (12, q_max, 2col, 768)
		wsl = self.W_w_s(q_col_s).squeeze(-1)  # (12,q_max,2col)where_start_logit改为wsl
		wel = self.W_w_e(q_col_e).squeeze(-1)  # (12,q_max,2col)where_end_logit改为wel

		wcl2, \
		snl2, wnl2, scl2, \
		sal2, wcoll2, wol2, \
		wsl2, wel2 = _get_logits(cls_emb, qs, sel_col_seq, where_col_seq,
														   where_col_seq.shape[1])
		# where_conn_logit2改为wcl2 / sel_num_logit2改为snl2 / where_num_logit2改为wnl2 / sel_col_logit2改为scl2
		# sel_agg_logit2改为sal2 / where_col_logit2改为wcoll2 / where_op_logit2改为wol2 /
		# where_start_logit2改为wsl2 / where_end_logit2改为wel2

		wcl = (wcl + wcl2) / 2
		snl = (snl + snl2) / 2
		wnl = (wnl + wnl2) / 2
		scl = (scl + scl2) / 2
		sal = (sal + sal2) / 2
		wcoll = (wcoll + wcoll2) / 2
		wol = (wol + wol2) / 2
		wsl = (wsl + wsl2) / 2
		wel = (wel + wel2) / 2

		# where_conn_logit, \
		# sel_num_logit, where_num_logit, sel_col_logit, \
		# sel_agg_logit, where_col_logit, where_op_logit, \
		# where_start_logit, where_end_logit = _get_logits(cls_emb, q_seq, sel_col_seq, where_col_seq,
		# 												   where_col_seq.shape[1])

		# 联合概率
		# sel_agg_logit = sel_agg_logit + sel_col_logit[:, :, None]
		# where_op_logit = where_op_logit + where_col_logit[:, :, None]
		# where_start_logit = where_start_logit + where_op_logit.max(-1)[0][:, None, :]
		# where_end_logit = where_end_logit + where_start_logit.max(1)[0][:, None, :]

		# 处理mask, 因为masked_fill要求fill的位置mask为1,保留的位置mask为0

		q_mask, sel_col_mask, where_col_mask = q_mask.bool(), sel_col_mask.bool(), where_col_mask.bool()
		qcol_mask = q_mask.unsqueeze(2) & where_col_mask.unsqueeze(1)  # (12,qen_max,1)  (12,1,2col)
		q_mask, sel_col_mask, where_col_mask, qcol_mask = ~q_mask, ~sel_col_mask, ~where_col_mask, ~qcol_mask
		# do mask

		scl = scl.masked_fill(sel_col_mask, -1e5)  # (123,col)()
		sal = sal.masked_fill(sel_col_mask.unsqueeze(-1).expand(-1, -1, 6), -1e5)
		wcoll = wcoll.masked_fill(where_col_mask, -1e5)
		wol = wol.masked_fill(where_col_mask.unsqueeze(-1).expand(-1, -1, 4), -1e5)
		wsl = wsl.masked_fill(qcol_mask, -1e5)
		wel = wel.masked_fill(qcol_mask, -1e5)

		if re_turn:
			return wcl, \
				   snl, wnl, scl, \
				   sal, wcoll, wol, \
				   wsl, wel
		else:
			return F.softmax(wcl, dim=-1), \
				   F.softmax(snl, dim=-1), \
				   F.softmax(wnl, dim=-1), \
				   F.softmax(scl, dim=-1), \
				   F.softmax(sal, dim=-1), \
				   F.softmax(wcoll, dim=-1), \
				   F.softmax(wol, dim=-1), \
				   F.softmax(wsl, dim=1), \
				   F.softmax(wel, dim=1)

	def loss(self, logits, labels, q_lens, col_nums):

		where_conn_logit, \
		sel_num_logit, where_num_logit, sel_col_logit, \
		sel_agg_logit, where_col_logit, where_op_logit, \
		where_start_logit, where_end_logit = logits

		where_conn_label, sel_num_label, where_num_label, \
		sel_col_label, sel_agg_label, where_col_label, where_op_label, \
		where_start_label, where_end_label = labels

		where_conn_label, sel_num_label, where_num_label, sel_agg_label, where_op_label, where_start_label, where_end_label = self.transform_inputs(
			(where_conn_label, sel_num_label, where_num_label, sel_agg_label, where_op_label, where_start_label,
			 where_end_label), dtype=torch.long)
		sel_col_label, where_col_label = self.transform_inputs((sel_col_label, where_col_label), dtype=torch.float)

		# q_lens, col_nums = self.transform_inputs((q_lens, col_nums))
		# q_lens, col_nums = q_lens.float(), col_nums.float()

		# Evaluate the cond conn type
		where_conn_loss = F.cross_entropy(where_conn_logit, where_conn_label)  # (12,3)  (12,1)
		sel_num_loss = F.cross_entropy(sel_num_logit, sel_num_label)  # (12,5)  (12,1)
		sel_col_loss = torch.abs(self.kl_loss(self.log_softmax(sel_col_logit), sel_col_label.float()))
		# print('--loss--',sel_col_logit.shape,'\n',sel_col_label.shape)
		sel_agg_loss = F.cross_entropy(sel_agg_logit.transpose(1, 2), sel_agg_label, ignore_index=-1)

		where_num_loss = F.cross_entropy(where_num_logit, where_num_label)
		where_col_loss = torch.abs(self.kl_loss(self.log_softmax(where_col_logit), where_col_label.float()))
		where_op_loss = F.cross_entropy(where_op_logit.transpose(1, 2), where_op_label, ignore_index=-1)
		where_start_loss = F.cross_entropy(where_start_logit, where_start_label, ignore_index=-1)
		where_end_loss = F.cross_entropy(where_end_logit, where_end_label, ignore_index=-1)

		loss = where_conn_loss + sel_num_loss + where_num_loss + sel_agg_loss \
			   + where_op_loss + sel_col_loss + where_col_loss + where_start_loss + where_end_loss

		return loss

	def transform_inputs(self, inputs, dtype=torch.long):
		for x in inputs:
			if isinstance(x, (list, tuple)):
				x = np.array(x)
			if self.gpu:
				yield torch.from_numpy(np.array(x)).to(dtype).cuda()
			else:
				yield torch.from_numpy(np.array(x)).to(dtype)

	def query(self, scores, q, col, sql_data, table_data, perm, st, ed, beam=True, k=10):

		valid_s_agg = {
			"real": frozenset([0, 1, 2, 3, 4, 5]),
			"text": frozenset([0, 4])
		}
		valid_w_op = {
			"real": frozenset([0, 1, 2, 3]),
			"text": frozenset([2, 3])
		}

		where_conn_score, \
		sel_num_score, where_num_score, sel_col_score, \
		sel_agg_score, where_col_score, where_op_score, \
		where_start_score, where_end_score = scores

		ret_queries = []
		B = sel_num_score.shape[0]
		col_vtypes = [table_data[sql_data[idx]['table_id']]['types'] for idx in perm[st:ed]]
		raw_q = [sql_data[idx]['question'] for idx in perm[st:ed]]
		table_headers = [table_data[sql_data[idx]['table_id']]['header'] for idx in perm[st:ed]]
		where_conn_pred = where_conn_score.argmax(-1).data.cpu().numpy().tolist()

		## filtering sel col and agg
		sel_x_agg_score = (sel_col_score[:, :, None] * sel_agg_score).view(B, -1)
		sel_x_agg_value = np.array(
			[[c_idx, agg] for c_idx in range(0, sel_col_score.shape[1]) for agg in range(0, 6)]).tolist()
		sel_x_agg_idx = sel_x_agg_score.argsort(dim=-1, descending=True).data.cpu().numpy().tolist()
		sel_num_pred = sel_num_score.argmax(-1).data.cpu().numpy().tolist()
		where_x_op_score = (where_col_score[:, :, None] * where_op_score).view(B, -1)
		where_x_op_value = np.array(
			[[c_idx, op] for c_idx in range(0, where_col_score.shape[1]) for op in range(0, 4)]).tolist()
		where_x_op_idx = where_x_op_score.argsort(dim=-1, descending=True).data.cpu().numpy().tolist()
		where_num_pred = where_num_score.argmax(-1).data.cpu().numpy().tolist()

		where_start_pred = where_start_score.argmax(1).data.cpu().numpy().tolist()
		where_end_pred = where_end_score.argmax(1).data.cpu().numpy().tolist()

		for b in range(B):
			cur_query = {}
			types = col_vtypes[b]

			sel_num = max(1, sel_num_pred[b])
			sel_idx_sorted = sel_x_agg_idx[b]

			cur_query['agg'] = []
			cur_query['cond_conn_op'] = where_conn_pred[b]
			cur_query['sel'] = []
			for idx in sel_idx_sorted:
				if len(cur_query['sel']) == sel_num:
					break
				sel_col_idx, agg = sel_x_agg_value[idx]
				if sel_col_idx < len(col[b]):
					sel_col_type = types[sel_col_idx]
					if beam and (agg not in valid_s_agg[sel_col_type]):
						continue
					cur_query['agg'].append(agg)
					cur_query['sel'].append(sel_col_idx)

			where_num = max(1, where_num_pred[b])
			where_idx_sorted = where_x_op_idx[b]
			cond_candis = []
			cur_query['conds'] = []

			for idx in where_idx_sorted:
				if len(cond_candis) >= where_num:
					break
				w_col_idx, op = where_x_op_value[idx]
				true_col_idx = w_col_idx // 2  # 每列预测两个条件
				if true_col_idx < len(col[b]):
					where_col_type = types[true_col_idx]
					# if beam and (op not in valid_w_op[where_col_type]):
					# 	continue
					cond_start = where_start_pred[b][w_col_idx]
					cond_end = where_end_pred[b][w_col_idx]
					cond_candis.append([w_col_idx, op, cond_start, cond_end])

			cond_candis.sort(key=lambda x: (x[0], x[2]))

			for cond_cand in cond_candis:
				true_col_idx = cond_cand[0] // 2  # 每列预测两个条件
				cons_toks = q[b][cond_cand[2]:cond_cand[3] + 1]
				cond_str = merge_tokens(cons_toks, raw_q[b])
				check = cond_str
				'''type = real 预测的cond_str单位进行转换'''
				# table_headers = table_data[raw_q[b]['table_id']]['header']
				# table_types = table_data[raw_q[b]['table_id']]['types']
				if col_vtypes[b][true_col_idx] == 'real':

					col_header = table_headers[b][true_col_idx]
					# print('----real----', cond_str, col_header)
					units = re.findall(r'[(（/-](.*)', str(col_header))
					# unit_keys = re.findall(r'万|百万|千万|亿', str(col_header))
					if units:
						unit = units[0]
						unit_keys = re.findall(r'百万|千万|万|百亿|千亿|万亿|亿', unit)
						# unit_keys = re.findall(r'[百千万亿]{1,}', unit)
						if unit_keys:
							unit_key = unit_keys[0]
							u_v, r = chs_to_digits(unit_key)

							if re.findall(unit_key, cond_str):
								cond_str = re.sub(unit_key, '', cond_str)
							elif re.findall(r'百万|千万|万|百亿|千亿|万亿|亿', cond_str):
								try:
									cond_v = unit_transform(cond_str)
									cond_str = float(cond_v) / r
								except Exception as exc:
									# print('gen_query_convert', exc, cond_str, r, unit_key)
									pass


						elif re.findall(r'元|米|平|套|枚|册|张|辆|个|股|户|m²|亩|人', unit):
							try:
								cond_str = unit_transform(cond_str)
							except Exception as exc:
								# print('gen_query_convert', exc)
								pass
						else:
							cond_str = re.sub(r'[百千万亿]{1,}', '', str(cond_str))

						cond_str = re.sub(r'[^0123456789.-]', '', str(cond_str))
					else:
						cond_str = re.sub(r'[百千万亿]{1,}', '', cond_str)
						cond_str = re.sub(r'[^0123456789.-]', '', cond_str)
				# print('----real----', cond_str,'***', check, col_header)

				cur_query['conds'].append([true_col_idx, cond_cand[1], cond_str])

			ret_queries.append(cur_query)
		return ret_queries

	def gen_ensemble(self, scores, q, col, sql_data, table_data, perm, st, ed):
		valid_s_agg = {
			"real": frozenset([0, 1, 2, 3, 4, 5]),
			"text": frozenset([0, 4])
		}
		valid_w_op = {
			"real": frozenset([0, 1, 2, 3]),
			"text": frozenset([2, 3])
		}

		where_conn_score, \
		sel_num_score, where_num_score, sel_col_score, \
		sel_agg_score, where_col_score, where_op_score, \
		where_start_score, where_end_score = scores

		where_conn_score = where_conn_score.data.cpu()
		sel_num_score = sel_num_score.data.cpu()
		where_num_score = where_num_score.data.cpu()
		sel_col_score = sel_col_score.data.cpu()
		sel_agg_score = sel_agg_score.data.cpu()
		where_col_score = where_col_score.data.cpu()
		where_op_score = where_op_score.data.cpu()
		where_start_score = where_start_score.data.cpu()
		where_end_score = where_end_score.data.cpu()

		B = sel_num_score.shape[0]

		col_vtypes = [table_data[sql_data[idx]['table_id']]['types'] for idx in perm[st:ed]]
		raw_q = [sql_data[idx]['question'] for idx in perm[st:ed]]
		table_headers = [table_data[sql_data[idx]['table_id']]['header'] for idx in perm[st:ed]]

		where_vals = []

		where_start_maxscore, where_start_pred = where_start_score.max(1)
		where_end_maxscore, where_end_pred = where_end_score.max(1)

		where_start_maxscore = where_start_maxscore.data.cpu().numpy()
		where_end_maxscore = where_end_maxscore.data.cpu().numpy()
		where_start_pred = where_start_pred.data.cpu().numpy()
		where_end_pred = where_end_pred.data.cpu().numpy()
		for b in range(B):
			cur_col_vals = []
			for idx in range(where_start_pred.shape[1]):
				cur_vals = Counter()
				start = where_start_pred[b][idx]
				end = where_end_pred[b][idx]
				cond_score = where_start_maxscore[b][idx] * where_end_maxscore[b][idx]
				true_col_idx = idx // 2
				if true_col_idx >= len(table_headers[b]):
					continue
				cons_toks = q[b][start:end + 1]

				cond_str = merge_tokens(cons_toks, raw_q[b])
				check = cond_str
				'''type = real 预测的cond_str单位进行转换'''

				if col_vtypes[b][true_col_idx] == 'real':

					col_header = table_headers[b][true_col_idx]
					# print('----real----', cond_str, col_header)
					units = re.findall(r'[(（/-](.*)', str(col_header))
					# unit_keys = re.findall(r'万|百万|千万|亿', str(col_header))
					if units:
						unit = units[0]
						unit_keys = re.findall(r'百万|千万|万|百亿|千亿|万亿|亿', unit)
						# unit_keys = re.findall(r'[百千万亿]{1,}', unit)
						if unit_keys:
							unit_key = unit_keys[0]
							u_v, r = chs_to_digits(unit_key)

							if re.findall(unit_key, cond_str):
								cond_str = re.sub(unit_key, '', cond_str)
							elif re.findall(r'百万|千万|万|百亿|千亿|万亿|亿', cond_str):
								try:
									cond_v = unit_transform(cond_str)
									cond_str = float(cond_v) / r
								except Exception as exc:
									# print('gen_query_convert', exc, cond_str, r, unit_key)
									pass

						elif re.findall(r'元|米|平|套|枚|册|张|辆|个|股|户|m²|亩|人', unit):
							try:
								cond_str = unit_transform(cond_str)
							except Exception as exc:
								# print('gen_query_convert', exc)
								pass
						else:
							cond_str = re.sub(r'[百千万亿]{1,}', '', str(cond_str))

						cond_str = re.sub(r'[^0123456789.-]', '', str(cond_str))
					else:
						cond_str = re.sub(r'[百千万亿]{1,}', '', cond_str)
						cond_str = re.sub(r'[^0123456789.-]', '', cond_str)
				# print('----real----', cond_str,'***', check, col_header)
				# if not cond_str:
				# 	cond_str = ''
				cur_vals[cond_str] = cond_score
				cur_col_vals.append(cur_vals)
			where_vals.append(cur_col_vals)
		return [where_conn_score, sel_num_score, where_num_score, sel_col_score,
				sel_agg_score, where_col_score, where_op_score, where_vals]


def gen_ensemble_query(batch_score, sql_data, table_data, perm, st, ed, beam=True, k=10):
	valid_s_agg = {
		"real": frozenset([0, 1, 2, 3, 4, 5]),
		"text": frozenset([0, 4])
	}
	valid_w_op = {
		"real": frozenset([0, 1, 2, 3]),
		"text": frozenset([2, 3])
	}

	where_conn_score, \
	sel_num_score, where_num_score, sel_col_score, \
	sel_agg_score, where_col_score, where_op_score, \
	where_vals = batch_score

	ret_queries = []
	B = sel_num_score.shape[0]
	col_vtypes = [table_data[sql_data[idx]['table_id']]['types'] for idx in perm[st:ed]]
	table_headers = [table_data[sql_data[idx]['table_id']]['header'] for idx in perm[st:ed]]
	where_conn_pred = where_conn_score.argmax(-1).data.cpu().numpy().tolist()

	## filtering sel col and agg
	sel_x_agg_score = (sel_col_score[:, :, None] * sel_agg_score).view(B, -1)
	sel_x_agg_value = np.array(
		[[c_idx, agg] for c_idx in range(0, sel_col_score.shape[1]) for agg in range(0, 6)]).tolist()
	sel_x_agg_idx = sel_x_agg_score.argsort(dim=-1, descending=True).data.cpu().numpy().tolist()
	sel_num_pred = sel_num_score.argmax(-1).data.cpu().numpy().tolist()
	where_x_op_score = (where_col_score[:, :, None] * where_op_score).view(B, -1)
	where_x_op_value = np.array(
		[[c_idx, op] for c_idx in range(0, where_col_score.shape[1]) for op in range(0, 4)]).tolist()
	where_x_op_idx = where_x_op_score.argsort(dim=-1, descending=True).data.cpu().numpy().tolist()
	where_num_pred = where_num_score.argmax(-1).data.cpu().numpy().tolist()

	for b in range(B):
		cur_query = {}
		types = col_vtypes[b]

		sel_num = max(1, sel_num_pred[b])
		sel_idx_sorted = sel_x_agg_idx[b]

		cur_query['agg'] = []
		cur_query['cond_conn_op'] = where_conn_pred[b]
		cur_query['sel'] = []
		for idx in sel_idx_sorted:
			if len(cur_query['sel']) == sel_num:
				break
			sel_col_idx, agg = sel_x_agg_value[idx]
			if sel_col_idx < len(table_headers[b]):
				sel_col_type = types[sel_col_idx]
				if beam and (agg not in valid_s_agg[sel_col_type]):
					continue
				cur_query['agg'].append(agg)
				cur_query['sel'].append(sel_col_idx)

		where_num = max(1, where_num_pred[b])
		where_idx_sorted = where_x_op_idx[b]
		cur_query['conds'] = []

		for idx in where_idx_sorted:
			if len(cur_query['conds']) >= where_num:
				break
			w_col_idx, op = where_x_op_value[idx]
			true_col_idx = w_col_idx // 2  # 每列预测两个条件
			if true_col_idx < len(table_headers[b]):
				where_col_type = types[true_col_idx]
				# if beam and (op not in valid_w_op[where_col_type]):
				# 	continue
				vals = list(sorted(list(where_vals[b][w_col_idx].items()), key=lambda x: x[1], reverse=True))
				cond_str = vals[0][0]

				cur_query['conds'].append([true_col_idx, op, cond_str])

		cur_query['conds'].sort(key=lambda x: x[0])
		ret_queries.append(cur_query)
	return ret_queries


"""
得到where之间关系的分类logit
:arg: cls_emb (None, 768)
:return cls_emb[:, :3] 3种
"""


def _get_where_conn_logit(cls_emb):
	return cls_emb[:, :3]


"""
得到select的列数目对应的logit
:arg: cls_emb (None, 768)
:return cls_emb[:, :8]
"""


def _get_sel_num_logit(cls_emb):
	return cls_emb[:, 3:8]


"""
得到where的列数目对应的logit
:arg cls_emb (None, 768)
:return cls_emb[:, 8:13]
"""


def _get_where_num_logit(cls_emb):
	return cls_emb[:, 8:13]


"""
得到是否是select的列对应的logit
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 0]
"""


def _get_sel_col_logit(col_seq):
	return col_seq[:, :, 0]


"""
得到selected列的聚合函数logit  6种
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 1: 7]
"""


def _get_sel_agg_logit(col_seq):
	return col_seq[:, :, 1: 7]


"""
得到是否是where中的列对应的Logit
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 7]
"""


def _get_where_col_logit(col_seq):
	return col_seq[:, :, 7]


"""
得到where列的operation logit  4种
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 1: 7]
"""


def _get_where_op_logit(col_seq):
	return col_seq[:, :, 8: 12]


"""
得到where列的value start
:args q_seq (None, max-qlen, 768)
	  max_col_num 
:return q_seq[:, :, :max_col_num]
"""


def _get_where_start_logit(q_seq, max_col_num):
	return q_seq[:, :, :max_col_num]


"""
得到where列的value end
:args q_seq (None, max-qlen, 768)
	  max_col_num 
:return q_seq[:, :, 100:max_col_num]
"""


def _get_where_end_logit(q_seq, max_col_num):
	return q_seq[:, :, 100:100 + max_col_num]


def _get_logits(cls_emb, q_seq, sel_col_seq, where_col_seq, max_col_num):
	return _get_where_conn_logit(cls_emb), \
		   _get_sel_num_logit(cls_emb), \
		   _get_where_num_logit(cls_emb), \
		   _get_sel_col_logit(sel_col_seq), \
		   _get_sel_agg_logit(sel_col_seq), \
		   _get_where_col_logit(where_col_seq), \
		   _get_where_op_logit(where_col_seq), \
		   _get_where_start_logit(q_seq, max_col_num), \
		   _get_where_end_logit(q_seq, max_col_num)


def merge_tokens(tok_list, raw_tok_str):
	if not tok_list:
		return ''
	tok_str = raw_tok_str  # .lower()
	alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
	special = {'-LRB-': '(',
			   '-RRB-': ')',
			   '-LSB-': '[',
			   '-RSB-': ']',
			   '``': '"',
			   '\'\'': '"',
			   '--': u'\u2013'}
	ret = ''
	double_quote_appear = 0
	for raw_tok in tok_list:
		if not raw_tok:
			continue
		raw_tok = raw_tok.replace("##", "").replace("[UNK]", "")
		tok = special.get(raw_tok, raw_tok)
		if tok == '"':
			double_quote_appear = 1 - double_quote_appear
		if len(ret) == 0:
			pass
		# elif len(ret) > 0 and ret + ' ' + tok in tok_str:
		# 	ret = ret + ' '
		elif len(ret) > 0 and ret + tok in tok_str:
			pass
		# elif tok == '"':
		# 	if double_quote_appear:
		# 		ret = ret + ' '
		# elif tok[0] not in alphabet:
		#     pass
		# elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
		# 		and (ret[-1] != '"' or not double_quote_appear):
		# ret = ret + ' '
		ret = ret + tok
	return ret.strip()
