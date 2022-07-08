# -*- coding: utf-8 -*-
# +
#coding=utf-8
#创建TCP服务器
from socket import *
from time import ctime
import os
import time
import json

HOST='172.17.0.5' #这个是我的服务器ip，根据情况改动
PORT=26206#我的端口号
BUFSIZ=1024
ADDR=(HOST,PORT)
 
tcpSerSock=socket(AF_INET,SOCK_STREAM) #创服务器套接字
tcpSerSock.bind(ADDR) #套接字与地址绑定
tcpSerSock.listen(5)  #监听连接,传入连接请求的最大数，一般为5就可以了
 
while True:
    answer_list = []
    print('waiting for connection...')
    tcpCliSock,addr =tcpSerSock.accept()
    print('...connected from:',addr)
 
    while True:
        stock_codes = tcpCliSock.recv(BUFSIZ).decode() #收到的客户端的数据需要解码（python3特性）
        print('stock_codes = ',stock_codes)#传入参数stock_codes
        
        with open('../data/final_test_1.json', 'w') as f:
            tid = "c98c2ca1332111e99c14542696d6e445"
            question = stock_codes
            #question = "2012年都有哪些涨价楼盘"
            fi = f'{{"table_id": "{tid}", "question": "{question}"}}'
            f.writelines(str(fi))
            f.close()
        os.system('sh start_test_bert.sh > log.log')
        
        if not stock_codes:
            break
        with open('../data/answer_1.json', 'r') as f:
            sql_answer = json.load(f)
            sql = sql_answer['SQL']
            answer = sql_answer['answer']
            f.close()            
        
        #stock_codes = '我是答案，答案放我这里'
        for item in answer:
            list2 = [str(i) for i in item]
            item = ' '.join(list2)
            answer_list.append(item)
            
        answer_list = ','.join(answer_list) 
        tcpCliSock.send(('[SQL]:\t%s\n\n[Anwser]:\t%s' %(sql,answer_list)).encode())
        #tcpCliSock.send(('[%s] %s' %(ctime(),stock_codes)).encode())  #发送给客户端的数据需要编码（python3特性）
        #after_close_simulation = tcpCliSock.recv(BUFSIZ).decode() #收到的客户端的数据需要解码（python3特性）
        #print('after_close_simulation = ',after_close_simulation)    #传入参数after_close_simulation
        #if not after_close_simulation:
            #break
        #tcpCliSock.send(('[%s] %s' %(ctime(),after_close_simulation)).encode())  #发送给客户端的数据需要编码（python3特性） 
 
    tcpCliSock.close()
tcpSerSock.close()
# -

# !ifconfig

import os
os.system('sh start_test_bert.sh > log.log')


# !top
